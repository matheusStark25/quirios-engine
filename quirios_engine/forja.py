"""
Forja quântica: um circuito quântico desenha a compressão de um modelo de IA.

    pip install -e ".[ia]"
    python -m quirios_engine.forja mnist --alvo 30                       # forja um modelo 30x menor
    python -m quirios_engine.forja mnist --alvo 30 --motor todos          # quântico x clássico x aleatório
    python -m quirios_engine.forja mnist --alvo 30 --modo-gerador ibm     # o gerador é medido numa QPU real

Cada camada grande de um modelo treinado pode virar uma rede tensorial (TT,
com 3, 4 ou 6 núcleos e ligação 2..16), uma fatoração de posto baixo ou
continuar densa. Com várias camadas, as combinações explodem, e a melhor
depende do modelo e da meta (tamanho, acurácia, velocidade).

Quem escolhe é o gerador quântico da evolução do QURIOS (evolucao_quantica):
as combinações ficam em superposição num circuito, cada medida é um projeto
de compressão, os melhores reforçam suas amplitudes e o emaranhamento aprende
quais escolhas de camadas diferentes andam juntas. Cada projeto é avaliado de
verdade: a compressão é aplicada aos pesos treinados, ajustada por alguns
passos e medida num conjunto de validação separado do teste.

O modelo vencedor sai com um COMPROVANTE selado (SHA-256 dos pesos, projeto,
métricas e histórico da busca; assinável com ML-DSA), e o mesmo orçamento de
avaliações é dado a uma busca evolutiva clássica e a uma busca aleatória
para a comparação justa.
"""
import argparse
import copy
import datetime
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np

from .evolucao_quantica import CodificacaoGenoma, GeradorQuantico

NIVEIS_TT = (2, 4, 8, 16)              # ligação da rede tensorial
NIVEIS_POSTO = (8, 32, 64, 128)       # posto da fatoração clássica
NUCLEOS = (3, 4, 6)


class _Codificacao(CodificacaoGenoma):
    """Genoma de qualquer espaço discreto <-> bits (sem os genes de circuito)."""

    def decodificar(self, x):
        g = {}
        for nome, opcoes, qs in self.genes:
            v = sum(((int(x) >> q) & 1) << k for k, q in enumerate(qs))
            g[nome] = opcoes[v % len(opcoes)]
        return g

    def codificar(self, genoma):
        x = 0
        for nome, opcoes, qs in self.genes:
            v = opcoes.index(genoma[nome]) if genoma[nome] in opcoes else 0
            for k, q in enumerate(qs):
                x |= ((v >> k) & 1) << q
        return x


def _chave(genoma):
    return json.dumps(genoma, sort_keys=True)


# ── os três motores de busca, com o mesmo orçamento ─────────────────────────

class Busca:
    """motor: "quantico" (gerador de Born com emaranhamento), "classico"
    (algoritmo genético: seleção, cruzamento e mutação) ou "aleatorio".
    O orçamento é o mesmo: populacao x (geracoes + 1) sorteios."""

    def __init__(self, espaco, avaliar, motor="quantico", populacao=6, geracoes=4, elite=0.34, passos=5,
                 lr=0.25, mutacao=0.08, temperatura=0.15, camadas_gerador=2, backend_gerador=None, seed=0,
                 ao_progresso=None, canonico=None):
        if motor not in ("quantico", "classico", "aleatorio"):
            raise ValueError("motor deve ser 'quantico', 'classico' ou 'aleatorio'")
        self.cod = _Codificacao(espaco)
        self.avaliar_fn, self.motor = avaliar, motor
        self.canonico = canonico or (lambda g: g)           # projetos equivalentes viram um só
        self.populacao, self.geracoes, self.elite = populacao, geracoes, elite
        self.passos, self.lr, self.mutacao, self.temperatura = passos, lr, mutacao, temperatura
        self.backend_gerador, self.seed, self.ao_progresso = backend_gerador, seed, ao_progresso
        n = self.cod.n_bits
        self.gerador = GeradorQuantico(n, [(i, (i + 1) % n) for i in range(n)], camadas_gerador)
        self.params = self.gerador.params_iniciais(seed)
        self.cache = {}

    def _avaliar(self, genoma):
        genoma = self.canonico(genoma)
        k = _chave(genoma)
        if k not in self.cache:
            self.cache[k] = dict(self.avaliar_fn(genoma), genoma=genoma)
        return self.cache[k]

    def _ranking(self):
        return sorted(self.cache.values(), key=lambda a: -a["fitness"])

    def _sortear(self, rng):
        if self.motor == "quantico":
            mutado = self.params + rng.normal(0, self.mutacao, self.params.shape)       # mutação quântica
            return [self.cod.decodificar(x) for x in self.gerador.amostrar(mutado, self.populacao, rng,
                                                                            self.backend_gerador)]
        if self.motor == "aleatorio" or not self.cache:
            return [self.cod.decodificar(int(rng.integers(0, 1 << self.cod.n_bits))) for _ in range(self.populacao)]
        elite = self._ranking()[:max(2, int(round(self.elite * self.populacao)))]   # genético clássico
        filhos = []
        for _ in range(self.populacao):
            a, b = rng.choice(len(elite), 2, replace=len(elite) < 2)
            filho = {nome: (elite[a] if rng.random() < 0.5 else elite[b])["genoma"][nome]
                     for nome, _, _ in self.cod.genes}
            for nome, opcoes, _ in self.cod.genes:
                if rng.random() < 1.0 / len(self.cod.genes):
                    filho[nome] = opcoes[int(rng.integers(len(opcoes)))]
            filhos.append(filho)
        return filhos

    def ligacoes_aprendidas(self, top=5):
        """Informação mútua (bits) entre genes na população do gerador (só no motor quântico)."""
        p = self.gerador.probabilidades(self.params)
        idx = np.arange(len(p))
        regs = list(self.cod.registros().items())
        valores = {nome: sum(((idx >> q) & 1) << k for k, q in enumerate(qs)) for nome, qs in regs}
        out = []
        for i in range(len(regs)):
            for j in range(i + 1, len(regs)):
                a, b = regs[i][0], regs[j][0]
                na, nb = 1 << len(regs[i][1]), 1 << len(regs[j][1])
                conj = np.bincount(valores[a] * nb + valores[b], weights=p, minlength=na * nb).reshape(na, nb)
                pa, pb = conj.sum(1, keepdims=True), conj.sum(0, keepdims=True)
                m = conj > 1e-15
                out.append({"genes": (a, b), "informacao_mutua": max(float(np.sum(conj[m] * np.log2(
                    conj[m] / (pa @ pb)[m]))), 0.0)})
        return sorted(out, key=lambda d: -d["informacao_mutua"])[:top]

    def executar(self):
        rng = np.random.default_rng(self.seed)
        t0 = time.perf_counter()
        historico = []
        for ger in range(self.geracoes + 1):
            geracao = [self._avaliar(g) for g in self._sortear(rng)]
            melhor = self._ranking()[0]
            h = {"geracao": ger, "melhor_fitness": melhor["fitness"], "melhor_acuracia": melhor["acuracia"],
                 "melhor_compressao": melhor["compressao"], "avaliados": len(self.cache),
                 "media_geracao": float(np.mean([a["fitness"] for a in geracao]))}
            if self.motor == "quantico":
                h["entropia_bits"] = self.gerador.entropia(self.params)
            historico.append(h)
            if self.ao_progresso:
                self.ao_progresso(dict(h, motor=self.motor))
            if ger == self.geracoes or self.motor != "quantico":
                continue
            elite = self._ranking()[:max(2, int(round(self.elite * self.populacao)))]     # herança
            xs = [self.cod.codificar(a["genoma"]) for a in elite]
            pesos = np.arange(len(elite), 0, -1, dtype=float)
            pesos /= pesos.sum()
            for _ in range(self.passos):
                _, grad = self.gerador.nll_e_gradiente(self.params, xs, pesos, self.temperatura)
                self.params = self.params - self.lr * grad
        r = {"motor": self.motor, "melhor": self._ranking()[0], "avaliados": len(self.cache),
             "historico": historico, "n_qubits_gerador": self.cod.n_bits,
             "projetos_possiveis": self.cod.n_genomas, "tempo_s": round(time.perf_counter() - t0, 1)}
        if self.motor == "quantico":
            r["ligacoes_aprendidas"] = [{"genes": list(l["genes"]), "bits": l["informacao_mutua"]}
                                        for l in self.ligacoes_aprendidas()]
            r["backend_gerador"] = getattr(self.backend_gerador, "nome", "exato")
        return r


# ── o problema: comprimir um modelo treinado ────────────────────────────────

def camadas_comprimiveis(modelo, min_dim=64):
    from torch import nn
    return [(nome, m) for nome, m in modelo.named_modules()
            if isinstance(m, nn.Linear) and min(m.in_features, m.out_features) >= min_dim]


def espaco_compressao(modelo, min_dim=64):
    espaco = {}
    for nome, _ in camadas_comprimiveis(modelo, min_dim):
        espaco[f"{nome}:metodo"] = ("tt", "posto_baixo", "densa")
        espaco[f"{nome}:nivel"] = (0, 1, 2, 3)
        espaco[f"{nome}:nucleos"] = NUCLEOS
    return espaco


def canonico_compressao(genoma):
    """Núcleos só importam na TT, e nível também não importa numa camada densa."""
    g = dict(genoma)
    for chave, v in genoma.items():
        nome, gene = chave.split(":")
        if nome != "rede" and gene == "metodo" and v != "tt":
            g[f"{nome}:nucleos"] = NUCLEOS[0]
            if v == "densa":
                g[f"{nome}:nivel"] = 0
    return g


def descrever(genoma):
    partes = {}
    for chave, v in genoma.items():
        nome, gene = chave.split(":")
        partes.setdefault(nome, {})[gene] = v
    out = [f"largura {partes.pop('rede')['largura']}"] if "rede" in partes else []
    for nome, g in partes.items():
        if g["metodo"] == "tt":
            out.append(f"camada {nome}: TT {g['nucleos']} núcleos, ligação {NIVEIS_TT[g['nivel']]}")
        elif g["metodo"] == "posto_baixo":
            out.append(f"camada {nome}: posto baixo {NIVEIS_POSTO[g['nivel']]}")
        else:
            out.append(f"camada {nome}: densa")
    return "; ".join(out)


def aplicar_projeto(modelo, genoma):
    """Cópia do modelo treinado com cada camada comprimida como o projeto manda."""
    from .ia_tensorial import CamadaPostoBaixo, CamadaTT
    novo = copy.deepcopy(modelo)
    for nome, lin in camadas_comprimiveis(novo):
        metodo, nivel = genoma[f"{nome}:metodo"], genoma[f"{nome}:nivel"]
        if metodo == "tt":
            camada = CamadaTT.de_linear(lin, NIVEIS_TT[nivel], genoma[f"{nome}:nucleos"])
        elif metodo == "posto_baixo":
            camada = CamadaPostoBaixo.de_linear(lin, min(NIVEIS_POSTO[nivel], min(lin.in_features, lin.out_features)))
        else:
            continue
        pai, _, filho = nome.rpartition(".")
        setattr(novo.get_submodule(pai) if pai else novo, filho, camada)
    return novo


class ProblemaCompressao:
    """avaliar(projeto): comprime o modelo treinado, ajusta por `passos_ajuste`
    lotes e mede no conjunto de VALIDAÇÃO (o teste fica intocado até o fim).
    Fitness = acurácia de validação - 0,002 log2(parâmetros), e -1 se o
    modelo não ficar pelo menos `alvo` vezes menor que o original."""

    def __init__(self, modelo, Xtr, ytr, Xval, yval, alvo=30.0, passos_ajuste=200, lote=128, lr=1e-3, seed=0):
        from .ia_tensorial import contar_params
        self.modelo, self.Xtr, self.ytr, self.Xval, self.yval = modelo, Xtr, ytr, Xval, yval
        self.alvo, self.passos, self.lote, self.lr, self.seed = alvo, passos_ajuste, lote, lr, seed
        self.params_original = contar_params(modelo)

    def ajustar(self, m, passos):
        import torch
        from torch import nn
        g = torch.Generator().manual_seed(self.seed)
        opt = torch.optim.Adam(m.parameters(), lr=self.lr)
        perda = nn.CrossEntropyLoss()
        m.train()
        for _ in range(passos):
            idx = torch.randint(0, len(self.Xtr), (self.lote,), generator=g)
            opt.zero_grad(set_to_none=True)
            perda(m(self.Xtr[idx]), self.ytr[idx]).backward()
            opt.step()
        m.eval()
        return m

    def __call__(self, genoma):
        import torch
        from .ia_tensorial import acuracia, contar_params
        torch.manual_seed(self.seed)
        t0 = time.perf_counter()
        m = self.ajustar(aplicar_projeto(self.modelo, genoma), self.passos)
        p = contar_params(m)
        acc = acuracia(m, self.Xval, self.yval)
        compressao = self.params_original / p
        fitness = acc - 0.002 * np.log2(p) - (1.0 if compressao < self.alvo else 0.0)
        return {"fitness": float(fitness), "acuracia": float(acc), "parametros": int(p),
                "compressao": float(compressao), "descricao": descrever(genoma), "s": time.perf_counter() - t0}


LARGURAS = (1024, 2048)


def espaco_projeto(n_ocultas=2):
    """Projeto de rede do zero: largura das camadas ocultas e, para cada camada
    grande, rede tensorial, posto baixo ou densa."""
    espaco = {"rede:largura": LARGURAS}
    for i in range(n_ocultas):
        nome = str(2 * i)
        espaco[f"{nome}:metodo"] = ("tt", "posto_baixo", "densa")
        espaco[f"{nome}:nivel"] = (0, 1, 2, 3)
        espaco[f"{nome}:nucleos"] = NUCLEOS
    return espaco


def construir_projeto(genoma, entrada=784, saida=10):
    """MLP com as camadas do projeto, pesos aleatórios (para treinar do zero)."""
    from torch import nn
    from .ia_tensorial import CamadaPostoBaixo, CamadaTT
    L = genoma.get("rede:largura", 1024)
    camadas, a = [], entrada
    nomes = sorted({k.split(":")[0] for k in genoma if not k.startswith("rede:")}, key=int)
    for nome in nomes:
        metodo, nivel = genoma[f"{nome}:metodo"], genoma[f"{nome}:nivel"]
        if metodo == "tt":
            camadas.append(CamadaTT(a, L, NIVEIS_TT[nivel], genoma[f"{nome}:nucleos"]))
        elif metodo == "posto_baixo":
            camadas.append(CamadaPostoBaixo(a, L, min(NIVEIS_POSTO[nivel], a, L)))
        else:
            camadas.append(nn.Linear(a, L))
        camadas.append(nn.ReLU())
        a = L
    return nn.Sequential(*camadas, nn.Linear(a, saida))


class ProblemaDoZero(ProblemaCompressao):
    """avaliar(projeto): constrói a rede do projeto, treina do zero por `passos`
    lotes e mede na validação. A meta de tamanho é relativa a `params_referencia`
    (o modelo denso original)."""

    def __init__(self, params_referencia, Xtr, ytr, Xval, yval, alvo=30.0, passos=470, lote=128, lr=1e-3, seed=0):
        self.Xtr, self.ytr, self.Xval, self.yval = Xtr, ytr, Xval, yval
        self.alvo, self.passos, self.lote, self.lr, self.seed = alvo, passos, lote, lr, seed
        self.params_original = params_referencia

    def __call__(self, genoma):
        import torch
        from .ia_tensorial import acuracia, contar_params
        t0 = time.perf_counter()
        torch.manual_seed(self.seed)
        m = construir_projeto(genoma, self.Xtr.shape[1], int(self.ytr.max()) + 1)
        p = contar_params(m)
        compressao = self.params_original / p
        if compressao < self.alvo:                     # estoura a meta de tamanho: nem treina
            return {"fitness": float(-1.0 - 0.002 * np.log2(p)), "acuracia": 0.0, "parametros": int(p),
                    "compressao": float(compressao), "descricao": descrever(genoma), "s": time.perf_counter() - t0}
        m = self.ajustar(m, self.passos)
        acc = acuracia(m, self.Xval, self.yval)
        fitness = acc - 0.002 * np.log2(p) - (1.0 if compressao < self.alvo else 0.0)
        return {"fitness": float(fitness), "acuracia": float(acc), "parametros": int(p),
                "compressao": float(compressao), "descricao": descrever(genoma), "s": time.perf_counter() - t0}


def _hash_pesos(modelo):
    import torch
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in modelo.state_dict().items()}, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest(), buf.getvalue()


def forjar_mnist(alvo=30.0, motores=("quantico",), populacao=6, geracoes=4, passos_ajuste=200, epocas_base=3,
                 epocas_final=3, oculta=1024, seed=0, backend_gerador=None, pasta=None, assinador=None,
                 ao_progresso=None, protocolo="do_zero"):
    """Treina o modelo original e forja a rede com cada motor (mesmo orçamento);
    o vencedor de cada motor é medido no TESTE. Grava os modelos e o comprovante.
    protocolo "do_zero": cada projeto (largura + tipo de cada camada) é treinado
    do zero; "compressao": cada projeto comprime os pesos do modelo já treinado."""
    import torch
    from .ia_tensorial import _dados_mnist, acuracia, contar_params, mlp, tempo_inferencia, treinar
    from .qpu import _selar
    torch.manual_seed(seed)
    Xtr, ytr, Xte, yte = _dados_mnist()
    Xtr, ytr, Xval, yval = Xtr[:55000], ytr[:55000], Xtr[55000:], ytr[55000:]   # validação separada do teste
    base = mlp("denso", oculta=oculta)
    t_base = treinar(base, Xtr, ytr, epocas_base, seed=seed)
    original = {"parametros": contar_params(base), "acuracia_teste": acuracia(base, Xte, yte),
                "ms_lote_1": tempo_inferencia(base, Xte[:1]), "ms_lote_1000": tempo_inferencia(base, Xte[:1000], 10),
                "s_por_epoca": t_base, "epocas": epocas_base}
    if protocolo == "do_zero":
        problema = ProblemaDoZero(original["parametros"], Xtr, ytr, Xval, yval, alvo, passos_ajuste, seed=seed)
        espaco = espaco_projeto()
    else:
        problema = ProblemaCompressao(base, Xtr, ytr, Xval, yval, alvo, passos_ajuste, seed=seed)
        espaco = espaco_compressao(base)
    resultados = {}
    for motor in motores:
        busca = Busca(espaco, problema, motor, populacao, geracoes, seed=seed, canonico=canonico_compressao,
                      backend_gerador=backend_gerador if motor == "quantico" else None, ao_progresso=ao_progresso)
        r = busca.executar()
        torch.manual_seed(seed)
        inicial = (construir_projeto(r["melhor"]["genoma"]) if protocolo == "do_zero"
                   else aplicar_projeto(base, r["melhor"]["genoma"]))
        m = problema.ajustar(inicial, int(epocas_final * len(Xtr) / problema.lote))   # treino final do vencedor
        r["final"] = {"parametros": contar_params(m), "acuracia_teste": acuracia(m, Xte, yte),
                      "compressao": original["parametros"] / contar_params(m),
                      "ms_lote_1": tempo_inferencia(m, Xte[:1]),
                      "ms_lote_1000": tempo_inferencia(m, Xte[:1000], 10)}
        r["_modelo"] = m
        resultados[motor] = r
    saida = {"formato": "quirios.forja", "data": datetime.datetime.now().isoformat(timespec="seconds"),
             "problema": f"MNIST, referência densa 784-{oculta}-{oculta}-10, meta: >= {alvo:g}x menor",
             "protocolo": protocolo, "original": original, "orcamento_por_motor": populacao * (geracoes + 1),
             "passos_por_projeto_na_busca": passos_ajuste, "epocas_treino_final": epocas_final,
             "motores": {k: {kk: vv for kk, vv in v.items() if kk != "_modelo"} for k, v in resultados.items()}}
    if pasta is not None:
        pasta = Path(pasta)
        pasta.mkdir(parents=True, exist_ok=True)
        for motor, r in resultados.items():
            h, dados = _hash_pesos(r["_modelo"])
            (pasta / f"modelo_forjado_{motor}.pt").write_bytes(dados)
            saida["motores"][motor]["pesos_sha256"] = h
        _selar(saida, pasta / "comprovante_forja.json", assinador)
    return saida


def texto_forja(res):
    o = res["original"]
    out = [f"Forja quântica — {res['problema']}; orçamento de {res['orcamento_por_motor']} projetos por motor",
           f"  original: {o['parametros']:,d} parâmetros, teste {100 * o['acuracia_teste']:.2f}%"]
    for motor, r in res["motores"].items():
        f, m = r["final"], r["melhor"]
        out.append(f"  [{motor:9s}] {f['compressao']:6.1f}x menor, teste {100 * f['acuracia_teste']:.2f}% "
                   f"(validação na busca {100 * m['acuracia']:.2f}%), {r['avaliados']} projetos avaliados, "
                   f"{r['tempo_s']:.0f}s  -> {m['descricao']}")
        if r.get("ligacoes_aprendidas"):
            l = r["ligacoes_aprendidas"][0]
            out.append(f"              ligação aprendida pelo emaranhamento: {l['genes'][0]} <-> {l['genes'][1]} "
                       f"({l['bits']:.3f} bits)")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m quirios_engine.forja",
                                 description="Forja quântica: um circuito quântico desenha a compressão de um modelo de IA.")
    sub = ap.add_subparsers(dest="acao", required=True)
    p = sub.add_parser("mnist")
    p.add_argument("--alvo", type=float, default=30.0, help="quantas vezes menor o modelo precisa ficar")
    p.add_argument("--motor", choices=["quantico", "classico", "aleatorio", "todos"], default="quantico")
    p.add_argument("--populacao", type=int, default=6)
    p.add_argument("--geracoes", type=int, default=4)
    p.add_argument("--passos-ajuste", type=int, default=470, help="passos de treino de cada projeto na busca")
    p.add_argument("--protocolo", choices=["do_zero", "compressao"], default="do_zero")
    p.add_argument("--modo-gerador", choices=["exato", "ruidoso", "ibm"], default="exato",
                   help="onde o circuito gerador é medido")
    p.add_argument("--saida", default="resultados/forja")
    p.add_argument("--chave-privada", default=None, help="assina o comprovante com ML-DSA")
    p.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    motores = ("quantico", "classico", "aleatorio") if a.motor == "todos" else (a.motor,)
    backend = None
    if a.modo_gerador != "exato":
        from .qpu import _backend
        backend = _backend(a.modo_gerador, shots=a.populacao)
    assinador = None
    if a.chave_privada:
        from .seguranca import Assinador
        assinador = Assinador(a.chave_privada)
    progresso = lambda h: print(f"  [{h['motor']}] geração {h['geracao']}: melhor {100 * h['melhor_acuracia']:.2f}% "
                                f"com {h['melhor_compressao']:.0f}x menos parâmetros ({h['avaliados']} avaliados)",
                                flush=True)
    res = forjar_mnist(a.alvo, motores, a.populacao, a.geracoes, a.passos_ajuste, seed=a.seed,
                       backend_gerador=backend, pasta=a.saida, assinador=assinador, ao_progresso=progresso,
                       protocolo=a.protocolo)
    print(texto_forja(res))
    print(f"\nmodelos e comprovante em {a.saida}")


if __name__ == "__main__":
    main()
