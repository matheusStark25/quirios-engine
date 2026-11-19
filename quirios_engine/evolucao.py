"""
Auto-evolução do núcleo quântico: o QURIOS procura sozinho circuitos
melhores e só troca o circuito em uso quando a melhora é comprovada.

    busca = BuscaEvolutiva(n_qubits=8, registro=RegistroCampeoes("campeoes.json"),
                           chave="breast_cancer")
    resultado = busca.executar(X, y)

    python -m quirios_engine.evolucao --dataset breast_cancer --rodadas 3

Como funciona (busca evolutiva de arquitetura, "quantum architecture search"):

  1. Cada circuito é um GENOMA (topologia, camadas, portas de rotação,
     emaranhador CNOT/CZ/RZZ, codificação ângulo/IQP, leitura Z/Pauli...).
  2. A aptidão de um genoma é a acurácia de validação cruzada repetida do
     mapa de features quântico (ou do kernel projetado, modo="kernel"),
     menos uma penalidade pelo número de portas de 2 qubits -- circuitos
     mais rasos sofrem menos ruído no hardware.
  3. A população evolui por torneio, cruzamento uniforme, mutação e elitismo.
     O genoma padrão do QURIOS e o campeão atual sempre participam: a busca
     nunca termina pior que o ponto de partida.
  4. PROMOÇÃO SEGURA: o melhor genoma só substitui o titular -- o campeão
     registrado, ou o circuito padrão do QURIOS enquanto não há campeão -- se
     (a) vencer em acurácia com p < alfa no teste t corrigido de
     Nadeau-Bengio, nas MESMAS dobras, ou (b) custar >= 25% menos portas de 2
     qubits sem perder acurácia (limite inferior do IC95 > -1 ponto).
     Sem evidência, o titular fica. O sistema melhora sozinho, rodada após
     rodada, e nunca regride.
  5. O registro guarda o campeão de cada problema e o histórico de tentativas.
     Com uma chave ML-DSA (seguranca.py), o registro é assinado com
     criptografia pós-quântica e verificado ao abrir.

Aviso de "quanticidade": um circuito sem portas de 2 qubits é um estado
produto -- simulável classicamente em tempo linear. Se o campeão não tem
emaranhamento, o ganho veio de um mapa de features trigonométrico, não de
efeito quântico; o resultado marca isso (`simulavel_classicamente`).
min_portas_2q > 0 restringe a busca a circuitos com emaranhamento.

O Orquestrador usa o campeão automaticamente
(candidatos_padrao(..., registro=..., chave=...)): modelos novos recebem o
melhor circuito já descoberto sem nenhuma mudança de código.

Referências: evolução de arquitetura com mutação/cruzamento (EQAS,
arXiv:2212.00421), reaproveitamento de conhecimento entre buscas (QUEST-A,
arXiv:2410.23560), neuro-evolução híbrida evolutiva-variacional
(arXiv:2602.03840).
"""
import argparse
import copy
import datetime
import json
import os
import threading
import time
from pathlib import Path

import numpy as np

from .core import QuantumConfig
from .estatistica import comparar_pareado

ESPACO_PADRAO = {
    "topologia": ("anel", "linear", "blocos", "todos_pares"),
    "n_camadas_ent": (0, 1, 2),
    "repeticoes_dados": (1, 2, 3),
    "rotacoes": (("ry", "rz"), ("ry",), ("rx", "ry"), ("rz", "ry", "rz")),
    "emaranhador": ("cnot", "cz", "rzz"),
    "codificacao": ("angulo", "iqp"),
    "leitura": ("z_zz", "pauli", "pauli2"),
    "camada_pos": (True, False),
    "escala_alpha": (float(np.pi / 4), float(np.pi / 2), float(np.pi)),
}

GENOMA_PADRAO = {"topologia": "anel", "n_camadas_ent": 2, "repeticoes_dados": 1,
                 "rotacoes": ("ry", "rz"), "emaranhador": "cnot", "codificacao": "angulo",
                 "leitura": "z_zz", "camada_pos": True, "escala_alpha": float(np.pi / 2)}


def normalizar(genoma):
    g = dict(GENOMA_PADRAO, **genoma)
    g["rotacoes"] = tuple(g["rotacoes"])
    g["escala_alpha"] = float(g["escala_alpha"])
    return g


def chave_genoma(genoma):
    g = normalizar(genoma)
    return json.dumps({k: list(v) if isinstance(v, tuple) else v for k, v in sorted(g.items())})


def config_do_genoma(genoma, n_qubits):
    g = normalizar(genoma)
    return QuantumConfig(n_qubits, g["topologia"], g["n_camadas_ent"], tam_bloco=min(8, n_qubits))


def espaco_valido(espaco, n_qubits):
    """Remove escolhas inviáveis para n_qubits (ex.: todos_pares acima de 8)."""
    e = {k: tuple(v) for k, v in espaco.items()}
    if n_qubits > 8:
        e["topologia"] = tuple(t for t in e["topologia"] if t != "todos_pares")
    return e


def descrever(genoma):
    g = normalizar(genoma)
    return (f"{g['codificacao']}/{'-'.join(g['rotacoes'])}/{g['emaranhador']}/{g['topologia']}"
            f"x{g['n_camadas_ent']}/rep{g['repeticoes_dados']}/{g['leitura']}"
            f"{'' if g['camada_pos'] else '/sem-pos'}")


def _paraJSON(v):
    if isinstance(v, tuple):
        return list(v)
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {k: _paraJSON(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_paraJSON(x) for x in v]
    return v


# ── registro de campeões ────────────────────────────────────────────────────

_travas_registro = {}
_trava_global = threading.Lock()


class _TravaArquivo:
    """Exclusão mútua entre threads E processos para um arquivo (arquivo .lock
    criado com O_EXCL; um lock abandonado há mais de `expira_s` é descartado)."""

    def __init__(self, caminho, timeout_s=60.0, expira_s=120.0):
        self.lock = Path(str(caminho) + ".lock")
        self.timeout_s, self.expira_s = timeout_s, expira_s
        with _trava_global:
            self.thread = _travas_registro.setdefault(str(Path(caminho).resolve()), threading.Lock())

    def __enter__(self):
        self.thread.acquire()
        limite = time.monotonic() + self.timeout_s
        while True:
            try:
                fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    if time.time() - self.lock.stat().st_mtime > self.expira_s:
                        self.lock.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() > limite:
                    self.thread.release()
                    raise TimeoutError(f"registro ocupado: {self.lock}")
                time.sleep(0.02)

    def __exit__(self, *exc):
        self.lock.unlink(missing_ok=True)
        self.thread.release()


class RegistroCampeoes:
    """Campeão de cada problema + histórico, num JSON. Opcionalmente assinado
    com ML-DSA (assinador=Assinador(...), verificador=Verificador(...)).
    Seguro para escrita concorrente (threads e processos): cada registro relê
    o arquivo sob trava antes de gravar, então nenhuma tentativa se perde."""

    def __init__(self, caminho, assinador=None, verificador=None):
        self.caminho = Path(caminho)
        self.assinador, self.verificador = assinador, verificador
        self.dados = self._ler()

    def _ler(self):
        if not self.caminho.exists():
            return {"formato": "quirios.campeoes", "versao_formato": 1, "problemas": {}}
        if self.verificador is not None and not self.verificador.verificar_arquivo(self.caminho):
            raise ValueError(f"assinatura pós-quântica inválida em {self.caminho}: "
                             f"registro alterado ou de outra chave")
        for tentativa in range(100):
            try:
                return json.loads(self.caminho.read_text(encoding="utf-8"))
            except PermissionError:                   # Windows: o arquivo está sendo trocado agora
                if tentativa == 99:
                    raise
                time.sleep(0.01 + 0.01 * (tentativa % 5))

    def campeao(self, chave):
        p = self.dados["problemas"].get(chave)
        return None if p is None else p.get("campeao")

    def historico(self, chave):
        return self.dados["problemas"].get(chave, {}).get("historico", [])

    def chaves(self):
        return sorted(self.dados["problemas"])

    def _salvar(self):
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.caminho.with_suffix(self.caminho.suffix + ".tmp")
        tmp.write_text(json.dumps(_paraJSON(self.dados), indent=2, ensure_ascii=False), encoding="utf-8")
        for tentativa in range(100):                      # escrita atômica
            try:
                os.replace(tmp, self.caminho)
                break
            except PermissionError:
                # Windows: o replace é negado enquanto outro processo (um leitor do registro,
                # antivírus, indexador) está com o arquivo aberto. A leitura dura milissegundos.
                if tentativa == 99:
                    raise
                time.sleep(0.01 + 0.01 * (tentativa % 5))
        if self.assinador is not None:
            self.assinador.assinar_arquivo(self.caminho)

    def registrar(self, chave, entrada, promover):
        with _TravaArquivo(self.caminho):
            self.dados = self._ler()                   # pega o que outros processos gravaram
            p = self.dados["problemas"].setdefault(chave, {"campeao": None, "historico": []})
            entrada = dict(entrada, promovido=bool(promover))
            p["historico"].append(entrada)
            if promover:
                p["campeao"] = entrada
            self._salvar()


# ── busca ───────────────────────────────────────────────────────────────────

class BuscaEvolutiva:
    def __init__(self, n_qubits, espaco=None, populacao=10, geracoes=6, modo="reservatorio",
                 backend=None, dobras=5, repeticoes=2, penalidade_custo=0.5, prob_mutacao=0.3,
                 elite=2, seed=0, tempo_max_s=None, registro=None, chave=None, alfa=0.05,
                 min_portas_2q=0, verbose=False):
        if modo not in ("reservatorio", "kernel"):
            raise ValueError("modo deve ser 'reservatorio' ou 'kernel'")
        if registro is not None and not chave:
            raise ValueError("com registro, informe a chave do problema")
        self.n_qubits = n_qubits
        self.espaco = espaco_valido(espaco or ESPACO_PADRAO, n_qubits)
        self.populacao, self.geracoes, self.modo = populacao, geracoes, modo
        self.backend, self.dobras, self.repeticoes = backend, dobras, repeticoes
        self.penalidade_custo, self.prob_mutacao, self.elite = penalidade_custo, prob_mutacao, elite
        self.seed, self.tempo_max_s = seed, tempo_max_s
        self.registro, self.chave, self.alfa, self.verbose = registro, chave, alfa, verbose
        self.min_portas_2q = min_portas_2q
        self._rng = np.random.default_rng(seed)
        self._cache = {}

    # genética
    def _aleatorio(self):
        return normalizar({k: v[self._rng.integers(len(v))] for k, v in self.espaco.items()})

    def _mutar(self, g):
        g = dict(g)
        genes = [k for k in self.espaco if len(self.espaco[k]) > 1]
        mudou = False
        for k in genes:
            if self._rng.random() < self.prob_mutacao:
                opcoes = [v for v in self.espaco[k] if v != g[k]]
                g[k] = opcoes[self._rng.integers(len(opcoes))]
                mudou = True
        if not mudou:                                         # pelo menos um gene muda
            k = genes[self._rng.integers(len(genes))]
            opcoes = [v for v in self.espaco[k] if v != g[k]]
            g[k] = opcoes[self._rng.integers(len(opcoes))]
        return normalizar(g)

    def _cruzar(self, a, b):
        return normalizar({k: (a if self._rng.random() < 0.5 else b)[k] for k in self.espaco})

    def _torneio(self, avaliados, k=3):
        idx = self._rng.choice(len(avaliados), size=min(k, len(avaliados)), replace=False)
        return max((avaliados[i] for i in idx), key=lambda a: a["fitness"])["genoma"]

    # aptidão
    def _divisoes(self, y):
        from .orquestrador import divisoes_cv
        return divisoes_cv(y, self.dobras, self.repeticoes, self.seed)[0]

    def _backend_para(self, genoma):
        if self.backend is not None:
            return self.backend
        from .orquestrador import escolher_backend
        return escolher_backend(config_do_genoma(genoma, self.n_qubits))[0]

    def avaliar(self, genoma, X, y, divisoes):
        chave = chave_genoma(genoma)
        if chave in self._cache:
            return self._cache[chave]
        from .orquestrador import DeGerador, DeModelo
        from .sistema import Quantico
        g = normalizar(genoma)
        t0 = time.perf_counter()
        backend = self._backend_para(g)
        q = Quantico.de_genoma(g, self.n_qubits, backend, seed=self.seed)
        q.preparar(X)
        custo = q.ansatz.portas_2q()
        try:
            if self.modo == "reservatorio":
                accs, _ = DeGerador(q).avaliar(X, y, divisoes)
            else:
                from .kernels import ClassificadorKernel, KernelQuantico
                cfg = config_do_genoma(g, self.n_qubits)
                fab = lambda: ClassificadorKernel(KernelQuantico(
                    cfg, backend=backend, seed=self.seed, codificacao=g["codificacao"],
                    rotacoes=g["rotacoes"], emaranhador=g["emaranhador"], camada_pos=g["camada_pos"]))
                accs, _ = DeModelo("kernel", fab, quantico=True).avaliar(X, y, divisoes)
            erro = None
        except Exception as e:                       # genoma inviável não derruba a busca
            accs, erro = np.zeros(len(divisoes)), f"{type(e).__name__}: {e}"
        acc = float(np.mean(accs))
        fitness = acc - (self.penalidade_custo / 100) * (custo / 100)
        if custo < self.min_portas_2q:                 # fora da restrição de "quanticidade"
            fitness, erro = -1.0, erro or f"menos de {self.min_portas_2q} portas de 2 qubits"
        r = {"genoma": g, "descricao": descrever(g), "acuracia": acc, "accs": np.asarray(accs, float),
             "custo": int(custo), "fitness": fitness, "simulavel_classicamente": custo == 0,
             "tempo_s": round(time.perf_counter() - t0, 3), "erro": erro}
        self._cache[chave] = r
        return r

    def _decidir(self, melhor, titular, n_tr, n_te, rotulo="campeão"):
        """Promove o melhor genoma sobre o titular (o campeão, ou o circuito
        padrão quando ainda não há campeão)? Só com evidência de ganho, ou
        economia de portas sem perda."""
        if chave_genoma(melhor["genoma"]) == chave_genoma(titular["genoma"]):
            return False, f"o {rotulo} atual continua sendo o melhor"
        t = comparar_pareado(melhor["accs"] - titular["accs"], correcao=n_te / n_tr)
        ganho, lo, p = 100 * t["ganho"], 100 * t["ic"][0], t["p_valor"]
        if t["ganho"] > 0 and p < self.alfa:
            return True, f"ganho de {ganho:+.2f} pp sobre o {rotulo} com evidência (p={p:.4f})"
        if melhor["custo"] <= 0.75 * titular["custo"] and lo > -1.0:
            return True, (f"{titular['custo'] - melhor['custo']} portas de 2 qubits a menos que o {rotulo} "
                          f"sem perda de acurácia ({ganho:+.2f} pp, IC95 inferior {lo:+.2f} pp)")
        return False, f"sem evidência de melhora sobre o {rotulo} ({ganho:+.2f} pp, p={p:.4f}); {rotulo} mantido"

    def executar(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        t0 = time.perf_counter()
        divisoes = self._divisoes(y)
        campeao_reg = self.registro.campeao(self.chave) if self.registro else None

        semente = [normalizar({k: v for k, v in GENOMA_PADRAO.items()})]
        if campeao_reg is not None:
            semente.append(normalizar(campeao_reg["genoma"]))
        pop, vistos = [], set()
        for g in semente + [self._aleatorio() for _ in range(4 * self.populacao)]:
            if chave_genoma(g) not in vistos and len(pop) < self.populacao:
                pop.append(g); vistos.add(chave_genoma(g))

        historico = []
        avaliados = [self.avaliar(g, X, y, divisoes) for g in pop]
        for ger in range(self.geracoes + 1):
            avaliados.sort(key=lambda a: -a["fitness"])
            historico.append({"geracao": ger, "melhor": avaliados[0]["fitness"],
                              "media": float(np.mean([a["fitness"] for a in avaliados])),
                              "melhor_descricao": avaliados[0]["descricao"]})
            if self.verbose:
                print(f"  geração {ger}: melhor {100 * avaliados[0]['acuracia']:.1f}% "
                      f"({avaliados[0]['descricao']}, {avaliados[0]['custo']} portas 2q)", flush=True)
            if ger == self.geracoes or (self.tempo_max_s and time.perf_counter() - t0 > self.tempo_max_s):
                break
            nova = [a["genoma"] for a in avaliados[:self.elite]]
            tentativas = 0
            while len(nova) < self.populacao and tentativas < 50 * self.populacao:
                tentativas += 1
                filho = self._mutar(self._cruzar(self._torneio(avaliados), self._torneio(avaliados)))
                if chave_genoma(filho) not in {chave_genoma(g) for g in nova}:
                    nova.append(filho)
            avaliados = [self.avaliar(g, X, y, divisoes) for g in nova]

        todos = sorted(self._cache.values(), key=lambda a: -a["fitness"])
        melhor = todos[0]
        base = self._cache[chave_genoma(GENOMA_PADRAO)]
        resultado = {"melhor": melhor, "padrao": base, "avaliados": len(self._cache),
                     "historico": historico, "tempo_s": round(time.perf_counter() - t0, 2),
                     "promovido": None, "motivo": None, "motor": "classico"}
        if self.registro is not None:
            resultado["promovido"], resultado["motivo"] = self._promover(
                melhor, campeao_reg, X, y, divisoes, motor="classico")
        return resultado

    def _promover(self, melhor, campeao_reg, X, y, divisoes, motor):
        """Decide e registra a promoção (mesma regra para os motores clássico e quântico)."""
        from . import __version__
        if campeao_reg:                      # sem campeão, o circuito padrão é o titular a ser batido
            titular, rotulo = self.avaliar(campeao_reg["genoma"], X, y, divisoes), "campeão"
        else:
            titular, rotulo = self.avaliar(normalizar(GENOMA_PADRAO), X, y, divisoes), "circuito padrão"
        promover, motivo = self._decidir(melhor, titular, len(divisoes[0][0]), len(divisoes[0][1]), rotulo)
        entrada = {"genoma": melhor["genoma"], "descricao": melhor["descricao"],
                   "acuracia": melhor["acuracia"], "accs": melhor["accs"], "custo": melhor["custo"],
                   "fitness": melhor["fitness"], "modo": self.modo, "motor": motor, "n_amostras": len(X),
                   "simulavel_classicamente": melhor["simulavel_classicamente"],
                   "n_qubits": self.n_qubits, "versao_quirios": __version__,
                   "data": datetime.datetime.now().isoformat(timespec="seconds"), "motivo": motivo}
        self.registro.registrar(self.chave, entrada, promover)
        return promover, motivo


def evoluir_continuamente(X, y, n_qubits, registro, chave, rodadas=3, tempo_max_s=None,
                          seed=0, verbose=False, motor="quantico", **kw):
    """Roda buscas seguidas, cada uma partindo do campeão anterior -- o ciclo
    de auto-melhoria. motor="quantico" (padrão) usa a EvolucaoQuantica (a
    população em superposição num circuito gerador); "classico", a BuscaEvolutiva.
    Para ao fim das rodadas ou do tempo máximo."""
    if motor not in ("quantico", "classico"):
        raise ValueError("motor deve ser 'quantico' ou 'classico'")
    t0, saidas = time.perf_counter(), []
    for r in range(rodadas if rodadas is not None else 10 ** 9):
        if tempo_max_s and time.perf_counter() - t0 > tempo_max_s:
            break
        if verbose:
            print(f"\nrodada {r + 1} (motor {motor})", flush=True)
        if motor == "quantico":
            from .evolucao_quantica import EvolucaoQuantica
            busca = EvolucaoQuantica(n_qubits, registro=registro, chave=chave, seed=seed + r,
                                     verbose=verbose, **kw)
        else:
            busca = BuscaEvolutiva(n_qubits, registro=registro, chave=chave, seed=seed + r,
                                   verbose=verbose, **kw)
        res = busca.executar(X, y)
        saidas.append(res)
        if verbose:
            if motor == "quantico":
                print(busca.resumo(res), flush=True)
            else:
                print(f"  -> {'PROMOVIDO' if res['promovido'] else 'mantido'}: {res['motivo']}", flush=True)
    return saidas


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m quirios_engine.evolucao",
                                 description="Auto-evolução do circuito quântico do QURIOS.")
    ap.add_argument("--dataset", default="breast_cancer")
    ap.add_argument("--n-qubits", type=int, default=8)
    ap.add_argument("--n-treino", type=int, default=120)
    ap.add_argument("--populacao", type=int, default=10)
    ap.add_argument("--geracoes", type=int, default=5)
    ap.add_argument("--rodadas", type=int, default=1, help="0 = contínuo até --tempo-max")
    ap.add_argument("--tempo-max", type=float, default=None, help="segundos")
    ap.add_argument("--modo", choices=["reservatorio", "kernel"], default="reservatorio")
    ap.add_argument("--motor", choices=["quantico", "classico"], default="quantico",
                    help="quantico: a população evolui em superposição num circuito gerador")
    ap.add_argument("--registro", default="campeoes_quirios.json")
    ap.add_argument("--min-portas-2q", type=int, default=0,
                    help="só aceita circuitos com emaranhamento (>= N portas de 2 qubits)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    from sklearn.model_selection import train_test_split
    from .benchmark import _preparar, carregar_dataset
    X, y = carregar_dataset(a.dataset)
    X_pool, X_te, y_pool, y_te = train_test_split(X, y, test_size=min(120, int(0.3 * len(y))),
                                                  stratify=y, random_state=42)
    nq = min(a.n_qubits, X.shape[1])
    X_pool, X_te = _preparar(X_pool, X_te, nq)
    r = np.random.default_rng(a.seed)
    sel = r.choice(len(y_pool), min(a.n_treino, len(y_pool)), replace=False)
    Xtr, ytr = X_pool[sel], y_pool[sel].astype(float)

    chave = f"{a.dataset}:{nq}q:{a.modo}" + (f":min2q={a.min_portas_2q}" if a.min_portas_2q else "")
    registro = RegistroCampeoes(a.registro)
    print(f"Auto-evolução QURIOS: {chave}, {len(ytr)} amostras de treino, registro {a.registro}")
    rodadas = None if a.rodadas == 0 else a.rodadas
    evoluir_continuamente(Xtr, ytr, nq, registro, chave, rodadas=rodadas, tempo_max_s=a.tempo_max,
                          seed=a.seed, verbose=True, populacao=a.populacao, geracoes=a.geracoes,
                          modo=a.modo, min_portas_2q=a.min_portas_2q, motor=a.motor)

    from .orquestrador import DeGerador
    from .sistema import Quantico
    camp = registro.campeao(chave)
    if camp is None:
        print("nenhum circuito venceu o padrão com evidência: o padrão QURIOS continua em uso")
    for nome, g in [("padrão QURIOS", GENOMA_PADRAO)] + ([("campeão", camp["genoma"])] if camp else []):
        q = Quantico.de_genoma(g, nq, seed=a.seed)
        c = DeGerador(q).ajustar(Xtr, ytr)
        acc = np.mean((c.predict_proba(X_te) > 0.5) == y_te)
        print(f"teste ({len(y_te)} amostras nunca vistas) -- {nome:14s}: {100 * acc:.1f}%  "
              f"[{descrever(g)}, {q.ansatz.portas_2q()} portas 2q]")
    if camp and camp.get("simulavel_classicamente"):
        print("aviso: o campeão não usa emaranhamento (0 portas de 2 qubits) -- é simulável "
              "classicamente; use --min-portas-2q para buscar só circuitos quânticos")


if __name__ == "__main__":
    main()
