"""
IA clássica acelerada por redes tensoriais (a matemática dos estados quânticos).

    pip install -e ".[ia]"
    python -m quirios_engine.ia_tensorial velocidade --larguras 4096 8192 --nucleos 6 --posto 8
    python -m quirios_engine.ia_tensorial mnist --posto 8
    python -m quirios_engine.ia_tensorial fronteira          # acurácia x parâmetros x tempo

CamadaTT substitui uma nn.Linear: a matriz de pesos M x N vira uma cadeia de
d núcleos (MPO), o formato do backend MPS do QURIOS. Uma camada 4096 x 4096
tem 16,8 milhões de pesos; em TT com 6 núcleos e ligação 8, 8.448. A cadeia
é cortada ao meio e a camada vira duas multiplicações grandes; medido na CPU
(menor ganho entre duas medições): 6x / 3,9x / 1,6x (4096) e 30x / 6,6x / 3,5x (8192)
mais rápida que a densa com lotes de 1 / 64 / 1000.

Comparação JUSTA, sempre: contra a camada densa original e contra a melhor
compressão clássica comum, a fatoração de posto baixo W = U V (SVD), com o
MESMO número de parâmetros que a camada TT.

comprimir_modelo troca as nn.Linear de um modelo já treinado (de qualquer
tamanho, inclusive camadas de modelos de linguagem) por camadas TT (TT-SVD
dos pesos) ou de posto baixo; depois um ajuste curto recupera a acurácia.
"""
import argparse
import copy
import math
import time

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as e:                      # pragma: no cover
    raise ImportError('A IA tensorial exige PyTorch: pip install -e ".[ia]"') from e

from .redes_tensoriais import carregar_mnist, fatorar, tt_svd_matriz


def _postos(ms, ns, posto):
    d, r = len(ms), [1]
    for k in range(d - 1):
        esq = math.prod(m * n for m, n in zip(ms[:k + 1], ns[:k + 1]))
        dir_ = math.prod(m * n for m, n in zip(ms[k + 1:], ns[k + 1:]))
        r.append(min(posto, esq, dir_))
    return r + [1]


def equilibrar(nucleos):
    """Mesma matriz, núcleos com o mesmo tamanho típico de entrada.

    O TT-SVD deixa os núcleos do meio com entradas minúsculas (são
    ortonormais e grandes) e põe a escala toda no último. O Adam dá passos
    de tamanho parecido em toda entrada, então um núcleo minúsculo muda
    dezenas de por cento por passo e o treino diverge. Multiplicar o núcleo
    k por c_k, com o produto dos c_k igual a 1, não muda W."""
    rms = [float(np.sqrt(np.mean(np.square(G)))) for G in nucleos]
    if min(rms) <= 0:
        return nucleos
    alvo = math.exp(sum(math.log(r) for r in rms) / len(rms))
    return [G * (alvo / r) for G, r in zip(nucleos, rms)]


class CamadaTT(nn.Module):
    """y = W x + b com W (saida x entrada) em formato tensor-train (MPO)."""

    def __init__(self, entrada, saida, posto, d=4, nucleos=None, vies=True):
        super().__init__()
        self.entrada, self.saida, self.posto, self.d = entrada, saida, posto, d
        self.ns, self.ms = fatorar(entrada, d), fatorar(saida, d)
        if nucleos is None:                       # inicialização tipo He: Var(W_ij) = 2/entrada
            r = _postos(self.ms, self.ns, posto)
            sigma = ((2.0 / entrada) / math.prod(r[1:-1])) ** (1 / (2 * d))
            g = np.random.default_rng(int(torch.randint(0, 2 ** 31, (1,))))
            nucleos = [g.normal(0, sigma, (r[k], self.ms[k], self.ns[k], r[k + 1])) for k in range(d)]
        self.nucleos = nn.ParameterList([nn.Parameter(torch.as_tensor(np.asarray(G), dtype=torch.float32))
                                         for G in nucleos])
        self.vies = nn.Parameter(torch.zeros(saida)) if vies else None

    @classmethod
    def de_linear(cls, linear, posto, d=4):
        """TT-SVD dos pesos de uma nn.Linear treinada."""
        W = linear.weight.detach().cpu().double().numpy()
        nucleos = equilibrar(tt_svd_matriz(W, fatorar(linear.out_features, d), fatorar(linear.in_features, d), posto))
        c = cls(linear.in_features, linear.out_features, posto, d, nucleos, vies=linear.bias is not None)
        if linear.bias is not None:
            c.vies.data.copy_(linear.bias.detach())
        return c

    def _metades(self):
        """Corta a cadeia em duas: W[(ml,mr),(nl,nr)] = sum_a L[ml,nl,a] R[a,mr,nr].
        As metades são pequenas (cabem no cache) e saem dos núcleos a cada passada."""
        s = self._corte()[0]

        def juntar(nucleos):
            T = nucleos[0]
            for G in nucleos[1:]:
                T = torch.tensordot(T, G, dims=([T.dim() - 1], [0]))
            k = len(nucleos)
            ordem = [0] + [1 + 2 * j for j in range(k)] + [2 + 2 * j for j in range(k)] + [2 * k + 1]
            return T.permute(ordem)                               # (r_esq, m..., n..., r_dir)

        L = juntar(list(self.nucleos[:s]))                       # (1, ml..., nl..., r)
        R = juntar(list(self.nucleos[s:]))                       # (r, mr..., nr..., 1)
        r = L.shape[-1]
        Ml, Nl = int(np.prod(self.ms[:s])), int(np.prod(self.ns[:s]))
        Mr, Nr = int(np.prod(self.ms[s:])), int(np.prod(self.ns[s:]))
        L = L.reshape(Ml, Nl * r)                                 # (ml, (nl, a))
        R = R.reshape(r, Mr, Nr).permute(2, 0, 1).reshape(Nr, r * Mr)   # (nr, (a, mr))
        return L, R, r, Nl, Mr

    def _corte(self):
        """Onde cortar a cadeia: o corte de menor custo por amostra, r (N Mr + M Nl)."""
        if getattr(self, "_corte_cache", None) is None:
            melhor = None
            for s in range(1, self.d):
                r = self.nucleos[s].shape[0]
                c = r * (self.entrada * int(np.prod(self.ms[s:])) + self.saida * int(np.prod(self.ns[:s])))
                if melhor is None or c < melhor[1]:
                    melhor = (s, c, r)
            self._corte_cache = melhor
        return self._corte_cache

    def _matriz(self):
        """W (saida x entrada) remontada dos núcleos (custo M N r, uma vez por passada).
        Em inferência fica guardada enquanto os núcleos não mudam."""
        versao = tuple(p._version for p in self.nucleos)
        if not torch.is_grad_enabled() and getattr(self, "_W", None) is not None and self._W[0] == versao:
            return self._W[1]
        L, R, r, Nl, Mr = self._metades()
        W = torch.einsum("xna,yam->xmny", L.reshape(L.shape[0], Nl, r), R.reshape(R.shape[0], r, Mr))
        W = W.reshape(self.saida, self.entrada)
        if not torch.is_grad_enabled():
            self._W = (versao, W)
        return W

    def forward(self, x):
        """Escolhe a conta mais barata para o lote: contrair as duas metades da cadeia
        (custo r (N Mr + M Nl) por amostra, a melhor em camadas largas e lotes pequenos)
        ou remontar W e multiplicar (M N por amostra + M N r uma vez). Nunca fica muito
        mais lenta que a camada densa e continua com os parâmetros da rede tensorial."""
        forma = x.shape[:-1]
        B = int(np.prod(forma)) if len(forma) else 1
        _, custo_tt, r = self._corte()
        densa = self.entrada * self.saida
        # inferência: W remontada fica guardada, então vale o custo por amostra; treino: remonta a cada passada
        usar_tt = custo_tt < densa if not torch.is_grad_enabled() else B * custo_tt <= densa * (B + r)
        if usar_tt:
            L, R, r, Nl, Mr = self._metades()
            T = (x.reshape(B * Nl, -1) @ R).reshape(B, Nl * r, Mr)   # contrai a metade direita
            y = torch.matmul(L, T).reshape(*forma, self.saida)       # contrai a metade esquerda
        else:
            y = (x.reshape(B, -1) @ self._matriz().T).reshape(*forma, self.saida)
        return y + self.vies if self.vies is not None else y

    def densa(self):
        """A matriz W que a camada representa (para conferência)."""
        from .redes_tensoriais import tt_para_matriz
        return tt_para_matriz([G.detach().double().numpy() for G in self.nucleos])

    def extra_repr(self):
        return f"entrada={self.entrada}, saida={self.saida}, posto={self.posto}, núcleos={self.ms}x{self.ns}"


class CamadaPostoBaixo(nn.Module):
    """y = U (V x) + b: a compressão clássica de referência (SVD truncado)."""

    def __init__(self, entrada, saida, posto, vies=True):
        super().__init__()
        s = (2.0 / (entrada * posto)) ** 0.25
        self.V = nn.Parameter(torch.randn(posto, entrada) * s)
        self.U = nn.Parameter(torch.randn(saida, posto) * s)
        self.vies = nn.Parameter(torch.zeros(saida)) if vies else None
        self.entrada, self.saida, self.posto = entrada, saida, posto

    @classmethod
    def de_linear(cls, linear, posto):
        W = linear.weight.detach().double()
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        c = cls(linear.in_features, linear.out_features, posto, vies=linear.bias is not None)
        raiz = S[:posto].sqrt()
        c.U.data.copy_((U[:, :posto] * raiz).float())
        c.V.data.copy_((raiz[:, None] * Vt[:posto]).float())
        if linear.bias is not None:
            c.vies.data.copy_(linear.bias.detach())
        return c

    def forward(self, x):
        y = (x @ self.V.T) @ self.U.T
        return y + self.vies if self.vies is not None else y


def params_camada_tt(entrada, saida, posto, d=4):
    ms, ns = fatorar(saida, d), fatorar(entrada, d)
    r = _postos(ms, ns, posto)
    return sum(r[k] * ms[k] * ns[k] * r[k + 1] for k in range(d))


def posto_equivalente(entrada, saida, posto_tt, d=4):
    """Posto da fatoração U V com (aproximadamente) os mesmos parâmetros da camada TT."""
    return max(1, round(params_camada_tt(entrada, saida, posto_tt, d) / (entrada + saida)))


def contar_params(modelo):
    return int(sum(p.numel() for p in modelo.parameters()))


def comprimir_modelo(modelo, metodo="tt", posto=8, d=4, min_dim=64):
    """Cópia do modelo com cada nn.Linear grande (as duas dimensões >= min_dim)
    trocada por CamadaTT (metodo="tt") ou CamadaPostoBaixo com os mesmos
    parâmetros (metodo="svd")."""
    novo = copy.deepcopy(modelo)
    for mod in list(novo.modules()):
        for nome, filho in list(mod.named_children()):
            if isinstance(filho, nn.Linear) and min(filho.in_features, filho.out_features) >= min_dim:
                if metodo == "tt":
                    setattr(mod, nome, CamadaTT.de_linear(filho, posto, d))
                elif metodo == "svd":
                    r = posto_equivalente(filho.in_features, filho.out_features, posto, d)
                    setattr(mod, nome, CamadaPostoBaixo.de_linear(filho, r))
                else:
                    raise ValueError("metodo deve ser 'tt' ou 'svd'")
    return novo


# ── modelos e medidas ───────────────────────────────────────────────────────

def mlp(tipo="denso", entrada=784, oculta=1024, saida=10, posto=8, d=4):
    def camada(a, b):
        if tipo == "tt":
            return CamadaTT(a, b, posto, d)
        if tipo == "svd":
            return CamadaPostoBaixo(a, b, posto_equivalente(a, b, posto, d))
        return nn.Linear(a, b)
    return nn.Sequential(camada(entrada, oculta), nn.ReLU(), camada(oculta, oculta), nn.ReLU(), nn.Linear(oculta, saida))


def treinar(modelo, X, y, epocas=3, lote=128, lr=1e-3, seed=0):
    """Adam + entropia cruzada. Devolve o tempo médio por época (s)."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(modelo.parameters(), lr=lr)
    perda = nn.CrossEntropyLoss()
    modelo.train()
    tempos = []
    for _ in range(epocas):
        t0 = time.perf_counter()
        ordem = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), lote):
            idx = ordem[i:i + lote]
            opt.zero_grad(set_to_none=True)
            perda(modelo(X[idx]), y[idx]).backward()
            opt.step()
        tempos.append(time.perf_counter() - t0)
    modelo.eval()
    return float(np.mean(tempos))


@torch.inference_mode()
def acuracia(modelo, X, y, lote=2000):
    modelo.eval()
    acertos = sum(int((modelo(X[i:i + lote]).argmax(1) == y[i:i + lote]).sum()) for i in range(0, len(X), lote))
    return acertos / len(X)


@torch.inference_mode()
def tempo_inferencia(modelo, x, repeticoes=30):
    """Mediana do tempo (ms) de uma passada com a entrada x."""
    modelo.eval()
    for _ in range(3):
        modelo(x)
    ts = []
    for _ in range(repeticoes):
        t0 = time.perf_counter()
        modelo(x)
        ts.append(time.perf_counter() - t0)
    return 1000 * float(np.median(ts))


# ── experimento 1: velocidade de uma camada do tamanho das de modelos grandes ─

def experimento_velocidade(larguras=(1024, 2048, 4096, 8192), posto=16, lotes=(1, 64), d=4, seed=0):
    torch.manual_seed(seed)
    linhas = []
    for L in larguras:
        camadas = {"densa": nn.Linear(L, L), f"TT (ligação {posto})": CamadaTT(L, L, posto, d),
                   "posto baixo (mesmos parâmetros)": CamadaPostoBaixo(L, L, posto_equivalente(L, L, posto, d))}
        base = None
        for nome, c in camadas.items():
            p = contar_params(c)
            linha = {"largura": L, "camada": nome, "parametros": p, "memoria_mb": 4 * p / 2 ** 20}
            for b in lotes:
                ms = tempo_inferencia(c, torch.randn(b, L), repeticoes=20 if L >= 8192 else 40)
                linha[f"ms_lote_{b}"] = ms
            if base is None:
                base = linha
            linha["compressao"] = base["parametros"] / p
            for b in lotes:
                linha[f"aceleracao_lote_{b}"] = base[f"ms_lote_{b}"] / linha[f"ms_lote_{b}"]
            linhas.append(linha)
    return {"posto": posto, "lotes": list(lotes), "threads": torch.get_num_threads(), "linhas": linhas}


# ── experimento 2: um modelo real (MNIST), treino e compressão ──────────────

def _dados_mnist():
    X, y = carregar_mnist()
    X = torch.as_tensor(X, dtype=torch.float32) / 255.0
    y = torch.as_tensor(y, dtype=torch.long)
    return X[:60000], y[:60000], X[60000:], y[60000:]            # divisão oficial treino/teste


def experimento_mnist(posto=8, epocas=4, oculta=1024, seed=0, ajuste_epocas=1):
    torch.manual_seed(seed)
    Xtr, ytr, Xte, yte = _dados_mnist()
    linhas, modelos = [], {}

    def registrar(nome, origem, m, t_epoca):
        linhas.append({"modelo": nome, "origem": origem, "parametros": contar_params(m),
                       "acuracia_teste": acuracia(m, Xte, yte), "s_por_epoca": t_epoca,
                       "ms_lote_1": tempo_inferencia(m, Xte[:1]), "ms_lote_1000": tempo_inferencia(m, Xte[:1000], 15)})

    for tipo in ("denso", "tt", "svd"):                          # do zero, mesmo treino para todos
        torch.manual_seed(seed)
        m = mlp(tipo, oculta=oculta, posto=posto)
        t = treinar(m, Xtr, ytr, epocas, seed=seed)
        modelos[tipo] = m
        registrar({"denso": "denso", "tt": f"TT (ligação {posto})", "svd": "posto baixo (mesmos parâmetros)"}[tipo],
                  "treinado do zero", m, t)
    for metodo in ("tt", "svd"):                                 # comprimir o denso já treinado
        m = comprimir_modelo(modelos["denso"], metodo, posto)
        nome = f"TT (ligação {posto})" if metodo == "tt" else "posto baixo (mesmos parâmetros)"
        registrar(nome, "denso comprimido, sem ajuste", m, None)
        t = treinar(m, Xtr, ytr, ajuste_epocas, seed=seed + 1)
        registrar(nome, f"denso comprimido + {ajuste_epocas} época de ajuste", m, t)
    base = linhas[0]
    for l in linhas:
        l["compressao"] = base["parametros"] / l["parametros"]
        l["aceleracao_lote_1"] = base["ms_lote_1"] / l["ms_lote_1"]
        l["aceleracao_lote_1000"] = base["ms_lote_1000"] / l["ms_lote_1000"]
    return {"posto": posto, "epocas": epocas, "oculta": oculta, "threads": torch.get_num_threads(), "linhas": linhas}


def mlp_posto_baixo(posto, entrada=784, oculta=1024, saida=10):
    return nn.Sequential(CamadaPostoBaixo(entrada, oculta, posto), nn.ReLU(), CamadaPostoBaixo(oculta, oculta, posto),
                         nn.ReLU(), nn.Linear(oculta, saida))


def experimento_fronteira(postos_tt=(2, 4, 8, 16), postos_svd=(2, 8, 32, 128), epocas=3, oculta=1024, seed=0):
    """Acurácia contra parâmetros e tempo, varrendo os postos das duas compressões:
    a comparação JUSTA é na mesma acurácia (quem precisa de menos para chegar lá)."""
    Xtr, ytr, Xte, yte = _dados_mnist()
    linhas = []

    def medir(nome, familia, posto, m):
        torch.manual_seed(seed)
        t = treinar(m, Xtr, ytr, epocas, seed=seed)
        linhas.append({"modelo": nome, "familia": familia, "posto": posto, "parametros": contar_params(m),
                       "acuracia_teste": acuracia(m, Xte, yte), "s_por_epoca": t,
                       "ms_lote_1": tempo_inferencia(m, Xte[:1]), "ms_lote_1000": tempo_inferencia(m, Xte[:1000], 15)})

    torch.manual_seed(seed)
    medir("denso", "denso", None, mlp("denso", oculta=oculta))
    for r in postos_tt:
        torch.manual_seed(seed)
        medir(f"TT ligação {r}", "rede tensorial (TT)", r, mlp("tt", oculta=oculta, posto=r))
    for r in postos_svd:
        torch.manual_seed(seed)
        medir(f"posto baixo {r}", "posto baixo (SVD)", r, mlp_posto_baixo(r, oculta=oculta))
    return {"epocas": epocas, "oculta": oculta, "threads": torch.get_num_threads(), "linhas": linhas}


def texto_fronteira(res):
    out = [f"MNIST 784-{res['oculta']}-{res['oculta']}-10, {res['epocas']} épocas por modelo, CPU {res['threads']} threads"]
    for l in sorted(res["linhas"], key=lambda l: (l["familia"], l["parametros"])):
        out.append(f"  {l['modelo']:18s} {l['parametros']:>10,d} parâm.  {100 * l['acuracia_teste']:5.2f}%  "
                   f"lote 1: {l['ms_lote_1']:6.3f} ms  lote 1000: {l['ms_lote_1000']:7.2f} ms")
    return "\n".join(out)


def texto_velocidade(res):
    out = [f"Uma camada L x L, CPU com {res['threads']} threads, ligação TT {res['posto']}"]
    for l in res["linhas"]:
        tempos = "  ".join(f"lote {b}: {l[f'ms_lote_{b}']:7.3f} ms ({l[f'aceleracao_lote_{b}']:5.1f}x)"
                           for b in res["lotes"])
        out.append(f"  L={l['largura']:5d} {l['camada']:32s} {l['parametros']:>11,d} parâmetros "
                   f"({l['compressao']:7.1f}x menor)  {tempos}")
    return "\n".join(out)


def texto_mnist(res):
    out = [f"MNIST, rede 784-{res['oculta']}-{res['oculta']}-10, {res['epocas']} épocas, CPU {res['threads']} threads"]
    for l in res["linhas"]:
        t = f"{l['s_por_epoca']:5.1f} s/época" if l["s_por_epoca"] else " " * 12
        out.append(f"  {l['modelo']:32s} {l['origem']:38s} {100 * l['acuracia_teste']:5.2f}%  "
                   f"{l['parametros']:>9,d} parâm. ({l['compressao']:5.1f}x menor)  {t}  "
                   f"inferência lote 1000: {l['ms_lote_1000']:6.2f} ms ({l['aceleracao_lote_1000']:4.1f}x)")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m quirios_engine.ia_tensorial",
                                 description="IA clássica acelerada por redes tensoriais, com comparação justa.")
    sub = ap.add_subparsers(dest="acao", required=True)
    pv = sub.add_parser("velocidade")
    pv.add_argument("--larguras", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    pv.add_argument("--posto", type=int, default=8)
    pm = sub.add_parser("mnist")
    pm.add_argument("--posto", type=int, default=8)
    pm.add_argument("--epocas", type=int, default=4)
    pm.add_argument("--oculta", type=int, default=1024)
    pf = sub.add_parser("fronteira", help="acurácia x parâmetros x tempo, varrendo os postos (comparação justa)")
    pf.add_argument("--postos-tt", type=int, nargs="+", default=[2, 4, 8, 16])
    pf.add_argument("--postos-svd", type=int, nargs="+", default=[2, 8, 32, 128])
    pf.add_argument("--epocas", type=int, default=3)
    pv.add_argument("--nucleos", type=int, default=6, help="d: número de núcleos da cadeia TT")
    a = ap.parse_args(argv)
    if a.acao == "velocidade":
        print(texto_velocidade(experimento_velocidade(a.larguras, a.posto, d=a.nucleos)))
    elif a.acao == "fronteira":
        print(texto_fronteira(experimento_fronteira(a.postos_tt, a.postos_svd, a.epocas)))
    else:
        print(texto_mnist(experimento_mnist(a.posto, a.epocas, a.oculta)))


if __name__ == "__main__":
    main()
