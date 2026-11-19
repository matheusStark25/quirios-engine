"""
Dados com estrutura quântica: classificar a FASE de um sistema quântico a
partir do seu estado -- dados que já nascem quânticos (saída de um simulador
quântico, de um sensor ou de outro processador).

    from quirios_engine.dados_quanticos import fases_ising, comparar, texto_comparacao
    d = fases_ising(n_spins=8, n_amostras=200, preparo="circuito")
    print(texto_comparacao(comparar(d)))

    python -m quirios_engine.dados_quanticos --n-spins 8 --shots 10 30 100

Modelo de Ising com campo transversal e acoplamentos desordenados (anel):

    H = - sum_i J_i Z_i Z_{i+1}  -  h sum_i X_i
    rótulo 0: fase ferromagnética    (h < J_c)
    rótulo 1: fase paramagnética     (h > J_c),  J_c = média geométrica de J

Preparo do estado:
    "exato"     estado fundamental por diagonalização (referência ideal)
    "circuito"  o estado que o PRÓPRIO circuito prepara (codificação "ising" do
                Ansatz: cronograma de 6 passos com RZZ/RX, fidelidade ~0,93 com
                8 spins) -- o mesmo circuito que roda na QPU. O pipeline inteiro
                (gerar o dado + classificar) fica dentro do processador quântico.

Visões dos MESMOS sistemas:
    X_hamiltoniano  (J, h): usado SÓ para preparar o estado no circuito
    X_estado        as 2^n amplitudes (só existem em simulação)
    X_medidas       o que se mede do estado com um número FINITO de shots:
                    <Z_i Z_i+1>, <X_i>, |magnetização|

COMPARAÇÃO JUSTA (padrão): o modelo quântico também mede com shots finitos e
recebe o MESMO orçamento total de medições do clássico (2N por sistema: N na
base Z e N na base X para o clássico; o kernel projetado divide 2N pelas 3
bases X/Y/Z). justo=False mostra o "teto teórico" (valores esperados exatos,
infinitos shots), que NÃO é uma comparação válida.

Resultado medido (8 spins, h entre 0,85 e 1,2, 40 sistemas de treino), com o
mesmo número de medições: o kernel quântico fica 6 a 11 pontos ABAIXO do
melhor clássico (20 medições: 64,0% vs 75,4%; 200: 86,1% vs 92,0%). Com
infinitos shots ele chegaria a 92-97% -- daí o "+18 pp" da v0.7, que era um
artefato dessa comparação injusta. Neste problema a fase está em correlações
LOCAIS (<ZZ>), que o clássico mede diretamente: não há o que o circuito
"comprima" melhor. Vantagem de amostragem é esperada quando a informação é
NÃO local (fases topológicas / ordem de string, QCNN -- Cong, Choi & Lukin,
Nat. Phys. 2019), que é o próximo benchmark.
"""
import argparse
import time

import threading

import numpy as np

from .core import QuantumConfig
from .estatistica import comparar_pareado

# O eigh do OpenBLAS (dsyevd; numpy 2.x no Windows) derruba o processo com access violation
# quando várias threads o chamam ao mesmo tempo -- achado pelo teste de estresse da API, com
# tarefas /dados-quanticos simultâneas. eigvalsh, svd, solve e qr aguentam a concorrência.
_TRAVA_EIGH = threading.Lock()


def _sinal(idx, q):
    return 1 - 2 * ((idx >> q) & 1)


def hamiltoniano_ising(J, h):
    """Matriz (densa ou esparsa) de H = -sum J_i Z_i Z_i+1 - h sum X_i (convenção LSB)."""
    n = len(J)
    dim = 1 << n
    idx = np.arange(dim)
    diag = -sum(J[i] * _sinal(idx, i) * _sinal(idx, (i + 1) % n) for i in range(n)).astype(float)
    linhas = np.concatenate([idx for _ in range(n)])
    colunas = np.concatenate([idx ^ (1 << q) for q in range(n)])
    try:
        from scipy.sparse import coo_matrix, diags
        return diags(diag) + coo_matrix((np.full(len(linhas), -h), (linhas, colunas)), shape=(dim, dim))
    except ImportError:
        H = np.diag(diag)
        H[linhas, colunas] += -h
        return H


def estado_fundamental(J, h):
    """Estado fundamental (real, amplitudes >= 0 pelo teorema de Perron-Frobenius)."""
    H = hamiltoniano_ising(J, h)
    if hasattr(H, "tocsr") and H.shape[0] > 256:
        from scipy.sparse.linalg import eigsh
        _, v = eigsh(H.tocsr(), k=1, which="SA")
        psi = v[:, 0]
    else:
        H = H.toarray() if hasattr(H, "toarray") else H
        with _TRAVA_EIGH:
            _, v = np.linalg.eigh(H)
        psi = v[:, 0]
    psi = psi * np.sign(psi[np.argmax(np.abs(psi))])
    return psi / np.linalg.norm(psi)


def ansatz_preparo(n_spins, cronograma=None, **genes):
    """Ansatz que só PREPARA o estado de Ising (sem camadas treináveis efetivas)."""
    from .ansatz import Ansatz
    return Ansatz(QuantumConfig(n_spins, "anel", 0), codificacao="ising", cronograma_ising=cronograma,
                  rotacoes=("rz",), camada_pos=False, **genes)


def estado_preparado(J, h, cronograma=None):
    """Estado que o circuito prepara (o que a QPU produziria sem ruído)."""
    a = ansatz_preparo(len(J), cronograma)
    return a.estado(np.concatenate([J, [h]]), np.zeros(a.n_params))


def otimizar_cronograma(n_spins=6, passos=6, n_instancias=24, tentativas=4, seed=0, h_min=0.7, h_max=1.4):
    """Otimiza (gammas, betas) para maximizar a fidelidade média com o estado
    fundamental em instâncias aleatórias. Reproduz CRONOGRAMA_ISING."""
    from scipy.optimize import minimize
    r = np.random.default_rng(seed)
    inst = []
    for _ in range(n_instancias):
        J = 1 + 0.3 * r.uniform(-1, 1, n_spins)
        h = r.uniform(h_min, h_max)
        inst.append((J, h, estado_fundamental(J, h)))

    def custo(v):
        cron = (v[:passos], v[passos:])
        return 1 - np.mean([abs(np.vdot(gs, estado_preparado(J, h, cron))) ** 2 for J, h, gs in inst])
    melhor = None
    for t in range(tentativas):
        rt = np.random.default_rng(t)
        v0 = np.concatenate([np.linspace(0.1, 0.6, passos) * rt.uniform(0.5, 1.5),
                             np.full(passos, 0.4) * rt.uniform(0.5, 1.5)])
        res = minimize(custo, v0, method="L-BFGS-B")
        if melhor is None or res.fun < melhor.fun:
            melhor = res
    return (tuple(melhor.x[:passos]), tuple(melhor.x[passos:])), 1 - melhor.fun


def _transformada_hadamard(v, n):
    """H^{x n} |v>: muda para a base X (Walsh-Hadamard rápida, O(n 2^n))."""
    v = np.asarray(v).astype(complex if np.iscomplexobj(v) else float).copy()
    for q in range(n):
        v = v.reshape(-1, 2, 1 << q)
        a, b = v[:, 0, :].copy(), v[:, 1, :].copy()
        v[:, 0, :], v[:, 1, :] = (a + b) / np.sqrt(2), (a - b) / np.sqrt(2)
        v = v.reshape(-1)
    return v


def medir(psi, n, shots, rng):
    """Features clássicas com `shots` medições em cada base (Z e X):
    [<Z_i Z_i+1>] (n), [<X_i>] (n), <|M|> (1)."""
    idx = np.arange(1 << n)
    pz = np.abs(psi) ** 2
    amostras_z = rng.choice(1 << n, size=shots, p=pz / pz.sum())
    z = np.array([_sinal(amostras_z, q) for q in range(n)])          # (n, shots)
    zz = np.array([np.mean(z[i] * z[(i + 1) % n]) for i in range(n)])
    mag = np.mean(np.abs(z.sum(0))) / n
    px = np.abs(_transformada_hadamard(psi, n)) ** 2
    amostras_x = rng.choice(idx, size=shots, p=px / px.sum())
    x = np.array([np.mean(_sinal(amostras_x, q)) for q in range(n)])
    return np.concatenate([zz, x, [mag]])


def fases_ising(n_spins=8, n_amostras=160, h_min=0.7, h_max=1.4, desordem=0.3, shots=50, seed=0,
                preparo="exato", cronograma=None):
    """Gera o conjunto de dados (ver o topo do módulo). A janela padrão de h fica
    em volta da transição de fase, onde distinguir as fases é difícil."""
    if not 3 <= n_spins <= 12:
        raise ValueError("n_spins deve estar entre 3 e 12 (estado exato de 2^n amplitudes)")
    if preparo not in ("exato", "circuito"):
        raise ValueError("preparo deve ser 'exato' ou 'circuito'")
    r = np.random.default_rng(seed)
    estados, medidas, hams, ys, hs, jcs, fids = [], [], [], [], [], [], []
    for _ in range(n_amostras):
        J = 1 + desordem * r.uniform(-1, 1, n_spins)
        h = r.uniform(h_min, h_max)
        jc = float(np.exp(np.mean(np.log(J))))
        alvo = estado_fundamental(J, h)
        psi = alvo if preparo == "exato" else estado_preparado(J, h, cronograma)
        fids.append(float(abs(np.vdot(alvo, psi)) ** 2))
        estados.append(psi)
        medidas.append(medir(psi, n_spins, shots, r))
        hams.append(np.concatenate([J, [h]]))
        ys.append(int(h > jc)); hs.append(h); jcs.append(jc)
    return {"X_estado": np.array(estados), "X_medidas": np.array(medidas), "X_hamiltoniano": np.array(hams),
            "y": np.array(ys, float), "h": np.array(hs), "J_c": np.array(jcs), "fidelidade": np.array(fids),
            "n_spins": n_spins, "shots": shots, "preparo": preparo, "cronograma": cronograma}


class _Normalizado:
    """Padroniza as features antes de um modelo clássico do sistema QURIOS."""

    def __init__(self, modelo):
        self.modelo = modelo

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.modelo.fit((X - self.mu) / self.sd, y)
        return self

    def predict_proba(self, X):
        p = np.asarray(self.modelo.predict_proba((X - self.mu) / self.sd))
        return p[:, 1] if p.ndim == 2 else p


def candidatos(n_spins, seed=0, incluir_svm=True, preparo="exato", shots=None, cronograma=None):
    """[(nome, tipo, visão, fábrica)]. Clássicos usam as medidas; quânticos o estado
    (amplitude, se preparo exato) ou o próprio circuito de preparo (ising, se preparo
    circuito). shots=None: modelos quânticos com valores esperados exatos (teto
    teórico); com N, o mesmo orçamento total do clássico (2N medições por sistema)."""
    from .backends import BackendShots
    from .kernels import ClassificadorKernel, KernelQuantico
    from .orquestrador import DeGerador
    from .sistema import RFF, Brutos
    from .vqc import VQCClassifier

    class _Gerador:
        def __init__(self, gerador):
            self.c = DeGerador(gerador)

        def fit(self, X, y):
            self.c.ajustar(X, y)
            return self

        def predict_proba(self, X):
            return self.c.predict_proba(X)

    cfg = QuantumConfig(n_spins, "anel", 1)
    out = [("logistica", "classico", "medidas", lambda: _Normalizado(_Gerador(Brutos()))),
           ("rff", "classico", "medidas", lambda: _Normalizado(_Gerador(RFF(escala=0.5, seed=seed))))]
    if incluir_svm:
        try:
            from sklearn.svm import SVC

            class _SVM:
                def fit(self, X, y):
                    self.m = SVC(gamma="scale").fit(X, y)
                    return self

                def predict_proba(self, X):
                    return 1 / (1 + np.exp(-self.m.decision_function(X)))
            out.append(("svm", "classico", "medidas", lambda: _Normalizado(_SVM())))
        except ImportError:
            pass
    if preparo == "circuito":
        visao, cod = "hamiltoniano", {"codificacao": "ising", "cronograma_ising": cronograma}
    else:
        visao, cod = "estado", {"codificacao": "amplitude", "constante_amplitude": None}
    b_kernel = BackendShots(max(1, round(2 * shots / 3)), seed) if shots else None     # 3 bases
    b_vqc = BackendShots(2 * shots, seed + 1) if shots else None                       # 1 base
    sufixo = f" ({2 * shots} medições)" if shots else " (teto: shots infinitos)"
    out += [("vqc" + sufixo, "quantico", visao,
             lambda: VQCClassifier(cfg, otimizador="spsa", epocas=30, seed=seed, backend=b_vqc, **cod)),
            ("kernel-projetado" + sufixo, "quantico", visao,
             lambda: ClassificadorKernel(KernelQuantico(cfg, seed=seed, backend=b_kernel, **cod)))]
    return out


def comparar(dados, n_treino=40, repeticoes=5, seed=0, incluir_svm=True, justo=True, verbose=False):
    """Clássicos (medidas) vs quânticos, nas mesmas divisões. justo=True (padrão):
    os quânticos medem com o MESMO orçamento de shots do clássico."""
    y = dados["y"]
    cands = candidatos(dados["n_spins"], seed, incluir_svm, dados.get("preparo", "exato"),
                       dados["shots"] if justo else None, dados.get("cronograma"))
    accs = {nome: [] for nome, *_ in cands}
    r = np.random.default_rng(seed)
    t0 = time.perf_counter()
    visoes = {"estado": "X_estado", "medidas": "X_medidas", "hamiltoniano": "X_hamiltoniano"}
    for rep in range(repeticoes):
        i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
        n0 = int(np.clip(round(n_treino * len(i0) / len(y)), 2, n_treino - 2))
        tr = np.concatenate([r.choice(i0, n0, replace=False), r.choice(i1, n_treino - n0, replace=False)])
        te = np.setdiff1d(np.arange(len(y)), tr)
        for nome, tipo, visao, fab in cands:
            X = dados[visoes[visao]]
            m = fab().fit(X[tr], y[tr])
            accs[nome].append(float(np.mean((np.asarray(m.predict_proba(X[te])) > 0.5) == y[te])))
            if verbose:
                print(f"  repetição {rep} {nome:32s} {100 * accs[nome][-1]:5.1f}%", flush=True)
    tipos = {nome: tipo for nome, tipo, *_ in cands}
    media = {k: float(np.mean(v)) for k, v in accs.items()}
    mc = max((k for k in accs if tipos[k] == "classico"), key=media.get)
    mq = max((k for k in accs if tipos[k] == "quantico"), key=media.get)
    n_te = len(y) - n_treino
    t = comparar_pareado(np.array(accs[mq]) - np.array(accs[mc]), correcao=n_te / n_treino)
    return {"n_spins": dados["n_spins"], "shots": dados["shots"], "n_treino": n_treino, "repeticoes": repeticoes,
            "preparo": dados.get("preparo", "exato"), "justo": justo,
            "fidelidade_media": float(np.mean(dados.get("fidelidade", [1.0]))),
            "modelos": {k: {"tipo": tipos[k], "media": media[k], "desvio": float(np.std(v))}
                        for k, v in accs.items()},
            "comparacao": {"melhor_classico": mc, "melhor_quantico": mq, "ganho_pp": round(100 * t["ganho"], 2),
                           "ic95_pp": [round(100 * t["ic"][0], 2), round(100 * t["ic"][1], 2)],
                           "p_valor": round(t["p_valor"], 4)},
            "tempo_s": round(time.perf_counter() - t0, 1)}


def texto_comparacao(res):
    regra = ("mesmo orçamento de medições" if res["justo"]
             else "TETO TEÓRICO: quânticos com shots infinitos (comparação injusta)")
    linhas = [f"Fases de Ising: {res['n_spins']} spins, preparo {res['preparo']} (fidelidade "
              f"{res['fidelidade_media']:.2f}), {res['n_treino']} sistemas de treino, {res['repeticoes']} "
              f"repetições; clássicos com {res['shots']} shots por base; {regra}"]
    for nome, v in sorted(res["modelos"].items(), key=lambda kv: -kv[1]["media"]):
        tag = "Q" if v["tipo"] == "quantico" else "C"
        linhas.append(f"  [{tag}] {nome:42s} {100 * v['media']:5.1f}% ± {100 * v['desvio']:4.1f}")
    c = res["comparacao"]
    linhas.append(f"  quântico ({c['melhor_quantico']}) - clássico ({c['melhor_classico']}): {c['ganho_pp']:+.2f} pp, "
                  f"IC95 [{c['ic95_pp'][0]:+.2f}, {c['ic95_pp'][1]:+.2f}], p={c['p_valor']}")
    return "\n".join(linhas)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m quirios_engine.dados_quanticos",
                                 description="Benchmark com dados quânticos: fases do modelo de Ising.")
    ap.add_argument("--n-spins", type=int, default=8)
    ap.add_argument("--amostras", type=int, default=160)
    ap.add_argument("--shots", type=int, nargs="+", default=[10, 100])
    ap.add_argument("--h-min", type=float, default=0.7)
    ap.add_argument("--h-max", type=float, default=1.4)
    ap.add_argument("--preparo", choices=["exato", "circuito"], default="circuito")
    ap.add_argument("--n-treino", type=int, default=40)
    ap.add_argument("--repeticoes", type=int, default=5)
    ap.add_argument("--teto", action="store_true", help="também mostra o teto teórico (shots infinitos)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    for s in a.shots:
        d = fases_ising(a.n_spins, a.amostras, h_min=a.h_min, h_max=a.h_max, shots=s, seed=a.seed,
                        preparo=a.preparo)
        print(texto_comparacao(comparar(d, a.n_treino, a.repeticoes, a.seed)), "\n", flush=True)
        if a.teto:
            print(texto_comparacao(comparar(d, a.n_treino, a.repeticoes, a.seed, justo=False)), "\n", flush=True)


if __name__ == "__main__":
    main()
