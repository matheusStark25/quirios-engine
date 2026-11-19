"""
Redes tensoriais: a matemática dos estados quânticos aplicada a dados e
modelos clássicos, e a ponte que leva esses dados para dentro da QPU.

    from quirios_engine.redes_tensoriais import tt_svd_matriz, vetor_para_mps, mps_para_circuito
    python -m quirios_engine.redes_tensoriais circuito      # imagens reais -> circuitos quânticos

Tensor-train (TT), também chamado MPS (vetores) ou MPO (matrizes), é o mesmo
formato do backend MPS do QURIOS: um objeto grande vira uma cadeia de
núcleos pequenos ligados por índices de "ligação" de tamanho chi. É assim
que se simulam 100 qubits num computador comum, e é o que permite:

1. Comprimir modelos de IA: uma matriz de pesos M x N vira d núcleos de
   tamanho chi^2 * m_k * n_k, em vez de M*N números (camadas em ia_tensorial.py).
2. Carregar dados clássicos na QPU: um vetor de 2^n números em MPS com
   ligação chi vira um circuito de n portas de log2(chi)+1 qubits
   (preparo sequencial, Schön et al. 2005), em vez das ~2^n portas de um
   carregamento genérico de amplitudes.

O circuito usa log2(chi) qubits auxiliares para a ligação; eles começam e
terminam em |0>, então o estado dos n qubits de dados é exatamente o MPS.
"""
import argparse
import math
import time

import numpy as np


# ── fatorar dimensões ───────────────────────────────────────────────────────

def fatorar(n, d):
    """n como produto de d fatores o mais parecidos possível (fatores 1 se faltar)."""
    primos, x, p = [], n, 2
    while p * p <= x:
        while x % p == 0:
            primos.append(p)
            x //= p
        p += 1
    if x > 1:
        primos.append(x)
    fatores = [1] * d
    for q in sorted(primos, reverse=True):
        fatores[int(np.argmin(fatores))] *= q
    return sorted(fatores)


# ── TT de matrizes (MPO) ────────────────────────────────────────────────────

def tt_svd_matriz(W, ms, ns, posto):
    """W (M x N) -> núcleos G_k (r_{k-1}, m_k, n_k, r_k), com M = prod(ms), N = prod(ns).
    SVD sequencial truncado no posto (ligação) máximo `posto`."""
    W = np.asarray(W)
    d = len(ms)
    if W.shape != (math.prod(ms), math.prod(ns)) or len(ns) != d:
        raise ValueError(f"W {W.shape} não fatora em {ms} x {ns}")
    T = W.reshape(*ms, *ns).transpose([i for k in range(d) for i in (k, d + k)])
    nucleos, r = [], 1
    C = T.reshape(ms[0] * ns[0], -1)
    for k in range(d - 1):
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        rk = int(min(posto, len(S)))
        nucleos.append(U[:, :rk].reshape(r, ms[k], ns[k], rk))
        C = (S[:rk, None] * Vt[:rk]).reshape(rk * ms[k + 1] * ns[k + 1], -1)
        r = rk
    nucleos.append(C.reshape(r, ms[-1], ns[-1], 1))
    return nucleos


def tt_para_matriz(nucleos):
    """Reconstrói a matriz densa (M x N) a partir dos núcleos TT."""
    d = len(nucleos)
    T = nucleos[0]
    for G in nucleos[1:]:
        T = np.tensordot(T, G, axes=([-1], [0]))
    T = T.reshape(T.shape[1:-1])                          # (m1, n1, m2, n2, ...)
    T = T.transpose([2 * k for k in range(d)] + [2 * k + 1 for k in range(d)])
    ms = [G.shape[1] for G in nucleos]
    ns = [G.shape[2] for G in nucleos]
    return T.reshape(math.prod(ms), math.prod(ns))


def params_tt(ms, ns, posto):
    """Número de parâmetros dos núcleos TT com ligação máxima `posto`."""
    d, r, total = len(ms), [1], 0
    for k in range(d - 1):                                 # postos limitados pelas duas pontas
        esq = math.prod(m * n for m, n in zip(ms[:k + 1], ns[:k + 1]))
        dir_ = math.prod(m * n for m, n in zip(ms[k + 1:], ns[k + 1:]))
        r.append(min(posto, esq, dir_))
    r.append(1)
    for k in range(d):
        total += r[k] * ms[k] * ns[k] * r[k + 1]
    return total


# ── MPS de vetores e circuito de preparo ────────────────────────────────────

def vetor_para_mps(v, posto):
    """Vetor de 2^n amplitudes -> núcleos (r_{k-1}, 2, r_k) com ligação <= posto.
    O qubit 0 é o bit MENOS significativo do índice (convenção do Qiskit)."""
    v = np.asarray(v, dtype=complex)
    n = int(round(math.log2(len(v))))
    if 2 ** n != len(v):
        raise ValueError("o vetor precisa ter 2^n entradas")
    T = v.reshape([2] * n).transpose(list(range(n))[::-1])          # eixo k = qubit k
    nucleos, r = [], 1
    C = T.reshape(2, -1)
    for k in range(n - 1):
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        rk = int(min(posto, len(S)))
        nucleos.append(U[:, :rk].reshape(r, 2, rk))
        C = (S[:rk, None] * Vt[:rk]).reshape(rk * 2, -1)
        r = rk
    nucleos.append(C.reshape(r, 2, 1))
    return nucleos


def mps_para_vetor(nucleos):
    T = nucleos[0]
    for G in nucleos[1:]:
        T = np.tensordot(T, G, axes=([-1], [0]))
    T = T.reshape([2] * len(nucleos))                     # eixo k = qubit k
    return T.transpose(list(range(len(nucleos)))[::-1]).reshape(-1)


def _canonica_direita(nucleos):
    """Forma canônica à direita (sum_{s,b} B[a,s,b] B*[a',s,b] = delta), normalizada."""
    B = [np.array(G, dtype=complex) for G in nucleos]
    for k in range(len(B) - 1, 0, -1):
        r0, _, r1 = B[k].shape
        Q, R = np.linalg.qr(B[k].reshape(r0, 2 * r1).conj().T)     # M^H = Q R  ->  M = R^H Q^H
        B[k] = Q.conj().T.reshape(-1, 2, r1)
        B[k - 1] = np.tensordot(B[k - 1], R.conj().T, axes=([2], [0]))
    B[0] = B[0] / np.linalg.norm(B[0])
    return B


def _completar_unitaria(V, colunas_validas):
    """Matriz unitária cujas colunas `colunas_validas` são as de V (ortonormais)."""
    from scipy.linalg import null_space
    D = V.shape[0]
    U = np.zeros((D, D), dtype=complex)
    U[:, colunas_validas] = V[:, colunas_validas]
    resto = null_space(V[:, colunas_validas].conj().T)
    livres = [j for j in range(D) if j not in set(colunas_validas)]
    U[:, livres] = resto[:, :len(livres)]
    return U


def mps_para_circuito(nucleos):
    """Circuito Qiskit que prepara o MPS: qubits 0..n-1 de dados, depois b = log2(chi)
    qubits de ligação (começam e terminam em |0>). Uma porta de b+1 qubits por sítio."""
    from qiskit import QuantumCircuit
    B = _canonica_direita(nucleos)
    n = len(B)
    chi = max(G.shape[2] for G in B[:-1]) if n > 1 else 1
    b = max(1, math.ceil(math.log2(chi))) if chi > 1 else 0
    dim = 2 ** b
    qc = QuantumCircuit(n + b)
    for k, G in enumerate(B):
        r0, _, r1 = G.shape
        if b == 0:                                          # produto: rotação de 1 qubit
            qc.initialize(G.reshape(2) / np.linalg.norm(G), [k])
            continue
        V = np.zeros((2 * dim, 2 * dim), dtype=complex)
        for a in range(r0):
            for s in range(2):
                for beta in range(r1):
                    V[beta + dim * s, a] = G[a, s, beta]        # entrada |a>_lig |0>_dado
        U = _completar_unitaria(V, list(range(r0)))
        qc.unitary(U, list(range(n, n + b)) + [k], label=f"MPS{k}")
    return qc


def fidelidade(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real))


def contar_cnots(qc, otimizacao=1):
    from qiskit import transpile
    t = transpile(qc, basis_gates=["cx", "u"], optimization_level=otimizacao)
    return int(t.count_ops().get("cx", 0))


# ── experimento: imagens reais dentro da QPU ────────────────────────────────

def carregar_mnist():
    """MNIST (70.000 dígitos 28x28, pixels 0..255) da OpenML, em cache local .npz
    (a primeira vez baixa ~15 MB e demora; depois é imediato)."""
    from pathlib import Path
    from sklearn.datasets import fetch_openml, get_data_home
    cache = Path(get_data_home()) / "quirios_mnist.npz"
    if cache.exists():
        d = np.load(cache)
        return d["X"], d["y"]
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="liac-arff")
    X, y = X.astype(np.uint8), y.astype(np.int64)
    np.savez_compressed(cache, X=X, y=y)
    return X, y


def imagens_mnist(k=8, seed=0):
    """k dígitos do MNIST (28x28) completados para 32x32 = 2^10 amplitudes, normalizados."""
    X, _ = carregar_mnist()
    idx = np.random.default_rng(seed).choice(len(X), k, replace=False)
    out = []
    for i in idx:
        im = np.zeros((32, 32))
        im[2:30, 2:30] = X[i].reshape(28, 28) / 255.0
        v = im.reshape(-1)
        out.append(v / np.linalg.norm(v))
    return np.array(out)


def experimento_circuito(imagens, postos=(2, 4, 8), exato=True):
    """Fidelidade e CNOTs para carregar cada imagem com MPS de ligação chi,
    contra o carregamento genérico de amplitudes (StatePreparation do Qiskit)."""
    from qiskit.circuit.library import StatePreparation
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    t0 = time.perf_counter()
    n = int(round(math.log2(imagens.shape[1])))
    linhas = []
    for chi in postos:
        fids, cxs, fids_circ = [], [], []
        for v in imagens:
            mps = vetor_para_mps(v, chi)
            fids.append(fidelidade(mps_para_vetor(mps), v))
            qc = mps_para_circuito(mps)
            cxs.append(contar_cnots(qc))
            sv = Statevector(qc).data.reshape(-1, 2 ** n)[0]            # auxiliares em |0>
            fids_circ.append(fidelidade(sv, v))
        linhas.append({"metodo": f"MPS chi={chi}", "chi": chi, "fidelidade": float(np.mean(fids)),
                       "fidelidade_circuito": float(np.mean(fids_circ)), "cnots": float(np.mean(cxs)),
                       "qubits": n + max(1, math.ceil(math.log2(chi)))})
    if exato:
        qc = QuantumCircuit(n)
        qc.append(StatePreparation(imagens[0]), range(n))
        linhas.append({"metodo": "carregamento genérico (exato)", "chi": None, "fidelidade": 1.0,
                       "fidelidade_circuito": 1.0, "cnots": float(contar_cnots(qc.decompose(reps=3))), "qubits": n})
    return {"n_qubits_dados": n, "imagens": len(imagens), "linhas": linhas,
            "tempo_s": round(time.perf_counter() - t0, 1)}


def texto_circuito(res):
    linhas = [f"Imagens do MNIST (32x32 = 2^{res['n_qubits_dados']} amplitudes) carregadas num estado quântico, "
              f"média de {res['imagens']} imagens"]
    for l in res["linhas"]:
        linhas.append(f"  {l['metodo']:32s} fidelidade {l['fidelidade_circuito']:.3f}   CNOTs {l['cnots']:7.0f}   "
                      f"qubits {l['qubits']}")
    return "\n".join(linhas)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m quirios_engine.redes_tensoriais",
                                 description="Redes tensoriais: dados clássicos dentro da QPU.")
    sub = ap.add_subparsers(dest="acao", required=True)
    pc = sub.add_parser("circuito", help="carrega imagens do MNIST em circuitos via MPS")
    pc.add_argument("--imagens", type=int, default=8)
    pc.add_argument("--postos", type=int, nargs="+", default=[2, 4, 8])
    a = ap.parse_args(argv)
    print(texto_circuito(experimento_circuito(imagens_mnist(a.imagens), a.postos)))


if __name__ == "__main__":
    main()
