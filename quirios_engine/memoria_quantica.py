"""
Aprender com dados quânticos usando MEMÓRIA QUÂNTICA: onde o quântico vence
o clássico com o mesmo número de experimentos.

    from quirios_engine.memoria_quantica import comparar, texto_comparacao
    print(texto_comparacao(comparar(n_qubits=(4, 5, 6), copias=(8, 16, 32, 64))))

    python -m quirios_engine.memoria_quantica --n-qubits 4 5 6 --copias 8 16 32 64 128

O dado: uma fonte (um sensor, um simulador quântico, outro processador)
entrega cópias de um estado de n qubits

    rho_P = (I + eps P) / 2^n,      P um operador de Pauli desconhecido.

A tarefa: dizer se o estado é SIMÉTRICO POR REVERSÃO TEMPORAL (rho real,
rho* = rho, o que acontece quando P tem um número par de Y) ou não.

Sem memória quântica (o "clássico"): cada cópia é medida assim que chega,
em qualquer base de Pauli, e só os bits ficam guardados. A decisão usa a
inferência bayesiana EXATA sobre todos os 4^n - 1 Paulis possíveis, a melhor
decodificação para essas medidas.

Com memória quântica (o QURIOS): duas cópias ficam guardadas e são medidas
JUNTAS na base de Bell (CNOT + H entre os pares de qubits). A medida devolve
um Pauli Q com probabilidade 4^-n (1 + eps^2 s c(P,Q)), em que s = +1 para
o estado simétrico e -1 para o outro, e c = +1 se Q comuta com P. Cada
experimento revela uma restrição linear sobre P, e O(n) experimentos bastam.

Resultado medido (eps = 0,9, 200 estados por ponto, mesmo número de cópias):
com 6 qubits e 64 cópias, 98% com memória contra 49% sem (chute: 50%); com
memória, 64 cópias bastam para 90% de 4 a 6 qubits, e sem memória nem 256
bastam a partir de 4 qubits. É a família de
tarefas em que Huang et al. (Science 376, 1182, 2022) e Chen, Cotler,
Huang e Li (FOCS 2021) provaram separação exponencial entre aprender com e
sem memória quântica, medida no processador do Google.

O que isto NÃO é: aceleração de dados clássicos (treinar um modelo de
linguagem, por exemplo). A vantagem existe quando o dado já nasce quântico e
chega ao processador sem passar por medidas.
"""
import argparse
import math
import time

import numpy as np

EPS_PADRAO = 0.9


# ── Paulis como pares de bits (x, z): X^x Z^z por qubit ─────────────────────

def _paridade(v):
    return (np.bitwise_count(np.asarray(v, dtype=np.int64)) & 1).astype(np.int64)


def todos_paulis(n):
    """(x, z) de todos os 4^n - 1 Paulis de n qubits, sem a identidade."""
    i = np.arange(1, 4 ** n, dtype=np.int64)
    return i & ((1 << n) - 1), i >> n


def comuta(x1, z1, x2, z2):
    """+1 se os Paulis comutam, -1 se anticomutam (produto simplético)."""
    return 1 - 2 * _paridade((np.asarray(x1) & z2) ^ (np.asarray(z1) & x2))


def simetria_T(x, z):
    """+1 se P^T = P (número par de Y: rho_P é real), -1 se P^T = -P."""
    return 1 - 2 * _paridade(np.asarray(x) & z)


def nome_pauli(x, z, n):
    return "".join("IXZY"[((int(x) >> q) & 1) | (((int(z) >> q) & 1) << 1)] for q in range(n))


# ── fonte: as duas maneiras de observar as cópias ──────────────────────────

def amostras_bell(n, x, z, eps, m, rng):
    """m experimentos com 2 cópias medidas na base de Bell: m Paulis Q (x, z),
    sorteados da distribuição exata p(Q) = 4^-n (1 + eps^2 s c(P,Q))."""
    s = simetria_T(x, z)
    out = np.empty((m, 2), dtype=np.int64)
    k = 0
    while k < m:
        qx, qz = rng.integers(0, 1 << n, 2)
        if rng.random() < (1 + eps ** 2 * s * comuta(x, z, qx, qz)) / 2:
            out[k] = qx, qz
            k += 1
    return out


def bases_aleatorias(k, n, rng):
    """k bases de medida, uma letra X/Y/Z por qubit, como pares (x, z)."""
    letras = rng.integers(1, 4, (k, n))                       # 1 = X, 2 = Z, 3 = Y
    pesos = 1 << np.arange(n, dtype=np.int64)
    return ((letras & 1) * pesos).sum(1), (((letras >> 1) & 1) * pesos).sum(1)


def amostras_uma_copia(n, x, z, eps, k, rng):
    """k cópias, cada uma medida sozinha numa base de Pauli aleatória:
    (xb, zb, bits). Distribuição exata: p(b|B) = 2^-n (1 + eps [B casa com P]
    (-1)^{paridade de b no suporte de P})."""
    xb, zb = bases_aleatorias(k, n, rng)
    bits = rng.integers(0, 1 << n, k)
    sup = int(x) | int(z)
    casa = (sup & ((x ^ xb) | (z ^ zb))) == 0
    alvo = np.where(rng.random(k) < (1 + eps) / 2, 0, 1)      # paridade sorteada: 0 com prob (1+eps)/2
    errada = casa & (_paridade(bits & sup) != alvo)
    bits[errada] ^= sup & -sup                                 # troca um bit do suporte
    return xb, zb, bits


# ── decodificação bayesiana exata sobre todos os Paulis ─────────────────────

def _decidir(log_ver, S):
    """P(simétrico | dados) com prior uniforme sobre os Paulis."""
    w = np.exp(log_ver - log_ver.max())
    return float((w * (S > 0)).sum() / w.sum())


def decidir_com_memoria(Q, n, eps=EPS_PADRAO):
    """Probabilidade de o estado ser simétrico, dados os Paulis de Bell Q (m x 2)."""
    PX, PZ = todos_paulis(n)
    S = simetria_T(PX, PZ)
    e = min(eps, 0.999) ** 2
    c = comuta(PX[:, None], PZ[:, None], Q[None, :, 0], Q[None, :, 1]) * S[:, None]
    return _decidir(np.log1p(e * c).sum(1), S)


def decidir_sem_memoria(xb, zb, bits, n, eps=EPS_PADRAO):
    """Probabilidade de o estado ser simétrico, dadas k medidas de uma cópia."""
    PX, PZ = todos_paulis(n)
    S = simetria_T(PX, PZ)
    sup = PX | PZ
    casa = (sup[:, None] & ((PX[:, None] ^ xb[None]) | (PZ[:, None] ^ zb[None]))) == 0
    par = 1 - 2 * _paridade(sup[:, None] & bits[None])
    return _decidir(np.log1p(min(eps, 0.999) * casa * par).sum(1), S)


# ── circuitos (Qiskit): o mesmo experimento numa QPU ───────────────────────

def _preparar(qc, qubits, x, z, bits):
    """Uma cópia de rho_P a partir de bits clássicos sorteados: o qubit pivô
    em |bit> e os outros em |bit aleatório>; depois V com V Z_pivo V^dag = P.
    Na média sobre os bits, o estado é exatamente (I + eps P)/2^n."""
    n = len(qubits)
    sup = [q for q in range(n) if ((x | z) >> q) & 1]
    k = sup[0]
    for q in range(n):
        if (bits >> q) & 1:
            qc.x(qubits[q])
    for j in sup[1:]:
        qc.cx(qubits[j], qubits[k])                    # Z_k -> Z_j Z_k
    for q in sup:
        xq, zq = (x >> q) & 1, (z >> q) & 1
        if xq and not zq:
            qc.h(qubits[q])                            # Z -> X
        elif xq and zq:
            qc.h(qubits[q]); qc.s(qubits[q])           # Z -> Y


def _bits_da_copia(n, x, z, eps, rng):
    k = [q for q in range(n) if ((x | z) >> q) & 1][0]
    bits = int(rng.integers(0, 1 << n))
    pivo = 0 if rng.random() < (1 + eps) / 2 else 1
    return (bits & ~(1 << k)) | (pivo << k)


def _preparar_produto(qc, qubits, x, z, bits):
    """Uma cópia de rho_P SEM portas de 2 qubits: cada qubit do suporte de P vai para
    um autoestado do seu Pauli (bit 1 = autovalor -1) e os outros para |bit>. Como
    P = produto dos Paulis de cada qubit, o autovalor de P é (-1)^(paridade dos bits
    do suporte); sorteando essa paridade como em _bits_produto, a média é exatamente
    (I + eps P)/2^n, o mesmo estado do preparo com CNOTs, com muito menos ruído."""
    for q in range(len(qubits)):
        if (bits >> q) & 1:
            qc.x(qubits[q])
        xq, zq = (x >> q) & 1, (z >> q) & 1
        if xq and not zq:
            qc.h(qubits[q])                            # |0>,|1> -> |+>,|->  (autoestados de X)
        elif xq and zq:
            qc.h(qubits[q]); qc.s(qubits[q])           # |0>,|1> -> |+i>,|-i> (autoestados de Y)


def _bits_produto(n, x, z, eps, rng):
    """Bits do preparo sem CNOT: a paridade dos bits do suporte é 0 (autovalor +1 de P)
    com probabilidade (1 + eps)/2; dentro disso, todos os bits são uniformes."""
    sup = [q for q in range(n) if ((x | z) >> q) & 1]
    bits = int(rng.integers(0, 1 << n))
    alvo = 0 if rng.random() < (1 + eps) / 2 else 1
    if (bin(bits & (x | z)).count("1") & 1) != alvo:
        bits ^= 1 << sup[0]
    return bits


def _uma_copia(qc, qubits, n, x, z, eps, rng, preparo):
    if preparo == "produto":
        _preparar_produto(qc, qubits, int(x), int(z), _bits_produto(n, int(x), int(z), eps, rng))
    elif preparo == "cnot":
        _preparar(qc, qubits, int(x), int(z), _bits_da_copia(n, x, z, eps, rng))
    else:
        raise ValueError("preparo deve ser 'cnot' ou 'produto'")


def circuito_com_memoria(n, x, z, eps, rng, preparo="cnot"):
    """Duas cópias (qubits 0..n-1 e n..2n-1) medidas juntas na base de Bell.
    preparo="produto" fabrica as cópias sem portas de 2 qubits (mesmo estado médio)."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2 * n)
    _uma_copia(qc, list(range(n)), n, x, z, eps, rng, preparo)
    _uma_copia(qc, list(range(n, 2 * n)), n, x, z, eps, rng, preparo)
    for i in range(n):
        qc.cx(i, n + i)
        qc.h(i)
    qc.measure_all()
    return qc


def pauli_de_bell(contagem, n):
    """Bits de uma medida de Bell -> Pauli Q (x, z): a cópia A dá z, a B dá x."""
    s = contagem.replace(" ", "")
    bit = lambda q: int(s[len(s) - 1 - q])                     # qubit 0 à direita
    return (sum(bit(n + i) << i for i in range(n)), sum(bit(i) << i for i in range(n)))


def circuito_sem_memoria(n, x, z, eps, xb, zb, rng, preparo="cnot"):
    """Uma cópia medida sozinha na base de Pauli (xb, zb)."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    _uma_copia(qc, list(range(n)), n, x, z, eps, rng, preparo)
    for q in range(n):
        xq, zq = (int(xb) >> q) & 1, (int(zb) >> q) & 1
        if xq and zq:
            qc.sdg(q); qc.h(q)                          # base Y
        elif xq:
            qc.h(q)                                     # base X
    qc.measure_all()
    return qc


# ── comparação ──────────────────────────────────────────────────────────────

def _wilson(acertos, total, z=1.96):
    if total == 0:
        return 0.0, 1.0
    p = acertos / total
    d = 1 + z * z / total
    c = (p + z * z / (2 * total)) / d
    m = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return max(0.0, c - m), min(1.0, c + m)


def sortear_instancias(n, k, rng):
    """k Paulis aleatórios com as duas classes equilibradas."""
    PX, PZ = todos_paulis(n)
    S = simetria_T(PX, PZ)
    sim, anti = np.where(S > 0)[0], np.where(S < 0)[0]
    escolha = [rng.choice(sim if i % 2 == 0 else anti) for i in range(k)]
    rng.shuffle(escolha)
    return [(int(PX[i]), int(PZ[i]), int(S[i])) for i in escolha]


def comparar(n_qubits=(4, 5, 6), copias=(8, 16, 32, 64), eps=EPS_PADRAO, instancias=200, seed=0):
    """Acurácia com e sem memória quântica, MESMO número de cópias do estado:
    c cópias = c/2 experimentos de Bell (com memória) ou c medidas de uma
    cópia (sem memória)."""
    t0 = time.perf_counter()
    linhas = []
    for n in n_qubits:
        rng = np.random.default_rng([seed, n])
        inst = sortear_instancias(n, instancias, rng)
        for c in copias:
            ac_q = ac_c = 0
            for x, z, s in inst:
                y = s > 0
                Q = amostras_bell(n, x, z, eps, max(1, c // 2), rng)
                ac_q += (decidir_com_memoria(Q, n, eps) > 0.5) == y
                xb, zb, bits = amostras_uma_copia(n, x, z, eps, c, rng)
                ac_c += (decidir_sem_memoria(xb, zb, bits, n, eps) > 0.5) == y
            for metodo, ac in (("com memória quântica", ac_q), ("sem memória (clássico)", ac_c)):
                lo, hi = _wilson(ac, instancias)
                linhas.append({"n_qubits": n, "copias": c, "metodo": metodo, "acuracia": ac / instancias,
                               "ic95": [lo, hi]})
    return {"eps": eps, "instancias": instancias, "linhas": linhas, "copias_para_90": copias_para(linhas, 0.9),
            "tempo_s": round(time.perf_counter() - t0, 1)}


def copias_para(linhas, alvo=0.9):
    """Menor número de cópias medido em que cada método atinge `alvo` (None: não atingiu)."""
    out = []
    for n in sorted({l["n_qubits"] for l in linhas}):
        for metodo in ("com memória quântica", "sem memória (clássico)"):
            ok = [l["copias"] for l in linhas if l["n_qubits"] == n and l["metodo"] == metodo
                  and l["acuracia"] >= alvo]
            out.append({"n_qubits": n, "metodo": metodo, "copias": min(ok) if ok else None})
    return out


def texto_comparacao(res):
    linhas = [f"Simetria por reversão temporal de rho = (I + {res['eps']} P)/2^n, {res['instancias']} estados por "
              f"ponto, mesmo número de cópias para os dois lados"]
    for n in sorted({l["n_qubits"] for l in res["linhas"]}):
        partes = []
        for c in sorted({l["copias"] for l in res["linhas"]}):
            q = next(l for l in res["linhas"] if l["n_qubits"] == n and l["copias"] == c and "com" in l["metodo"])
            k = next(l for l in res["linhas"] if l["n_qubits"] == n and l["copias"] == c and "sem" in l["metodo"])
            partes.append(f"{c:4d} cópias: Q {100 * q['acuracia']:5.1f}% C {100 * k['acuracia']:5.1f}%")
        linhas.append(f"  n={n} | " + " | ".join(partes))
    alvo = {(d["n_qubits"], d["metodo"]): d["copias"] for d in res["copias_para_90"]}
    for n in sorted({d["n_qubits"] for d in res["copias_para_90"]}):
        q, c = alvo[(n, "com memória quântica")], alvo[(n, "sem memória (clássico)")]
        linhas.append(f"  cópias para 90% com n={n}: com memória {q or 'não atingiu'}, sem memória "
                      f"{c or 'não atingiu'}")
    return "\n".join(linhas)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m quirios_engine.memoria_quantica",
                                 description="Aprender com e sem memória quântica: mesmo número de cópias.")
    ap.add_argument("--n-qubits", type=int, nargs="+", default=[4, 5, 6])
    ap.add_argument("--copias", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--eps", type=float, default=EPS_PADRAO)
    ap.add_argument("--instancias", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if max(a.n_qubits) > 8:
        raise SystemExit("--n-qubits até 8 (a decodificação exata percorre 4^n Paulis)")
    print(texto_comparacao(comparar(a.n_qubits, a.copias, a.eps, a.instancias, a.seed)))


if __name__ == "__main__":
    main()
