"""
Núcleo do circuito quântico: portas, aplicação de portas, permutações CNOT,
geradores de topologia. Tudo parametrizado por n_qubits -- nada hardcoded.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def apply_gate(state, gate, qubit, n):
    """Aplica porta 2x2 no qubit dado, in-place. O(2^n), sem construir matriz NxN."""
    half = 1 << qubit
    stride = 1 << (qubit + 1)
    size = 1 << n
    for b in range(size // stride):
        base = b * stride
        for j in range(half):
            i0 = base + j
            i1 = i0 + half
            a = state[i0]
            b_ = state[i1]
            state[i0] = gate[0, 0] * a + gate[0, 1] * b_
            state[i1] = gate[1, 0] * a + gate[1, 1] * b_


@njit(cache=True)
def apply_perm(state, perm):
    """Aplica permutação (CNOT composto), in-place."""
    tmp = state[perm].copy()
    state[:] = tmp


def gate_H():
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def gate_T():
    return np.diag([1.0 + 0j, np.exp(1j * np.pi / 4)]).astype(complex)


def gate_Ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def gate_Rz(phi):
    return np.diag([np.exp(-1j * phi / 2), np.exp(1j * phi / 2)]).astype(complex)


def build_cnot_perm(ctrl, tgt, n, dim):
    """
    Permutação de índices equivalente a um CNOT(ctrl, tgt).

    Convenção de bit: qubit q <-> bit de posição q (LSB = qubit 0),
    IDÊNTICA à usada em apply_gate (half=1<<qubit, stride=1<<(qubit+1)).
    As duas funções precisam concordar, ou o entrelaçamento acontece
    entre bits diferentes dos que os dados foram codificados -- bug
    real encontrado e corrigido aqui: a versão anterior usava a
    convenção oposta (MSB), produzindo entrelaçamento entre qubits
    diferentes dos pretendidos (validado com par de Bell: dava 0.0
    em vez do 1.0 esperado antes da correção).
    """
    perm = np.arange(dim)
    for i in range(dim):
        if (i >> ctrl) & 1:
            perm[i] = i ^ (1 << tgt)
    return perm


def compose_perms(pairs, n, dim):
    """Compõe uma lista de pares CNOT numa única permutação (1 operação)."""
    p = np.arange(dim)
    for ctrl, tgt in pairs:
        p = build_cnot_perm(ctrl, tgt, n, dim)[p]
    return p


# ── GERADORES DE TOPOLOGIA -- validos para QUALQUER n_qubits >= 2 ──────────────
# Nenhum par hardcoded. Testado para N=4,6,10,14 antes de entrar aqui.

def topologia_anel(n_qubits):
    """Cada qubit conectado ao próximo, em anel. 'Hardware-efficient ansatz'
    padrão da literatura VQE/QAOA. Válida para qualquer N >= 2."""
    if n_qubits < 2:
        raise ValueError("Topologia de entrelaçamento exige pelo menos 2 qubits")
    return [(i, (i + 1) % n_qubits) for i in range(n_qubits)]


def topologia_linear(n_qubits):
    """Cada qubit conectado só ao próximo, sem fechar o anel (n-1 pares)."""
    if n_qubits < 2:
        raise ValueError("Topologia de entrelaçamento exige pelo menos 2 qubits")
    return [(i, i + 1) for i in range(n_qubits - 1)]


def topologia_todos_pares(n_qubits, max_qubits=8):
    """Todos os pares possíveis -- só viável para N pequeno (cresce O(n^2))."""
    if n_qubits > max_qubits:
        raise ValueError(
            f"Topologia todos-os-pares desabilitada acima de {max_qubits} "
            f"qubits (cresce O(n^2), fica caro). Use topologia_anel."
        )
    return [(i, j) for i in range(n_qubits) for j in range(n_qubits) if i != j]


def topologia_blocos(n_qubits, tam_bloco=8, n_conexoes=3):
    """Anel DENTRO de blocos de tam_bloco qubits + n_conexoes CNOTs entre
    blocos vizinhos. Mantém o gradiente forte quando N cresce (o anel
    completo sofre barren plateau a partir de N~12). Validado em N=12 e
    N=16 (Breast Cancer): anel puro 51.7%/60.8% -> blocos 82-92%.
    Use com readout='total' -- com leitura de 1 qubit, a maior parte
    dos blocos fica fora do cone de luz da medicao."""
    if n_qubits < 2:
        raise ValueError("Topologia de entrelaçamento exige pelo menos 2 qubits")
    pares = []
    n_blocos = (n_qubits + tam_bloco - 1) // tam_bloco
    for b in range(n_blocos):
        qb = list(range(b * tam_bloco, min((b + 1) * tam_bloco, n_qubits)))
        if len(qb) >= 2:
            pares += [(qb[i], qb[(i + 1) % len(qb)]) for i in range(len(qb))]
    for b in range(n_blocos - 1):
        a0, b0 = b * tam_bloco, (b + 1) * tam_bloco
        tam_a = min(tam_bloco, n_qubits - a0)
        tam_b = min(tam_bloco, n_qubits - b0)
        pares += [(a0 + k, b0 + k) for k in range(min(n_conexoes, tam_a, tam_b))]
    return pares


# Acima disso o vetor de estado (2^n complexos) passa de ~270 MB.
MAX_QUBITS_VETORIAL = 24


TOPOLOGIAS = {
    "anel": topologia_anel,
    "linear": topologia_linear,
    "todos_pares": topologia_todos_pares,
    "blocos": topologia_blocos,
}


class QuantumConfig:
    """Configuração de um circuito QURIOS -- qualquer N, qualquer topologia."""

    def __init__(self, n_qubits=10, topologia="anel", n_camadas_entrelacamento=2,
                 tam_bloco=8, n_conexoes=3):
        if n_qubits < 2:
            raise ValueError("n_qubits deve ser >= 2")
        if topologia not in TOPOLOGIAS:
            raise ValueError(f"topologia deve ser uma de {list(TOPOLOGIAS.keys())}")

        self.n_qubits = n_qubits
        self.dim = 1 << n_qubits
        self.topologia_nome = topologia
        self.n_camadas = n_camadas_entrelacamento
        self.tam_bloco, self.n_conexoes = tam_bloco, n_conexoes

        gerador = TOPOLOGIAS[topologia]
        if topologia == "blocos":
            pares = gerador(n_qubits, tam_bloco, n_conexoes)
        else:
            pares = gerador(n_qubits)
        # Gera N camadas de entrelaçamento com rotação de offset, pra cobrir
        # mais pares de qubits em topologias esparsas (anel/linear) ao longo
        # de várias camadas, sem virar todos-os-pares.
        # O offset cicla em 1..n-1: offset == n geraria pares (i, i), um
        # "CNOT" de um qubit nele mesmo, que não é unitário (permutação não
        # bijetora). Para n_camadas < n_qubits o comportamento é o da v0.3.
        self.pares_camadas = []   # pares (ctrl, alvo) por camada -- usados pelo backend Qiskit
        for c in range(n_camadas_entrelacamento):
            offset = c % (n_qubits - 1) + 1
            pares_c = [(i, (i + offset) % n_qubits) for i in range(n_qubits)] \
                if topologia == "anel" else pares
            self.pares_camadas.append(list(pares_c))
            # A permutação densa tem 2^n entradas: só é montada quando o
            # simulador vetorial a usa (n pequeno). Backends tensoriais/Qiskit
            # usam apenas pares_camadas e funcionam com 50-100+ qubits.
        self._camadas_cnot = None

    @property
    def camadas_cnot(self):
        if self._camadas_cnot is None:
            if self.n_qubits > MAX_QUBITS_VETORIAL:
                raise ValueError(
                    f"simulação vetorial exata limitada a {MAX_QUBITS_VETORIAL} qubits "
                    f"(2^{self.n_qubits} amplitudes); use BackendMPS ou um backend Qiskit")
            self._camadas_cnot = [compose_perms(p, self.n_qubits, self.dim)
                                  for p in self.pares_camadas]
        return self._camadas_cnot

    def para_dict(self):
        return {"n_qubits": self.n_qubits, "topologia": self.topologia_nome,
                "n_camadas_entrelacamento": self.n_camadas,
                "tam_bloco": self.tam_bloco, "n_conexoes": self.n_conexoes}

    @classmethod
    def de_dict(cls, d):
        return cls(**d)

    def __repr__(self):
        return (f"QuantumConfig(n_qubits={self.n_qubits}, dim={self.dim}, "
                f"topologia='{self.topologia_nome}', n_camadas={self.n_camadas})")
