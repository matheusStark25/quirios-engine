"""
Ansatz: a descrição única de um circuito QURIOS. Todo backend (simulação
exata, MPS, Qiskit, IBM) executa o mesmo Ansatz com o mesmo vetor de
parâmetros, então trocar de backend nunca muda o modelo.

O Ansatz é um GENOMA: cada escolha abaixo é um gene que a busca evolutiva
(evolucao.py) pode mutar. Os valores padrão reproduzem exatamente o
circuito da v0.2-v0.5.

    codificacao   "angulo"    H, T e Ry(pi*x) por qubit (padrão)
                  "iqp"       mapa ZZ de Havlicek et al. (Nature 2019), por camada:
                              H, Rz(2*pi*x_i), RZZ(2*pi^2*(1-x_i)*(1-x_j)) nos pares
                  "amplitude" estado inicial [x, c] / ||[x, c]|| (sem dados nas camadas)
                  "ising"     o dado é um SISTEMA FÍSICO: x = (J_0..J_{n-1}, h) e o circuito
                              prepara o estado fundamental de H = -sum J_i Z_i Z_i+1 - h sum X_i
                              a partir de |+>^n com um cronograma tipo QAOA de P passos:
                              exp(+i g_k sum J_i Z_i Z_i+1) exp(+i b_k h sum X) (portas RZZ e RX,
                              nativas/fracionárias no IBM Heron). O cronograma padrão (6 passos,
                              CRONOGRAMA_ISING) foi otimizado em simulação: fidelidade média
                              0,93 com 8 spins usando 48 RZZ (a rampa linear com o mesmo
                              número de portas fica abaixo de 0,4). Os parâmetros (J, h) só
                              PREPARAM o estado; o modelo só vê o estado.
    rotacoes      portas treináveis por qubit em cada camada, ex. ("ry", "rz") (padrão),
                  ("rz", "ry", "rz") = SU(2) completo, ("rx", "ry"), ("ry",)
    emaranhador   "cnot" (padrão), "cz", ou "rzz" -- RZZ(theta) TREINÁVEL, a porta
                  fracionária nativa dos processadores IBM Heron (sem decompor em CNOTs)
    leitura       o que se mede no fim:
                  "z"      P(qubit q = 1)                                  (n)
                  "z_zz"   P(qubit q = 1) + <Z_i Z_i+1> no anel (padrão)    (2n)
                  "pauli"  <X_q>, <Y_q>, <Z_q> + <Z_i Z_i+1>                (4n)
                  "pauli2" <X_q>, <Y_q>, <Z_q> + os 9 Paulis de 2 qubits    (12n)
                           nos pares do anel (estimados com sombras clássicas)

Estrutura de cada camada l = 0..L-1:

    [dados da camada l] -> rotações treináveis -> emaranhador
    ... e no fim Ry(alpha[q]) (camada_pos=True)

    Na codificação por ângulo, a camada l lê a feature (l*n + q) % d: d = n
    reenvia os mesmos dados (data re-uploading); d = L*n entrega um bloco
    novo de features por camada.

Codificação por amplitude: [x, c] / ||[x, c]|| com c = constante_amplitude
(padrão 1). Sem c, x e -x viram o MESMO estado (só fase global) e nenhuma
medida os distingue -- no Breast Cancer padronizado a acurácia caía para
47,5%. constante_amplitude=None usa x/||x||. d features cabem em
ceil(log2(d + 1)) qubits; em hardware a preparação custa ~2^n CNOTs.

Vetor de parâmetros (plano):
    [rotação 0 (L*n), rotação 1 (L*n), ..., RZZ (L*pares) se rzz, alpha (n)]
Cada parâmetro está em exatamente uma rotação de Pauli (Rx/Ry/Rz/RZZ), então
a regra de parameter-shift (+-pi/2) é exata para todos.
"""
import numpy as np

from .core import QuantumConfig, apply_gate, apply_perm, gate_H, gate_T, gate_Ry, gate_Rz

CODIFICACOES = ("angulo", "iqp", "amplitude", "ising")

# Cronograma (gamma_k, beta_k) da preparação do estado de Ising, otimizado por
# dados_quanticos.otimizar_cronograma(n_spins=6, passos=6): fidelidade média 0,98
# no treino (6 spins) e 0,93 com 8 spins.
CRONOGRAMA_ISING = ((0.119, 0.279, 0.381, 0.432, 0.459, 0.398),
                    (0.708, 0.693, 0.669, 0.619, 0.547, 0.260))
EMARANHADORES = ("cnot", "cz", "rzz")
LEITURAS = ("z", "z_zz", "pauli", "pauli2")
PORTAS_ROTACAO = ("rx", "ry", "rz")
INICIALIZACOES = ("padrao", "pequena")
_PADRAO_ROTACOES = ("ry", "rz")


def gate_Rx(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


_PORTAS = {"rx": gate_Rx, "ry": gate_Ry, "rz": gate_Rz}


def pares_zz(n):
    """Pares vizinhos do anel usados na leitura de correlações."""
    return [(i, (i + 1) % n) for i in range(n)]


def _sinal(idx, q):
    return 1 - 2 * ((idx >> q) & 1)


def esperanca_pauli(st, ops, qubits):
    """<psi| P |psi> para um produto de Paulis (ex. "XY" nos qubits (2, 5))."""
    idx = np.arange(len(st))
    flip, fase = 0, np.ones(len(st), dtype=complex)
    for op, q in zip(ops, qubits):
        if op == "Z":
            fase = fase * _sinal(idx, q)
        elif op == "X":
            flip |= 1 << q
        elif op == "Y":
            flip |= 1 << q
            fase = fase * (1j * _sinal(idx, q))           # Y|b> = i(-1)^b |1-b>
        else:
            raise ValueError(f"Pauli desconhecido: {op}")
    return float(np.real(np.sum(np.conj(st[idx ^ flip]) * fase * st)))


class Ansatz:
    def __init__(self, config, n_features=None, n_camadas=None, codificacao="angulo",
                 camada_pos=True, constante_amplitude=1.0, rotacoes=_PADRAO_ROTACOES,
                 emaranhador="cnot", leitura="z_zz", cronograma_ising=None):
        if codificacao not in CODIFICACOES:
            raise ValueError(f"codificacao deve ser uma de {CODIFICACOES}")
        if emaranhador not in EMARANHADORES:
            raise ValueError(f"emaranhador deve ser um de {EMARANHADORES}")
        if leitura not in LEITURAS:
            raise ValueError(f"leitura deve ser uma de {LEITURAS}")
        rotacoes = tuple(rotacoes)
        if not rotacoes or any(r not in PORTAS_ROTACAO for r in rotacoes):
            raise ValueError(f"rotacoes deve ser uma sequência não vazia de {PORTAS_ROTACAO}")
        n = config.n_qubits
        extra = 1 if (codificacao == "amplitude" and constante_amplitude is not None) else 0
        if n_features is None:
            n_features = {"amplitude": (1 << n) - extra, "ising": n + 1}.get(codificacao, n)
        if codificacao == "ising" and n_features != n + 1:
            raise ValueError(f"ising: x = (J_0..J_{n - 1}, h) tem {n + 1} valores, não {n_features}")
        if n_features < 1:
            raise ValueError("n_features deve ser >= 1")
        if codificacao == "amplitude" and n_features + extra > (1 << n):
            raise ValueError(f"amplitude: {n_features} features não cabem em {n} qubits "
                             f"(máximo {(1 << n) - extra}); use "
                             f"{qubits_para_amplitude(n_features, constante_amplitude is not None)} qubits")
        if n_camadas is None:
            n_camadas = 1 if codificacao in ("amplitude", "ising") else -(-n_features // n)
        if n_camadas < 1:
            raise ValueError("n_camadas deve ser >= 1")
        self.config = config
        self.n_features = int(n_features)
        self.n_camadas = int(n_camadas)
        self.codificacao = codificacao
        self.camada_pos = bool(camada_pos)
        self.constante_amplitude = None if constante_amplitude is None else float(constante_amplitude)
        self.rotacoes = rotacoes
        self.emaranhador = emaranhador
        self.leitura = leitura
        g, b = cronograma_ising if cronograma_ising is not None else CRONOGRAMA_ISING
        if len(g) != len(b) or len(g) < 1:
            raise ValueError("cronograma_ising = (gammas, betas) com o mesmo número (>= 1) de passos")
        self.cronograma_ising = (tuple(float(v) for v in g), tuple(float(v) for v in b))

    # ── geometria ───────────────────────────────────────────────────────────

    @property
    def n_qubits(self):
        return self.config.n_qubits

    @property
    def padrao(self):
        """True se o genoma é o circuito clássico do QURIOS (v0.2-v0.5)."""
        return (self.rotacoes == _PADRAO_ROTACOES and self.emaranhador == "cnot"
                and self.codificacao == "angulo")

    def pares_emaranhamento(self):
        """Pares (ctrl, alvo) de todas as subcamadas de CNOT/CZ/RZZ, em ordem."""
        return [p for sub in self.config.pares_camadas for p in sub]

    def pares_iqp(self):
        sub = self.config.pares_camadas[0] if self.config.pares_camadas else pares_zz(self.n_qubits)
        return sorted({tuple(sorted(p)) for p in sub})

    @property
    def n_params(self):
        n, L = self.n_qubits, self.n_camadas
        rzz = L * len(self.pares_emaranhamento()) if self.emaranhador == "rzz" else 0
        return len(self.rotacoes) * L * n + rzz + (n if self.camada_pos else 0)

    def portas_2q(self):
        """Portas de 2 qubits por circuito (custo em hardware, antes do roteamento)."""
        por_camada = len(self.pares_emaranhamento())
        if self.codificacao == "iqp":
            por_camada += len(self.pares_iqp())
        preparo = len(self.cronograma_ising[0]) * self.n_qubits if self.codificacao == "ising" else 0
        return self.n_camadas * por_camada + preparo

    def _passos_adiabaticos(self, x):
        """[(ângulos RZZ por par do anel, ângulo RX)] do preparo do estado de Ising:
        exp(+i g J ZZ) = RZZ(-2 g J), exp(+i b h X) = RX(-2 b h)."""
        n = self.n_qubits
        J, h = x[:n], x[n]
        return [(-2 * g * J, -2 * b * h) for g, b in zip(*self.cronograma_ising)]

    def indices_dados(self):
        """Matriz (L, n): qual feature cada qubit lê em cada camada (ângulo/IQP)."""
        n, d = self.n_qubits, self.n_features
        return (np.arange(self.n_camadas)[:, None] * n + np.arange(n)[None, :]) % d

    def features_usadas(self):
        """Features que efetivamente entram no circuito (ângulo com L*n < d descarta o resto)."""
        if self.codificacao == "amplitude":
            return list(range(self.n_features))
        return sorted(set(self.indices_dados().ravel().tolist()))

    # ── parâmetros ──────────────────────────────────────────────────────────

    def blocos(self, params):
        """{"rot": (R, L, n), "rzz": (L, pares) ou None, "alpha": (n,) ou None}"""
        params = np.asarray(params, float)
        if params.shape != (self.n_params,):
            raise ValueError(f"esperado vetor com {self.n_params} parâmetros, recebido {params.shape}")
        n, L, R = self.n_qubits, self.n_camadas, len(self.rotacoes)
        rot = params[:R * L * n].reshape(R, L, n)
        i = R * L * n
        rzz = None
        if self.emaranhador == "rzz":
            npar = len(self.pares_emaranhamento())
            rzz = params[i:i + L * npar].reshape(L, npar)
            i += L * npar
        return {"rot": rot, "rzz": rzz, "alpha": params[i:] if self.camada_pos else None}

    def desempacotar(self, params):
        """(theta, phi, alpha) -- formato da v0.2-v0.5, só para genomas com 2 rotações sem RZZ."""
        if len(self.rotacoes) != 2 or self.emaranhador == "rzz":
            raise ValueError("desempacotar só vale para 2 rotações sem RZZ; use blocos()")
        b = self.blocos(params)
        return b["rot"][0], b["rot"][1], b["alpha"]

    def empacotar(self, theta, phi, alpha=None):
        if len(self.rotacoes) != 2 or self.emaranhador == "rzz":
            raise ValueError("empacotar só vale para 2 rotações sem RZZ")
        partes = [np.ravel(theta), np.ravel(phi)]
        if self.camada_pos:
            if alpha is None:
                raise ValueError("este ansatz tem camada_pos=True: alpha é obrigatório")
            partes.append(np.ravel(alpha))
        params = np.concatenate(partes).astype(float)
        self.blocos(params)                          # valida o tamanho
        return params

    def mascara_camada(self, l):
        """Parâmetros treináveis de uma camada (rotações e RZZ da camada l) + a camada pós."""
        n, L, R = self.n_qubits, self.n_camadas, len(self.rotacoes)
        m = np.zeros(self.n_params, dtype=bool)
        for r in range(R):
            m[r * L * n + l * n:r * L * n + (l + 1) * n] = True
        i = R * L * n
        if self.emaranhador == "rzz":
            npar = len(self.pares_emaranhamento())
            m[i + l * npar:i + (l + 1) * npar] = True
            i += L * npar
        if self.camada_pos:
            m[i:] = True
        return m

    def params_iniciais(self, seed=None, inicializacao="padrao", escala_alpha=np.pi / 2):
        """padrao: 1ª rotação ~ U(-0.1, 0.1), demais ~ U(-pi, pi), RZZ ~ U(-pi, pi),
        alpha ~ U(-escala_alpha, escala_alpha) -- mesma sequência aleatória da
        v0.2-v0.5 para o genoma padrão de 1 camada.
        pequena: todos ~ N(0, (0.1/sqrt(L))^2) (mitigação de barren plateau)."""
        if inicializacao not in INICIALIZACOES:
            raise ValueError(f"inicializacao deve ser uma de {INICIALIZACOES}")
        r = np.random.default_rng(seed)
        n, L = self.n_qubits, self.n_camadas
        npar = len(self.pares_emaranhamento()) if self.emaranhador == "rzz" else 0
        partes = []
        if inicializacao == "padrao":
            for k in range(len(self.rotacoes)):
                partes.append(r.uniform(-0.1, 0.1, L * n) if k == 0 else r.uniform(-np.pi, np.pi, L * n))
            if npar:
                partes.append(r.uniform(-np.pi, np.pi, L * npar))
            if self.camada_pos:
                partes.append(r.uniform(-escala_alpha, escala_alpha, n))
        else:
            s = 0.1 / np.sqrt(L)
            tam = len(self.rotacoes) * L * n + L * npar + (n if self.camada_pos else 0)
            partes.append(r.normal(0, s, tam))
        return np.concatenate(partes).astype(float)

    # ── dados ───────────────────────────────────────────────────────────────

    def checar_x(self, x):
        x = np.asarray(x, float)
        if x.shape != (self.n_features,):
            raise ValueError(f"amostra deve ter {self.n_features} features, recebeu {x.shape}")
        return x

    def _amplitudes(self, x):
        v = np.zeros(1 << self.n_qubits, dtype=complex)
        v[:len(x)] = x
        if self.constante_amplitude is not None:
            v[len(x)] = self.constante_amplitude
        nrm = np.linalg.norm(v)
        if nrm == 0:
            raise ValueError("codificação por amplitude não aceita vetor nulo")
        return v / nrm

    # ── simulação exata (vetor de estado, n <= MAX_QUBITS_VETORIAL) ────────

    def estado(self, x, params):
        x = self.checar_x(x)
        b = self.blocos(params)
        cfg, n = self.config, self.n_qubits
        diagonal = self.codificacao in ("iqp", "ising") or self.emaranhador in ("cz", "rzz")
        idx_bits = np.arange(cfg.dim) if diagonal else None       # 2^n inteiros: só se preciso
        H = gate_H()
        if self.codificacao == "amplitude":
            st = self._amplitudes(x)
        elif self.codificacao == "ising":
            st = np.zeros(cfg.dim, dtype=complex)
            st[0] = 1.0
            for q in range(n): apply_gate(st, H, q, n)
            for angs, rx in self._passos_adiabaticos(x):
                for i in range(n):
                    j = (i + 1) % n
                    st *= np.exp(-0.5j * angs[i] * _sinal(idx_bits, i) * _sinal(idx_bits, j))
                for q in range(n): apply_gate(st, gate_Rx(rx), q, n)
        else:
            st = np.zeros(cfg.dim, dtype=complex)
            st[0] = 1.0
            if self.codificacao == "angulo":
                T = gate_T()
                for q in range(n): apply_gate(st, H, q, n)
                for q in range(n): apply_gate(st, T, q, n)
            idx = self.indices_dados()
        pares = self.pares_emaranhamento()
        perms = cfg.camadas_cnot if self.emaranhador == "cnot" else None
        for l in range(self.n_camadas):
            if self.codificacao == "angulo":
                for q in range(n): apply_gate(st, gate_Ry(np.pi * x[idx[l, q]]), q, n)
            elif self.codificacao == "iqp":
                for q in range(n): apply_gate(st, H, q, n)
                for q in range(n): apply_gate(st, gate_Rz(2 * np.pi * x[idx[l, q]]), q, n)
                for i, j in self.pares_iqp():
                    ang = 2 * np.pi ** 2 * (1 - x[idx[l, i]]) * (1 - x[idx[l, j]])
                    st *= np.exp(-0.5j * ang * _sinal(idx_bits, i) * _sinal(idx_bits, j))
            for r, nome in enumerate(self.rotacoes):
                porta = _PORTAS[nome]
                for q in range(n): apply_gate(st, porta(b["rot"][r, l, q]), q, n)
            if self.emaranhador == "cnot":
                for camada in perms:
                    apply_perm(st, camada)
            elif self.emaranhador == "cz":
                for i, j in pares:
                    st *= np.where(((idx_bits >> i) & 1) & ((idx_bits >> j) & 1), -1.0, 1.0)
            else:
                for k, (i, j) in enumerate(pares):
                    st *= np.exp(-0.5j * b["rzz"][l, k] * _sinal(idx_bits, i) * _sinal(idx_bits, j))
        if b["alpha"] is not None:
            for q in range(n): apply_gate(st, gate_Ry(b["alpha"][q]), q, n)
        return st

    # ── leitura ─────────────────────────────────────────────────────────────

    def observaveis(self):
        """Lista de (Paulis, qubits) que a leitura mede, na ordem das features."""
        n = self.n_qubits
        if self.leitura in ("z", "z_zz"):
            obs = [("Z", (q,)) for q in range(n)]
            if self.leitura == "z_zz":
                obs += [("ZZ", p) for p in pares_zz(n)]
            return obs
        obs = [(P, (q,)) for q in range(n) for P in "XYZ"]
        if self.leitura == "pauli":
            return obs + [("ZZ", p) for p in pares_zz(n)]
        return obs + [(a + b, p) for p in pares_zz(n) for a in "XYZ" for b in "XYZ"]

    @property
    def n_saida(self):
        return len(self.observaveis())

    def converter(self, evs):
        """Valores esperados -> features. Nas leituras "z"/"z_zz" os n primeiros
        viram P(qubit = 1) = (1 - <Z>)/2, como na v0.2-v0.5."""
        evs = np.asarray(evs, float).copy()
        if self.leitura in ("z", "z_zz"):
            evs[:self.n_qubits] = (1 - evs[:self.n_qubits]) / 2
        return evs

    def medir(self, st):
        """Features exatas a partir do vetor de estado."""
        n = self.n_qubits
        if self.leitura in ("z", "z_zz"):
            pr = np.abs(st) ** 2
            idx = np.arange(len(pr))
            evs = [float(np.sum(pr * _sinal(idx, q))) for q in range(n)]
            if self.leitura == "z_zz":
                evs += [float(np.sum(pr * _sinal(idx, i) * _sinal(idx, j))) for i, j in pares_zz(n)]
            return self.converter(evs)
        return self.converter([esperanca_pauli(st, ops, qs) for ops, qs in self.observaveis()])

    # ── tradução para Qiskit (MPS, Aer, hardware IBM) ──────────────────────

    def circuito_qiskit(self, x, params, medir=True):
        from qiskit import QuantumCircuit
        x = self.checar_x(x)
        b = self.blocos(params)
        n = self.n_qubits
        qc = QuantumCircuit(n)
        if self.codificacao == "amplitude":
            from qiskit.circuit.library import StatePreparation
            qc.append(StatePreparation(self._amplitudes(x)), range(n))
        elif self.codificacao == "ising":
            for q in range(n):
                qc.h(q)
            for angs, rx in self._passos_adiabaticos(x):
                for i in range(n):
                    qc.rzz(float(angs[i]), i, (i + 1) % n)
                for q in range(n):
                    qc.rx(float(rx), q)
        else:
            if self.codificacao == "angulo":
                for q in range(n):
                    qc.h(q); qc.t(q)
            idx = self.indices_dados()
        pares = self.pares_emaranhamento()
        for l in range(self.n_camadas):
            if self.codificacao == "angulo":
                for q in range(n):
                    qc.ry(float(np.pi * x[idx[l, q]]), q)
            elif self.codificacao == "iqp":
                for q in range(n):
                    qc.h(q); qc.rz(float(2 * np.pi * x[idx[l, q]]), q)
                for i, j in self.pares_iqp():
                    qc.rzz(float(2 * np.pi ** 2 * (1 - x[idx[l, i]]) * (1 - x[idx[l, j]])), i, j)
            for r, nome in enumerate(self.rotacoes):
                for q in range(n):
                    getattr(qc, nome)(float(b["rot"][r, l, q]), q)
            if self.emaranhador == "cnot":
                for sub in self.config.pares_camadas:
                    for c, t in reversed(sub):          # mesma ordem do simulador vetorial
                        qc.cx(c, t)
            elif self.emaranhador == "cz":
                for i, j in pares:
                    qc.cz(i, j)
            else:
                for k, (i, j) in enumerate(pares):
                    qc.rzz(float(b["rzz"][l, k]), i, j)
        if b["alpha"] is not None:
            for q in range(n):
                qc.ry(float(b["alpha"][q]), q)
        if medir:
            qc.measure_all()
        return qc

    # ── serialização ────────────────────────────────────────────────────────

    def para_dict(self):
        return {"config": self.config.para_dict(), "n_features": self.n_features,
                "n_camadas": self.n_camadas, "codificacao": self.codificacao,
                "camada_pos": self.camada_pos, "constante_amplitude": self.constante_amplitude,
                "rotacoes": list(self.rotacoes), "emaranhador": self.emaranhador,
                "leitura": self.leitura,
                "cronograma_ising": [list(self.cronograma_ising[0]), list(self.cronograma_ising[1])]}

    @classmethod
    def de_dict(cls, d):
        d = dict(d)
        return cls(QuantumConfig.de_dict(d.pop("config")), **d)

    def __repr__(self):
        return (f"Ansatz(n_qubits={self.n_qubits}, n_features={self.n_features}, "
                f"n_camadas={self.n_camadas}, codificacao='{self.codificacao}', "
                f"rotacoes={self.rotacoes}, emaranhador='{self.emaranhador}', "
                f"leitura='{self.leitura}', camada_pos={self.camada_pos}, n_params={self.n_params})")


def qubits_para_amplitude(n_features, com_constante=True):
    """Menor n (>= 2) com 2^n >= n_features (+1 da amplitude constante)."""
    total = n_features + (1 if com_constante else 0)
    return max(2, int(np.ceil(np.log2(max(total, 1)))))
