"""
Kernels quânticos.

    KernelQuantico(config, tipo="projetado")   o circuito como similaridade entre amostras
    ClassificadorKernel(kernel)                kernel ridge + calibração (sem dependências)
    alinhamento(K, y)                          quão bem o kernel separa as classes
    concentracao(K)                            diagnóstico de concentração exponencial
    treinar_alinhamento(kernel, X, y)          ajusta o circuito para maximizar o alinhamento

Tipos:
    "fidelidade"  K(x, x') = |<psi(x)|psi(x')>|^2. O kernel quântico clássico;
                  com muitos qubits os valores CONCENTRAM exponencialmente em
                  torno de um número fixo e o kernel vira ruído (Thanasilp,
                  Wang, Cerezo & Holmes, Nat. Commun. 2024). Vetor de estado:
                  até 24 qubits.
    "projetado"   K(x, x') = exp(-gamma * sum_q ||rho_q(x) - rho_q(x')||_F^2),
                  com rho_q a matriz de densidade de 1 qubit (Huang et al.,
                  Nat. Commun. 2021). Compara projeções locais em vez do
                  estado inteiro, o que atenua a concentração; precisa só de
                  <X>, <Y>, <Z> de cada qubit, então roda em QUALQUER backend
                  (exato, MPS com 100 qubits, shots, hardware).

concentracao(K) devolve a variância dos valores fora da diagonal: perto de
zero, o kernel não distingue amostras e nenhum classificador aprende.
"""
import numpy as np

from .ansatz import Ansatz
from .backends import BackendExato
from .vqc import _sig, logistica_l2

TIPOS = ("projetado", "fidelidade")


class KernelQuantico:
    def __init__(self, config, tipo="projetado", gamma=None, backend=None, seed=0,
                 n_camadas=None, codificacao="angulo", **genes):
        if tipo not in TIPOS:
            raise ValueError(f"tipo deve ser um de {TIPOS}")
        self.config, self.tipo, self.gamma_fixo = config, tipo, gamma
        self.backend = backend or BackendExato()
        self.seed, self.n_camadas, self.codificacao = seed, n_camadas, codificacao
        genes = dict(genes)
        if tipo == "projetado":
            genes["leitura"] = "pauli"            # precisa de <X>, <Y>, <Z> por qubit
        self.genes = genes
        self.nome = f"kernel-{tipo}[{self.backend.nome}]"
        self.params = None

    def preparar(self, n_features):
        if self.params is None or self.ansatz.n_features != n_features:
            self.ansatz = Ansatz(self.config, n_features=n_features, n_camadas=self.n_camadas,
                                 codificacao=self.codificacao, **self.genes)
            self.params = self.ansatz.params_iniciais(self.seed)
        return self

    def representar(self, X, params=None):
        """Projetado: (amostras, 3n) com <X>,<Y>,<Z> de cada qubit.
        Fidelidade: vetores de estado (amostras, 2^n)."""
        X = np.atleast_2d(np.asarray(X, float))
        self.preparar(X.shape[1])
        p = self.params if params is None else params
        if self.tipo == "projetado":
            return self.backend.features(X, self.ansatz, p)[:, :3 * self.ansatz.n_qubits]
        return np.array([self.ansatz.estado(x, p) for x in X])

    def _distancias(self, R1, R2):
        # ||rho - rho'||_F^2 = 1/2 * sum_P (<P> - <P>')^2 para cada qubit
        return 0.5 * (np.sum(R1 ** 2, 1)[:, None] + np.sum(R2 ** 2, 1)[None, :] - 2 * R1 @ R2.T)

    def de_representacoes(self, R1, R2=None, gamma=None):
        R2 = R1 if R2 is None else R2
        if self.tipo == "fidelidade":
            return np.abs(R1.conj() @ R2.T) ** 2
        D = np.maximum(self._distancias(R1, R2), 0.0)
        return np.exp(-(gamma if gamma is not None else self.gamma_) * D)

    def ajustar_gamma(self, R):
        """gamma = 1 / mediana das distâncias (heurística da mediana)."""
        if self.gamma_fixo is not None:
            self.gamma_ = float(self.gamma_fixo)
        elif self.tipo == "projetado":
            D = self._distancias(R, R)[np.triu_indices(len(R), 1)]
            med = float(np.median(D)) if len(D) else 1.0
            self.gamma_ = 1.0 / med if med > 1e-12 else 1.0
        return self

    def matriz(self, X1, X2=None):
        R1 = self.representar(X1)
        if not hasattr(self, "gamma_"):
            self.ajustar_gamma(R1)
        return self.de_representacoes(R1, None if X2 is None else self.representar(X2))


def alinhamento(K, y):
    """Alinhamento centralizado kernel-alvo (Cortes et al.) em [-1, 1]."""
    t = 2 * np.asarray(y, float) - 1
    n = len(t)
    H = np.eye(n) - np.ones((n, n)) / n
    Kc, T = H @ K @ H, H @ np.outer(t, t) @ H
    den = np.linalg.norm(Kc) * np.linalg.norm(T)
    return float(np.sum(Kc * T) / den) if den > 0 else 0.0


def concentracao(K):
    """Média e variância dos valores fora da diagonal (variância ~0 = kernel concentrado)."""
    fora = K[~np.eye(len(K), dtype=bool)]
    v = float(np.var(fora)) if fora.size else 0.0
    return {"media": float(np.mean(fora)) if fora.size else 0.0, "variancia": v, "concentrado": v < 1e-4}


class ClassificadorKernel:
    """Kernel ridge (alvo +-1) que se calibra sozinho.

    lam="auto" (e gamma, se o kernel projetado não tiver gamma fixo) são
    escolhidos pelo erro quadrático leave-one-out (PRESS), que no kernel
    ridge tem fórmula fechada: f_-i(x_i) = t_i - alfa_i / [(K + lam I)^-1]_ii
    -- avaliar a grade inteira custa algumas inversões de matriz e nenhum
    circuito a mais. (O erro quadrático é contínuo; a acurácia LOO, com
    poucas amostras, varia em degraus e escolhe mal.)
    As probabilidades vêm de uma logística sobre as previsões leave-one-out."""

    GRADE_LAM = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
    GRADE_GAMMA = (0.3, 1.0, 3.0)                 # multiplicadores da heurística da mediana

    def __init__(self, kernel, lam="auto"):
        self.kernel, self.lam = kernel, lam
        self.quantico = True

    @property
    def circuitos_executados(self):
        return getattr(self, "_circuitos", 0)

    @staticmethod
    def _loo(K, t, lam):
        G = np.linalg.inv(K + lam * np.eye(len(K)))
        alfa = G @ t
        return alfa, t - alfa / np.diag(G)

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        R = self.kernel.representar(X)
        self.kernel.ajustar_gamma(R)
        t = 2 * y - 1
        lams = self.GRADE_LAM if self.lam == "auto" else (float(self.lam),)
        auto_gamma = self.kernel.tipo == "projetado" and self.kernel.gamma_fixo is None
        gammas = [self.kernel.gamma_ * m for m in self.GRADE_GAMMA] if auto_gamma else [None]
        melhor = None
        for g in gammas:
            K = self.kernel.de_representacoes(R, gamma=g)
            for lam in lams:
                alfa, loo = self._loo(K, t, lam)
                press = float(np.mean((t - loo) ** 2))
                if melhor is None or press < melhor[0]:
                    melhor = (press, g, lam, alfa, loo, K)
        _, g, lam, self.alfa_, loo, K = melhor
        if g is not None:
            self.kernel.gamma_ = g
        acc_loo = float(np.mean((loo > 0) == (t > 0)))
        self.lam_, self.acuracia_loo_ = lam, acc_loo
        self.platt_ = logistica_l2(loo[:, None], y, C=10.0)
        self.R_, self._circuitos = R, len(X)
        self.diagnostico_ = {"alinhamento": alinhamento(K, y), "lam": lam, "acuracia_loo": acc_loo,
                             **concentracao(K)}
        return self

    def decisao(self, X):
        Kx = self.kernel.de_representacoes(self.kernel.representar(X), self.R_)
        self._circuitos += len(Kx)
        return Kx @ self.alfa_

    def predict_proba(self, X):
        w, b = self.platt_
        return _sig(self.decisao(X) * w[0] + b)

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)


def treinar_alinhamento(kernel, X, y, passos=40, seed=0):
    """Ajusta os parâmetros do circuito com SPSA para MAXIMIZAR o alinhamento
    kernel-alvo (kernel quântico treinável). Devolve o histórico do alinhamento."""
    from .otimizadores import SPSA
    X, y = np.asarray(X, float), np.asarray(y, float)
    kernel.preparar(X.shape[1])
    R0 = kernel.representar(X)
    kernel.ajustar_gamma(R0)

    def perda(p):
        return -alinhamento(kernel.de_representacoes(kernel.representar(X, p)), y)

    opt = SPSA(seed=seed)
    tudo = np.ones(kernel.ansatz.n_params, bool)
    p = kernel.params.copy()
    opt.iniciar(perda, p, tudo, passos)
    melhor_v = -perda(p)
    melhor_p, hist = p.copy(), [melhor_v]
    for _ in range(passos):
        p, _ = opt.passo(perda, p, tudo)
        v = -perda(p)
        hist.append(v)
        if v > melhor_v:                     # guarda o melhor ponto, não o último
            melhor_v, melhor_p = v, p.copy()
    kernel.params = melhor_p
    kernel.ajustar_gamma(kernel.representar(X))
    return hist
