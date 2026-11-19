"""
Otimizadores dos parâmetros do circuito.

    ParameterShift  gradiente EXATO pela regra de parameter-shift.
                    Custo por passo: 2 execuções do circuito POR PARÂMETRO.
    SPSA            Simultaneous Perturbation Stochastic Approximation
                    (Spall, 1992). Estima o gradiente perturbando TODOS os
                    parâmetros de uma vez numa direção aleatória +-1.
                    Custo por passo: 2 execuções, com 10 ou 10^9 parâmetros.
                    Precisa de mais passos para convergir, mas o custo por
                    passo deixa de crescer com o tamanho do modelo -- e é
                    robusto ao ruído de shots e de hardware.

Exemplo de custo (circuitos por passo, lote de 16 amostras):
    modelo com   72 parâmetros: parameter-shift ~2.300  | SPSA 32
    modelo com 1000 parâmetros: parameter-shift ~32.000 | SPSA 32
"""
import numpy as np


class ParameterShift:
    nome = "parameter_shift"

    def __init__(self, lr=0.3):
        self.lr = lr

    def iniciar(self, perda, params, mascara, n_passos):
        pass

    def aplicar(self, params, grad, mascara):
        return params - self.lr * np.where(mascara, grad, 0.0)

    def circuitos_por_passo(self, n_params, lote):
        return (2 * n_params + 1) * lote             # +1: avaliação no ponto atual


class SPSA:
    """Ganhos padrão de Spall: a_k = a / (k + 1 + A)^alfa, c_k = c / (k + 1)^gama.

    a=None calibra o passo inicial para ~passo_alvo radianos medindo o
    gradiente estimado em `calibracao` perturbações antes de começar.
    A=None usa 10% do número de passos previsto."""
    nome = "spsa"

    def __init__(self, a=None, c=0.1, A=None, alfa=0.602, gama=0.101,
                 passo_alvo=0.2, calibracao=5, seed=None):
        self.a, self.c, self.A, self.alfa, self.gama = a, c, A, alfa, gama
        self.passo_alvo, self.calibracao, self.seed = passo_alvo, calibracao, seed
        self.k = 0

    def _delta(self, mascara):
        d = self._rng.choice([-1.0, 1.0], size=len(mascara))
        return np.where(mascara, d, 0.0)

    def iniciar(self, perda, params, mascara, n_passos):
        self._rng = np.random.default_rng(self.seed)
        self.k = 0
        self._A = 0.1 * n_passos if self.A is None else self.A
        self._a = self.a
        if self._a is None:
            mags = []
            for _ in range(max(1, self.calibracao)):
                d = self._delta(mascara)
                mags.append(abs(perda(params + self.c * d) - perda(params - self.c * d)) / (2 * self.c))
            mag = float(np.mean(mags))
            self._a = self.passo_alvo * (self._A + 1) ** self.alfa / mag if mag > 1e-12 else self.passo_alvo

    def ganhos(self):
        ak = self._a / (self.k + 1 + self._A) ** self.alfa
        ck = self.c / (self.k + 1) ** self.gama
        return ak, ck

    def passo(self, perda, params, mascara):
        """Um passo SPSA. Devolve (novos parâmetros, estimativa da perda)."""
        ak, ck = self.ganhos()
        d = self._delta(mascara)
        lp, lm = perda(params + ck * d), perda(params - ck * d)
        g = (lp - lm) / (2 * ck) * d                 # 1/delta_i == delta_i para delta = +-1
        self.k += 1
        return params - ak * g, 0.5 * (lp + lm)

    def circuitos_por_passo(self, n_params, lote):
        return 2 * lote


def criar_otimizador(otimizador, lr=0.3, seed=None):
    if isinstance(otimizador, (ParameterShift, SPSA)):
        return otimizador
    if otimizador == "parameter_shift":
        return ParameterShift(lr=lr)
    if otimizador == "spsa":
        return SPSA(seed=seed)
    raise ValueError("otimizador deve ser 'parameter_shift', 'spsa' ou uma instância de "
                     "ParameterShift/SPSA")
