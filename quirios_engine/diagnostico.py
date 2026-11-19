"""
Diagnósticos: o QURIOS mede os próprios limites antes de gastar QPU.

    variancia_gradiente(ansatz)   termômetro de barren plateau
    estimar_circuitos(...)        quantos circuitos um treino vai executar
"""
import numpy as np

from .otimizadores import criar_otimizador


def variancia_gradiente(ansatz, n_amostras=40, inicializacao="padrao", parametro=0,
                        qubit_medido=0, seed=0):
    """Var[ d<Z_q> / d params[parametro] ] sobre inicializações e dados aleatórios.

    Barren plateau = essa variância caindo exponencialmente com o número de
    qubits: o gradiente vira ruído e nenhum otimizador sai do lugar. Compare
    topologias, inicializações e números de camadas antes de treinar.
    Simulação exata (n <= 24)."""
    r = np.random.default_rng(seed)
    n = ansatz.n_qubits
    bit = ((np.arange(1 << n) >> qubit_medido) & 1).astype(float)
    sinal = 1 - 2 * bit                                   # autovalores de Z_q
    e = np.zeros(ansatz.n_params); e[parametro] = np.pi / 2
    gs = []
    for _ in range(n_amostras):
        params = ansatz.params_iniciais(int(r.integers(1 << 31)), inicializacao)
        if ansatz.codificacao == "angulo":
            x = r.uniform(0, 1, ansatz.n_features)
        else:
            x = r.normal(0, 1, ansatz.n_features)
        zp = np.sum(np.abs(ansatz.estado(x, params + e)) ** 2 * sinal)
        zm = np.sum(np.abs(ansatz.estado(x, params - e)) ** 2 * sinal)
        gs.append((zp - zm) / 2)
    return float(np.var(gs))


def estimar_circuitos(ansatz, n_amostras, epocas, otimizador="parameter_shift", lote=None):
    """Circuitos executados por um treino do VQCClassifier (equivalente em
    hardware): passos do otimizador + 1 avaliação do treino inteiro por época
    para resolver a leitura. Use para orçar tempo de QPU antes de enviar."""
    opt = criar_otimizador(otimizador)
    if lote is None:
        lote = min(n_amostras, 16) if opt.nome == "spsa" else n_amostras
    # o custo é linear no tamanho do lote: somar os lotes da época = usar n_amostras
    por_epoca = opt.circuitos_por_passo(ansatz.n_params, n_amostras) + n_amostras
    calibracao = 2 * opt.calibracao * lote if opt.nome == "spsa" and opt.a is None else 0
    return int(n_amostras + calibracao + epocas * por_epoca)
