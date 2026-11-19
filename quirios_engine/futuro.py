"""
Hardware do futuro: simula o QURIOS em processadores que ainda não existem,
para saber o que esperar de cada geração -- e quando um modelo quântico
passa a valer a pena.

    from quirios_engine.futuro import projetar_hardware
    tabela = projetar_hardware(X, y, ansatz, params)
    for linha in tabela: print(linha)

Gerações (taxas de erro por porta e de leitura; ruído despolarizante + erro
de leitura simétrico -- um modelo simples, não a física completa de um chip):

    "hoje"               ~ medianas públicas de um IBM Heron r2 (aproximadas)
    "proxima"            3x menos erro que hoje
    "dez_vezes"          10x menos erro
    "cem_vezes"          100x menos erro
    "tolerante_a_falhas" qubits lógicos com correção de erro (~1e-6), a meta
                         dos roteiros de 2029 (IBM Starling e equivalentes)

BackendQiskit(modo="parametrico", erros="proxima") ou erros={"erro_1q": ...,
"erro_2q": ..., "erro_leitura": ...} executa qualquer circuito nessas condições.
"""
import time

import numpy as np

GERACOES = {
    "hoje": {"erro_1q": 2.5e-4, "erro_2q": 3e-3, "erro_leitura": 1.2e-2},
    "proxima": {"erro_1q": 8e-5, "erro_2q": 1e-3, "erro_leitura": 4e-3},
    "dez_vezes": {"erro_1q": 2.5e-5, "erro_2q": 3e-4, "erro_leitura": 1.2e-3},
    "cem_vezes": {"erro_1q": 2.5e-6, "erro_2q": 3e-5, "erro_leitura": 1.2e-4},
    "tolerante_a_falhas": {"erro_1q": 1e-7, "erro_2q": 1e-6, "erro_leitura": 1e-6},
}

_PORTAS_1Q = ["rx", "ry", "rz", "h", "t", "tdg", "s", "sdg", "x", "sx", "u", "p"]
_PORTAS_2Q = ["cx", "cz", "rzz", "ecr", "swap"]


def resolver_erros(erros):
    """Nome de geração, dict de taxas ou None (= "hoje") -> dict completo."""
    if erros is None:
        erros = "hoje"
    if isinstance(erros, str):
        if erros not in GERACOES:
            raise ValueError(f"geração desconhecida '{erros}' (disponíveis: {list(GERACOES)})")
        return dict(GERACOES[erros])
    faltando = {"erro_1q", "erro_2q", "erro_leitura"} - set(erros)
    if faltando:
        raise ValueError(f"faltam taxas de erro: {sorted(faltando)}")
    if any(not 0 <= float(v) < 0.5 for v in erros.values()):
        raise ValueError("taxas de erro devem estar em [0, 0.5)")
    return {k: float(erros[k]) for k in ("erro_1q", "erro_2q", "erro_leitura")}


def modelo_de_ruido(erro_1q, erro_2q, erro_leitura):
    """NoiseModel do Aer: despolarizante em cada porta + erro de leitura."""
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
    nm = NoiseModel()
    if erro_1q > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(erro_1q, 1), _PORTAS_1Q)
    if erro_2q > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(erro_2q, 2), _PORTAS_2Q)
    if erro_leitura > 0:
        p = erro_leitura
        nm.add_all_qubit_readout_error(ReadoutError([[1 - p, p], [p, 1 - p]]))
    return nm


def _acuracia_cv(F, y, dobras, seed):
    """Logística L2 só sobre as features quânticas (sem os dados brutos, para
    o efeito do ruído não ficar diluído)."""
    from .sistema import _ajustar, _folds
    from .vqc import _sig
    k = max(2, min(dobras, int(np.bincount(y.astype(int)).min())))
    partes = _folds(y, k, seed)
    accs = []
    for i in range(k):
        te = partes[i]; tr = np.concatenate([partes[j] for j in range(k) if j != i])
        mu, sd, w, b = _ajustar(F[tr], y[tr], 1.0)
        accs.append(np.mean((_sig(((F[te] - mu) / sd) @ w + b) > 0.5) == y[te]))
    return float(np.mean(accs))


def projetar_hardware(X, y, ansatz, params, geracoes=None, shots=4000, mitigacao=(), seed=0,
                      dobras=5):
    """Para cada geração: erro das features contra a simulação exata e acurácia
    de validação cruzada usando SÓ as features quânticas -- a "curva de
    prontidão" do modelo. Devolve lista de dicts (a primeira linha é o ideal)."""
    from .backends import BackendExato, BackendQiskit
    X, y = np.asarray(X, float), np.asarray(y, float)
    geracoes = list(GERACOES) if geracoes is None else list(geracoes)
    exato = BackendExato().features(X, ansatz, params)
    linhas = []
    for g in [None] + geracoes:
        t0 = time.perf_counter()
        if g is None:
            F, nome = exato, "ideal"
        else:
            F = BackendQiskit("parametrico", erros=g, shots=shots, seed=seed,
                              mitigacao=mitigacao).features(X, ansatz, params)
            nome = g if isinstance(g, str) else "personalizado"
        linhas.append({"geracao": nome,
                       "erro_features": float(np.abs(F - exato).mean()),
                       "correlacao": float(np.corrcoef(F.ravel(), exato.ravel())[0, 1]),
                       "acuracia_cv": _acuracia_cv(F, y, dobras, seed),
                       "tempo_s": round(time.perf_counter() - t0, 2)})
    return linhas
