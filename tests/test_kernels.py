import numpy as np
import pytest

from quirios_engine import (ClassificadorKernel, KernelQuantico, QuantumConfig, alinhamento,
                            concentracao, treinar_alinhamento)


def _dados(n=40, d=4, seed=0):
    r = np.random.default_rng(seed)
    X = r.uniform(0, 1, (n, d))
    return X, (X[:, 0] + X[:, 1] > 1).astype(float)


@pytest.mark.parametrize("tipo", ["fidelidade", "projetado"])
def test_kernel_valido(tipo):
    X, _ = _dados(n=12)
    K = KernelQuantico(QuantumConfig(4, "anel", 1), tipo=tipo).matriz(X)
    assert K.shape == (12, 12)
    assert np.allclose(np.diag(K), 1.0) and np.allclose(K, K.T, atol=1e-12)
    assert np.all(K > -1e-12) and np.all(K <= 1 + 1e-12)
    assert np.linalg.eigvalsh(K).min() > -1e-8          # semidefinido positivo


def test_kernel_projetado_mps_igual_exato():
    pytest.importorskip("qiskit_aer")
    from quirios_engine import BackendMPS
    X, _ = _dados(n=6, d=5)
    cfg = QuantumConfig(5, "linear", 1)
    K1 = KernelQuantico(cfg, gamma=0.7).matriz(X)
    K2 = KernelQuantico(cfg, gamma=0.7, backend=BackendMPS()).matriz(X)
    assert np.abs(K1 - K2).max() < 1e-8


def test_classificador_kernel_aprende_e_generaliza():
    X, y = _dados(n=60, seed=1)
    Xt, yt = _dados(n=200, seed=2)
    # sem camadas de CNOT: neste problema linear o entrelaçamento PIORA o kernel
    # projetado (~0,93 -> ~0,8) -- o tipo de escolha que a auto-evolução faz sozinha
    clf = ClassificadorKernel(KernelQuantico(QuantumConfig(4, "anel", 0))).fit(X, y)
    assert np.mean(clf.predict(Xt) == yt) > 0.85
    assert clf.lam_ in ClassificadorKernel.GRADE_LAM
    assert abs(clf.acuracia_loo_ - np.mean(clf.predict(Xt) == yt)) < 0.1     # LOO estima bem
    p = clf.predict_proba(Xt)
    assert np.all((p >= 0) & (p <= 1))
    assert set(clf.diagnostico_) >= {"alinhamento", "variancia", "concentrado", "lam"}
    fixo = ClassificadorKernel(KernelQuantico(QuantumConfig(4, "anel", 0)), lam=0.5).fit(X, y)
    assert fixo.lam_ == 0.5


def test_alinhamento_e_concentracao():
    y = np.array([0, 1, 0, 1, 1, 0], float)
    t = 2 * y - 1
    assert abs(alinhamento(np.outer(t, t), y) - 1) < 1e-12
    assert concentracao(np.ones((5, 5)))["concentrado"] is True
    assert concentracao(np.eye(5) + 0.5 * np.random.default_rng(0).random((5, 5)))["concentrado"] is False


def test_fidelidade_concentra_mais_que_projetado():
    # Thanasilp et al. 2024: o kernel de fidelidade concentra com qubits/profundidade;
    # o projetado (Huang et al. 2021) compara só projeções locais e concentra menos.
    X, _ = _dados(n=20, d=10, seed=3)
    cfg = QuantumConfig(10, "anel", 2)
    vf = concentracao(KernelQuantico(cfg, tipo="fidelidade", n_camadas=3).matriz(X))["variancia"]
    vp = concentracao(KernelQuantico(cfg, tipo="projetado", n_camadas=3).matriz(X))["variancia"]
    assert vf < vp / 10


def test_treinar_alinhamento_nunca_piora():
    X, y = _dados(n=24, seed=4)
    k = KernelQuantico(QuantumConfig(4, "anel", 1))
    hist = treinar_alinhamento(k, X, y, passos=15, seed=0)
    final = alinhamento(k.matriz(X), y)
    assert max(hist) >= hist[0]
    assert final >= hist[0] - 1e-9


def test_kernel_entra_no_orquestrador():
    from quirios_engine.orquestrador import Orquestrador, candidatos_padrao
    X, y = _dados(n=30, seed=5)
    orq = Orquestrador(candidatos_padrao(QuantumConfig(4, "anel", 1), incluir_kernel=True),
                       repeticoes=1).fit(X, y)
    nomes = [c["nome"] for c in orq.relatorio["candidatos"]]
    assert "kernel-projetado" in nomes
