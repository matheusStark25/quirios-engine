import numpy as np
import pytest

from quirios_engine import (QuantumConfig, VQCClassifier, cone_de_luz, entropia_medicao,
                            loss_e_grad, logistica_l2, pureza)
from quirios_engine.vqc import _marg, _marg_lento


def test_gradiente_parameter_shift_bate_com_diferenca_finita():
    cfg4 = QuantumConfig(4, "anel", 1)
    r = np.random.default_rng(1)
    x = r.uniform(0, 1, 4); th, ph, al = (r.uniform(-3, 3, 4) for _ in range(3))
    w, b = r.normal(0, 1, 4), 0.2
    _, gs = loss_e_grad(x, 1.0, th, ph, al, w, b, cfg4)

    def L(t, p, a):
        return loss_e_grad(x, 1.0, t, p, a, w, b, cfg4)[0]

    eps, dmax = 1e-6, 0.0
    for k in range(3):
        for i in range(4):
            ar_p = [th.copy(), ph.copy(), al.copy()]; ar_m = [th.copy(), ph.copy(), al.copy()]
            ar_p[k][i] += eps; ar_m[k][i] -= eps
            dmax = max(dmax, abs((L(*ar_p) - L(*ar_m)) / (2 * eps) - gs[k][i]))
    assert dmax < 1e-4


def test_regressao_logistica_interna_igual_ao_sklearn():
    sklm = pytest.importorskip("sklearn.linear_model")
    r = np.random.default_rng(2)
    Z = r.normal(0, 1, (60, 5)); yy = (Z[:, 0] + 0.5 * r.normal(0, 1, 60) > 0).astype(float)
    w1, b1 = logistica_l2(Z, yy, C=1.0)
    sk = sklm.LogisticRegression(C=1.0, max_iter=5000, tol=1e-10).fit(Z, yy)
    assert max(np.abs(w1 - sk.coef_[0]).max(), abs(b1 - sk.intercept_[0])) < 1e-4


@pytest.mark.parametrize("topo, kw", [("anel", {}), ("linear", {}), ("blocos", {"tam_bloco": 4})])
def test_todos_os_qubits_influenciam_a_saida(topo, kw):
    # Guarda contra o bug de cone de luz da v0.1 (leitura de 1 qubit)
    n = 8
    cfg = QuantumConfig(n, topo, 1, **kw)
    r = np.random.default_rng(3)
    _, gs = loss_e_grad(r.uniform(0, 1, n), 1.0, r.uniform(-1, 1, n), r.uniform(-3, 3, n),
                        r.uniform(-1.5, 1.5, n), r.normal(0, 1, n), 0.0, cfg)
    mortos = [q for q in range(n) if abs(gs[0][q]) < 1e-12]
    assert not mortos


def test_cone_de_luz_detecta_leitura_cega():
    assert cone_de_luz(QuantumConfig(8, "anel", 1)) == [0, 7]


def test_camada_pos_cnot_cria_interferencia():
    r = np.random.default_rng(4)
    x = r.uniform(0, 1, 6); th = r.uniform(-1, 1, 6); al = r.uniform(-1, 1, 6)
    c6 = QuantumConfig(6, "anel", 1)
    ph2 = r.uniform(-3, 3, 6)
    efeito_sem = np.abs(_marg(x, th, np.zeros(6), None, c6) - _marg(x, th, ph2, None, c6)).max()
    efeito_com = np.abs(_marg(x, th, np.zeros(6), al, c6) - _marg(x, th, ph2, al, c6)).max()
    assert efeito_sem < 1e-12      # sem camada pós: equivalente clássico, phi não age
    assert efeito_com > 1e-3       # com camada pós: phi altera as medidas


def test_simulacao_rapida_igual_porta_a_porta():
    dmax = 0.0
    for n, topo, kw in [(6, "anel", {}), (8, "anel", {}), (12, "blocos", {"tam_bloco": 6})]:
        cfg = QuantumConfig(n, topo, 2 if topo == "anel" else 1, **kw)
        r = np.random.default_rng(n)
        for al in [r.uniform(-3, 3, n), None]:
            a = [r.uniform(0, 1, n), r.uniform(-3, 3, n), r.uniform(-3, 3, n)]
            dmax = max(dmax, np.abs(_marg(*a, al, cfg) - _marg_lento(*a, al, cfg)).max())
    assert dmax < 1e-12


def test_aprende_sinal_real():
    r = np.random.default_rng(5)
    X = r.uniform(0, 1, (60, 4)); Y = (X[:, 0] > 0.5).astype(float)
    clf = VQCClassifier(QuantumConfig(4, "anel", 1), epocas=15, seed=0).fit(X, Y)
    assert (clf.predict(X) == Y).mean() > 0.85


def test_nao_decora_ruido():
    r = np.random.default_rng(5)
    r.uniform(0, 1, (60, 4))             # mesma sequência aleatória da suíte v0.3
    Xr = r.uniform(0, 1, (60, 4)); Yr = r.integers(0, 2, 60).astype(float)
    Xt = r.uniform(0, 1, (200, 4)); Yt = r.integers(0, 2, 200).astype(float)
    clf = VQCClassifier(QuantumConfig(4, "anel", 1), epocas=15, seed=0).fit(Xr, Yr)
    assert 0.40 <= (clf.predict(Xt) == Yt).mean() <= 0.60


@pytest.fixture(scope="module")
def clf_treinado():
    r = np.random.default_rng(6)
    X = r.uniform(0, 1, (30, 4)); Y = (X[:, 1] > 0.5).astype(float)
    return VQCClassifier(QuantumConfig(4, "anel", 1), epocas=3, seed=0).fit(X, Y), X


def test_extrair_features_formato_e_conteudo(clf_treinado):
    clf, X = clf_treinado
    F = clf.extrair_features(X[:5])
    assert F.shape == (5, 4 + 2)
    # as marginais são as mesmas usadas pela leitura do classificador
    assert np.allclose(F[:, :4], clf._M(X[:5]), atol=1e-12)
    st = clf.estado_quantico(X[0])
    assert np.isclose(F[0, 4], entropia_medicao(st))
    assert np.isclose(F[0, 5], pureza(st))


def test_estado_quantico_inclui_camada_pos(clf_treinado):
    clf, X = clf_treinado
    st = clf.estado_quantico(X[0])
    assert abs(np.sum(np.abs(st) ** 2) - 1) < 1e-9
    probs = np.abs(st) ** 2
    m1 = np.array([probs[((np.arange(16) >> q) & 1) == 1].sum() for q in range(4)])
    assert np.allclose(m1, clf._M(X[:1])[0], atol=1e-12)


def test_prever_sem_treinar_da_erro_claro():
    clf = VQCClassifier(QuantumConfig(4, "anel", 1))
    with pytest.raises(RuntimeError, match="não treinado"):
        clf.predict(np.zeros((1, 4)))
