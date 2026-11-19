"""Fase 2: treino que escala (SPSA, parameter-shift geral, por camadas, diagnóstico)."""
import numpy as np
import pytest

from quirios_engine import (SPSA, Ansatz, BackendExato, ParameterShift, QuantumConfig,
                            VQCClassifier, estimar_circuitos, loss_e_grad, variancia_gradiente)


def _dados(n=48, d=4, seed=5):
    r = np.random.default_rng(seed)
    X = r.uniform(0, 1, (n, d))
    return X, (X[:, 0] + X[:, 1] > 1).astype(float)


# ── otimizadores ────────────────────────────────────────────────────────────

def test_spsa_converge_num_quadratico():
    alvo = np.linspace(-1, 1, 20)
    f = lambda p: float(np.sum((p - alvo) ** 2))
    opt, p, m = SPSA(seed=0), np.zeros(20), np.ones(20, bool)
    opt.iniciar(f, p, m, 500)
    for _ in range(500):
        p, _ = opt.passo(f, p, m)
    assert np.abs(p - alvo).max() < 0.01


def test_spsa_custa_2_avaliacoes_por_passo_qualquer_tamanho():
    for P in (10, 10_000):
        chamadas = []
        f = lambda p: chamadas.append(1) or float(np.sum(p ** 2))
        opt = SPSA(a=0.1, seed=0)
        opt.iniciar(f, np.ones(P), np.ones(P, bool), 10)
        chamadas.clear()
        opt.passo(f, np.ones(P), np.ones(P, bool))
        assert len(chamadas) == 2


def test_mascara_congela_parametros():
    p0 = np.ones(6); m = np.array([1, 1, 0, 0, 1, 0], bool)
    opt = SPSA(a=0.5, seed=1)
    f = lambda p: float(np.sum(p ** 2))
    opt.iniciar(f, p0, m, 10)
    p1, _ = opt.passo(f, p0, m)
    assert np.array_equal(p1[~m], p0[~m]) and not np.array_equal(p1[m], p0[m])
    p2 = ParameterShift(lr=0.1).aplicar(p0, np.full(6, 3.0), m)
    assert np.array_equal(p2[~m], p0[~m])


# ── VQC: caminho geral == caminho rápido ────────────────────────────────────

def test_grad_shift_geral_igual_ao_analitico():
    X, y = _dados(n=10)
    clf = VQCClassifier(QuantumConfig(4, "anel", 1), epocas=1, seed=0).fit(X, y)
    g, _ = clf._grad_shift(clf.params, X, y, np.ones(clf.ansatz.n_params, bool))
    acc = [np.zeros(4), np.zeros(4), np.zeros(4)]
    for xi, yi in zip(X, y):
        _, gs = loss_e_grad(xi, yi, clf.theta, clf.phi, clf.alpha, clf.w, clf.b, clf.config)
        acc = [a + gg for a, gg in zip(acc, gs)]
    assert np.allclose(g, np.concatenate(acc) / len(X), atol=1e-10)


def test_caminho_geral_reproduz_caminho_rapido():
    # treino_por_camadas com L=1 força o caminho geral sem mudar o algoritmo
    X, y = _dados()
    cfg = QuantumConfig(4, "anel", 1)
    rapido = VQCClassifier(cfg, epocas=4, seed=0).fit(X, y)
    geral = VQCClassifier(cfg, epocas=4, seed=0, treino_por_camadas=True).fit(X, y)
    assert np.allclose(rapido.params, geral.params, atol=1e-9)
    assert np.allclose(rapido.historico_loss, geral.historico_loss, atol=1e-9)


# ── VQC com SPSA ────────────────────────────────────────────────────────────

def test_vqc_spsa_reduz_a_perda():
    X, y = _dados(n=64)
    clf = VQCClassifier(QuantumConfig(4, "anel", 1), epocas=20, seed=0, otimizador="spsa").fit(X, y)
    assert clf.historico_loss[-1] < 0.7 * clf.historico_loss[0]
    assert (clf.predict(X) == y).mean() > 0.8


def test_custo_spsa_nao_cresce_com_o_numero_de_parametros():
    X, y = _dados(n=32)
    cfg = QuantumConfig(4, "anel", 1)
    c = [VQCClassifier(cfg, epocas=3, seed=0, otimizador="spsa", n_camadas=L).fit(X, y).circuitos_executados
         for L in (1, 4)]
    assert c[0] == c[1]


def test_custo_parameter_shift_cresce_com_o_numero_de_parametros():
    X, y = _dados(n=8)
    cfg = QuantumConfig(4, "anel", 1)
    c = [VQCClassifier(cfg, epocas=1, seed=0, n_camadas=L).fit(X, y).circuitos_executados for L in (1, 2)]
    assert c[1] > 1.5 * c[0]


@pytest.mark.parametrize("otimizador, L, n", [("spsa", 1, 32), ("spsa", 3, 32), ("spsa", 2, 37),
                                              ("parameter_shift", 1, 32), ("parameter_shift", 2, 32)])
def test_estimar_circuitos_bate_com_o_executado(otimizador, L, n):
    X, y = _dados(n=n)
    cfg = QuantumConfig(4, "anel", 1)
    clf = VQCClassifier(cfg, epocas=3, seed=0, otimizador=otimizador, n_camadas=L).fit(X, y)
    assert estimar_circuitos(clf.ansatz, n, 3, otimizador) == clf.circuitos_executados


def test_vqc_spsa_com_backend_mps():
    pytest.importorskip("qiskit_aer")
    from quirios_engine import BackendMPS
    X, y = _dados(n=16)
    clf = VQCClassifier(QuantumConfig(4, "linear", 1), epocas=3, seed=0, otimizador="spsa",
                        backend=BackendMPS()).fit(X, y)
    ref = VQCClassifier(QuantumConfig(4, "linear", 1), epocas=3, seed=0, otimizador="spsa",
                        backend=BackendExato(), treino_por_camadas=True).fit(X, y)
    assert np.allclose(clf.params, ref.params, atol=1e-6)


# ── treino por camadas e inicialização ──────────────────────────────────────

def test_estagios_do_treino_por_camadas():
    X, y = _dados(n=8)
    clf = VQCClassifier(QuantumConfig(4, "anel", 1), epocas=8, n_camadas=3, treino_por_camadas=True)
    clf.fit(X, y)
    est = clf._estagios()
    assert [e for e, _ in est] == [2, 2, 2, 2]
    assert est[0][1].sum() == 4 + 4 + 4        # theta[0], phi[0], alpha
    assert est[-1][1].all()
    assert len(clf.historico_loss) == 8


def test_treino_por_camadas_passa_por_todas_as_camadas_mesmo_com_poucas_epocas():
    X, y = _dados(n=8)
    clf = VQCClassifier(QuantumConfig(4, "anel", 1), epocas=1, n_camadas=2,
                        treino_por_camadas=True, seed=0).fit(X, y)
    est = clf._estagios()
    assert [e for e, _ in est] == [1, 1]
    m0, m1 = est[0][1], est[1][1]
    assert not np.any(m0[:4] & m1[:4])          # theta de camadas diferentes
    assert len(clf.historico_loss) == 2


def test_inicializacao_pequena():
    a = Ansatz(QuantumConfig(6, "anel", 1), n_camadas=4)
    p = a.params_iniciais(0, "pequena")
    assert np.abs(p).max() < 0.3
    with pytest.raises(ValueError):
        a.params_iniciais(0, "gigante")


def test_variancia_gradiente_detecta_plato():
    # com 4 camadas, a variância cai mais de 10x de 4 para 10 qubits
    v = [variancia_gradiente(Ansatz(QuantumConfig(n, "anel", 2), n_camadas=4), n_amostras=40)
         for n in (4, 10)]
    assert v[0] > 10 * v[1] > 0


# ── persistência ────────────────────────────────────────────────────────────

def test_salvar_e_carregar(tmp_path):
    X, y = _dados(n=16)
    clf = VQCClassifier(QuantumConfig(4, "anel", 1), epocas=2, seed=0, n_camadas=2).fit(X, y)
    clf.salvar(tmp_path / "modelo.json")
    outro = VQCClassifier.carregar(tmp_path / "modelo.json")
    assert np.allclose(outro.predict_proba(X), clf.predict_proba(X))
    assert outro.ansatz.n_camadas == 2


def test_carregar_modelo_ibm_exige_backend(tmp_path):
    X, y = _dados(n=8)
    clf = VQCClassifier(QuantumConfig(4, "anel", 1), epocas=1, seed=0).fit(X, y)
    clf.salvar(tmp_path / "m.json")
    txt = (tmp_path / "m.json").read_text(encoding="utf-8").replace('"backend": null', '"backend": "ibm"')
    (tmp_path / "m.json").write_text(txt, encoding="utf-8")
    with pytest.raises(ValueError, match="backend"):
        VQCClassifier.carregar(tmp_path / "m.json")
