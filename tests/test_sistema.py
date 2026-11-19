import numpy as np
import pytest

from quirios_engine import RFF, Brutos, ClassificadorQURIOS, QuantumConfig, Quantico


@pytest.fixture(scope="module")
def sistema_treinado():
    r = np.random.default_rng(11)
    Xs = r.uniform(0, 1, (60, 6)); ys = (Xs[:, 0] + Xs[:, 1] > 1).astype(float)
    clf = ClassificadorQURIOS([Brutos(), RFF(), Quantico(QuantumConfig(6, "anel", 1))]).fit(Xs, ys)
    return clf, Xs, ys


def test_relatorio_cobre_todos_os_geradores(sistema_treinado):
    clf, _, _ = sistema_treinado
    assert len(clf.relatorio_cv) == 3
    assert "escolhido" in clf.relatorio()


def test_modelo_escolhido_aprende_sinal(sistema_treinado):
    clf, Xs, ys = sistema_treinado
    assert (clf.predict(Xs) == ys).mean() > 0.85


def test_quantico_com_mais_features_que_qubits_usa_reuploading():
    g = Quantico(QuantumConfig(4, "anel", 1)).preparar(np.zeros((5, 12)))
    assert g.ansatz.n_camadas == 3
    F = g.transformar(np.random.default_rng(0).uniform(0, 1, (5, 12)))
    assert F.shape == (5, 8)


def test_quantico_reproduz_sorteio_da_v04():
    # mesmos theta/phi/alpha que a v0.4 sorteava em Quantico.preparar
    g = Quantico(QuantumConfig(6, "anel", 1), seed=7).preparar(np.zeros((2, 6)))
    r = np.random.default_rng(7)
    esperado = np.concatenate([r.uniform(-0.1, 0.1, 6), r.uniform(-np.pi, np.pi, 6),
                               r.uniform(-np.pi / 2, np.pi / 2, 6)])
    assert np.array_equal(g.params, esperado)
