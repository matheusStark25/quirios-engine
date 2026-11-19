"""Ansatz (fase 1) e mais dados por qubit (fase 3): re-uploading e amplitude."""
import numpy as np
import pytest

from quirios_engine import (Ansatz, BackendExato, QuantumConfig, VQCClassifier,
                            qubits_para_amplitude)
from quirios_engine.vqc import estado, marginais


def _fidelidade(a, b):
    return abs(np.vdot(a, b)) ** 2


# ── equivalência com o circuito da v0.2-v0.4 ────────────────────────────────

@pytest.mark.parametrize("n, topo, cam", [(4, "anel", 1), (6, "anel", 2), (8, "linear", 1)])
def test_ansatz_1_camada_igual_ao_circuito_antigo(n, topo, cam):
    cfg = QuantumConfig(n, topo, cam)
    r = np.random.default_rng(n)
    x, th, ph, al = r.uniform(0, 1, n), r.uniform(-3, 3, n), r.uniform(-3, 3, n), r.uniform(-3, 3, n)
    a = Ansatz(cfg)
    assert _fidelidade(a.estado(x, a.empacotar(th, ph, al)), estado(x, th, ph, al, cfg)) > 1 - 1e-12


def test_empacotar_desempacotar_e_tamanho():
    a = Ansatz(QuantumConfig(4, "anel", 1), n_camadas=3)
    assert a.n_params == 2 * 3 * 4 + 4
    p = np.arange(a.n_params, dtype=float)
    th, ph, al = a.desempacotar(p)
    assert th.shape == (3, 4) and ph.shape == (3, 4) and al.shape == (4,)
    assert np.array_equal(a.empacotar(th, ph, al), p)
    with pytest.raises(ValueError):
        a.desempacotar(np.zeros(5))


def test_serializacao_do_ansatz():
    a = Ansatz(QuantumConfig(5, "blocos", 1, tam_bloco=3, n_conexoes=2), n_features=15,
               codificacao="angulo", camada_pos=False)
    b = Ansatz.de_dict(a.para_dict())
    assert b.para_dict() == a.para_dict()
    assert b.config.pares_camadas == a.config.pares_camadas


# ── fase 3: data re-uploading ───────────────────────────────────────────────

def test_indices_de_dados():
    a = Ansatz(QuantumConfig(4, "anel", 1), n_features=12)
    assert a.n_camadas == 3
    assert a.indices_dados().tolist() == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    b = Ansatz(QuantumConfig(4, "anel", 1), n_camadas=3)          # d = n: reenvia os mesmos dados
    assert b.indices_dados().tolist() == [[0, 1, 2, 3]] * 3
    c = Ansatz(QuantumConfig(4, "anel", 1), n_features=12, n_camadas=2)
    assert c.features_usadas() == list(range(8))                  # 2 camadas x 4 qubits < 12


def test_reuploading_todas_as_features_afetam_a_saida():
    a = Ansatz(QuantumConfig(4, "anel", 1), n_features=12)
    r = np.random.default_rng(0)
    p, x = a.params_iniciais(0), r.uniform(0, 1, 12)
    base = marginais(a.estado(x, p), 4)
    for j in range(12):
        x2 = x.copy(); x2[j] = (x2[j] + 0.37) % 1
        assert np.abs(marginais(a.estado(x2, p), 4) - base).max() > 1e-6, f"feature {j} sem efeito"


def test_reuploading_qiskit_identico():
    qi = pytest.importorskip("qiskit.quantum_info")
    a = Ansatz(QuantumConfig(4, "anel", 2), n_features=12)
    r = np.random.default_rng(1)
    x, p = r.uniform(0, 1, 12), a.params_iniciais(1)
    sv = qi.Statevector(a.circuito_qiskit(x, p, medir=False)).data
    assert _fidelidade(a.estado(x, p), sv) > 1 - 1e-10


def test_reuploading_mps_igual_ao_exato():
    pytest.importorskip("qiskit_aer")
    from quirios_engine import BackendMPS
    a = Ansatz(QuantumConfig(6, "linear", 1), n_features=18)
    X = np.random.default_rng(2).uniform(0, 1, (3, 18)); p = a.params_iniciais(2)
    assert np.abs(BackendMPS().features(X, a, p) - BackendExato().features(X, a, p)).max() < 1e-8


def test_vqc_com_mais_features_que_qubits_aprende():
    r = np.random.default_rng(3)
    X = r.uniform(0, 1, (48, 12)); y = (X[:, 9] + X[:, 10] > 1).astype(float)   # sinal só no 3º bloco
    clf = VQCClassifier(QuantumConfig(4, "anel", 1), epocas=8, seed=0).fit(X, y)
    assert clf.ansatz.n_camadas == 3
    assert clf.historico_loss[-1] < clf.historico_loss[0]
    assert (clf.predict(X) == y).mean() > 0.75


# ── fase 3: codificação por amplitude ───────────────────────────────────────

def test_qubits_para_amplitude():
    # com a amplitude constante: d + 1 amplitudes
    assert [qubits_para_amplitude(d) for d in (2, 4, 5, 8, 30, 1024)] == [2, 3, 3, 4, 5, 11]
    assert [qubits_para_amplitude(d, False) for d in (2, 4, 5, 8, 30, 1024)] == [2, 2, 3, 3, 5, 10]


def test_amplitude_prepara_x_com_constante_normalizado():
    # sem camadas de entrelaçamento e com parâmetros zero, o circuito é a identidade
    a = Ansatz(QuantumConfig(3, "anel", 0), n_features=6, codificacao="amplitude", camada_pos=False)
    x = np.array([3.0, -1.0, 0.5, 2.0, 0.0, 1.0])
    st = a.estado(x, np.zeros(a.n_params))
    esperado = np.zeros(8); esperado[:6] = x; esperado[6] = 1.0
    assert np.allclose(st, esperado / np.linalg.norm(esperado))
    sem = Ansatz(QuantumConfig(3, "anel", 0), n_features=8, codificacao="amplitude", camada_pos=False,
                 constante_amplitude=None)
    x8 = np.arange(1.0, 9.0)
    assert np.allclose(sem.estado(x8, np.zeros(sem.n_params)), x8 / np.linalg.norm(x8))


def test_constante_distingue_x_de_menos_x():
    # Regressão: sem a constante, x e -x são o mesmo estado (fase global) -> mesmas features
    r = np.random.default_rng(10)
    x = r.normal(0, 1, 7)
    for constante, iguais in [(None, True), (1.0, False)]:
        a = Ansatz(QuantumConfig(3, "anel", 1), n_features=7, codificacao="amplitude", n_camadas=2,
                   constante_amplitude=constante)
        p = a.params_iniciais(0)
        F = BackendExato().features(np.vstack([x, -x]), a, p)
        assert np.allclose(F[0], F[1]) == iguais


def test_amplitude_rejeita_entradas_invalidas():
    with pytest.raises(ValueError, match="não cabem"):
        Ansatz(QuantumConfig(3, "anel", 1), n_features=8, codificacao="amplitude")   # 8 + 1 > 2^3
    a = Ansatz(QuantumConfig(3, "anel", 1), n_features=8, codificacao="amplitude", constante_amplitude=None)
    with pytest.raises(ValueError, match="nulo"):
        a.estado(np.zeros(8), a.params_iniciais(0))
    with pytest.raises(ValueError):
        Ansatz(QuantumConfig(3, "anel", 1), codificacao="fase")


def test_amplitude_qiskit_identico():
    qi = pytest.importorskip("qiskit.quantum_info")
    a = Ansatz(QuantumConfig(4, "anel", 1), n_features=13, codificacao="amplitude", n_camadas=2)
    r = np.random.default_rng(4)
    x, p = r.normal(0, 1, 13), a.params_iniciais(4)
    sv = qi.Statevector(a.circuito_qiskit(x, p, medir=False)).data
    assert _fidelidade(a.estado(x, p), sv) > 1 - 1e-10


def test_amplitude_mps_igual_ao_exato():
    pytest.importorskip("qiskit_aer")
    from quirios_engine import BackendMPS
    a = Ansatz(QuantumConfig(5, "linear", 1), n_features=30, codificacao="amplitude")
    X = np.random.default_rng(5).normal(0, 1, (3, 30)); p = a.params_iniciais(5)
    assert np.abs(BackendMPS().features(X, a, p) - BackendExato().features(X, a, p)).max() < 1e-8


def test_vqc_amplitude_30_features_em_5_qubits():
    r = np.random.default_rng(6)
    X = r.normal(0, 1, (48, 30)); y = (X[:, 0] - X[:, 7] > 0).astype(float)
    clf = VQCClassifier(QuantumConfig(5, "linear", 1), codificacao="amplitude", n_camadas=2,
                        epocas=8, seed=0).fit(X, y)
    assert clf.ansatz.n_qubits == 5 and clf.ansatz.n_features == 30
    assert clf.historico_loss[-1] < clf.historico_loss[0]
