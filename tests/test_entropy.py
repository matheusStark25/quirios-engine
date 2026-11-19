import numpy as np
import pytest

from quirios_engine import (QuantumConfig, apply_gate, apply_perm, gate_H,
                            entropia_emaranhamento, entropia_medicao,
                            mutual_information, pureza)


@pytest.fixture
def bell():
    c2 = QuantumConfig(n_qubits=2, topologia="linear", n_camadas_entrelacamento=1)
    st = np.zeros(4, complex); st[0] = 1
    apply_gate(st, gate_H(), 0, 2); apply_perm(st, c2.camadas_cnot[0])
    return st


def test_bell_entropia_emaranhamento_igual_a_1(bell):
    assert abs(entropia_emaranhamento(bell, 2, [0]) - 1) < 1e-6


def test_bell_mutual_information_igual_a_2(bell):
    assert abs(mutual_information(bell, 2, [0], [1]) - 2) < 1e-6


def test_estado_produto_nao_tem_emaranhamento():
    st = np.zeros(8, complex); st[0] = 1
    for q in range(3):
        apply_gate(st, gate_H(), q, 3)
    assert abs(entropia_emaranhamento(st, 3, [0])) < 1e-9
    assert abs(entropia_medicao(st) - 3) < 1e-9          # uniforme em 8 estados
    assert abs(pureza(st) - 1 / 8) < 1e-12


def test_convencao_lsb_mede_o_qubit_certo():
    # Regressão v0.6: Bell nos qubits 0 e 1, qubit 2 em |0>. A versão antiga
    # usava a convenção MSB e trocava o qubit 0 pelo qubit 2.
    st = np.zeros(8, complex); st[0] = st[3] = 1 / np.sqrt(2)     # (|000> + |011>)/sqrt2
    assert abs(entropia_emaranhamento(st, 3, [0]) - 1) < 1e-9
    assert abs(entropia_emaranhamento(st, 3, [1]) - 1) < 1e-9
    assert abs(entropia_emaranhamento(st, 3, [2])) < 1e-9
    assert abs(mutual_information(st, 3, [0], [1]) - 2) < 1e-9
    assert abs(mutual_information(st, 3, [0], [2])) < 1e-9


def test_entropia_escala_para_16_qubits():
    r = np.random.default_rng(0)
    st = r.normal(size=1 << 16) + 1j * r.normal(size=1 << 16)
    st /= np.linalg.norm(st)
    s = entropia_emaranhamento(st, 16, [0, 1, 2])          # a versão antiga pediria ~68 GB
    assert 2.5 < s <= 3.0                                  # estado aleatório: quase máximo


def test_mutual_information_exige_subsistemas_disjuntos(bell):
    with pytest.raises(ValueError):
        mutual_information(bell, 2, [0], [0, 1])
