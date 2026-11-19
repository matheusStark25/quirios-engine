import numpy as np
import pytest

pytest.importorskip("qiskit_aer")

from quirios_engine import Ansatz, BackendExato, QuantumConfig, criar_backend  # noqa: E402
from quirios_engine.ibm_estimador import BackendEstimador  # noqa: E402


def _caso(n=5, leitura="z_zz", seed=0):
    a = Ansatz(QuantumConfig(n, "linear", 2), leitura=leitura)
    X = np.random.default_rng(seed).uniform(0, 1, (3, n))
    return a, X, a.params_iniciais(seed)


@pytest.mark.parametrize("leitura", ["z_zz", "pauli"])
def test_estimador_local_exato(leitura):
    a, X, p = _caso(leitura=leitura)
    F = BackendEstimador("local").features(X, a, p)
    assert np.abs(F - BackendExato().features(X, a, p)).max() < 1e-8


def test_obp_deixa_o_circuito_mais_raso_dentro_do_orcamento():
    pytest.importorskip("qiskit_addon_obp")
    a, X, p = _caso(n=6)
    b = BackendEstimador("local", obp=True, obp_erro_max=0.02)
    F = b.features(X, a, p)
    red = b.ultima_reducao
    assert red["profundidade_executada"] < red["profundidade_original"]
    assert np.abs(F - BackendExato().features(X, a, p)).max() < 0.05
    assert "+obp" in b.nome


def test_estimador_local_com_ruido_do_hardware_de_hoje():
    a, X, p = _caso()
    exato = BackendExato().features(X, a, p)
    F = BackendEstimador("local", erros="hoje").features(X, a, p)
    assert 0 < np.abs(F - exato).mean() < 0.1


def test_opcoes_da_pilha_de_mitigacao_ibm():
    b = BackendEstimador("ibm", resiliencia=2, desacoplamento=True, twirling=True, shots=8000,
                         opcoes_extra={"twirling": {"num_randomizations": 32}})
    o = b.opcoes_ibm()
    assert o["resilience_level"] == 2 and o["default_shots"] == 8000
    assert o["dynamical_decoupling"] == {"enable": True, "sequence_type": "XpXm"}
    assert o["twirling"]["enable_gates"] is True and o["twirling"]["num_randomizations"] == 32
    with pytest.raises(ValueError):
        BackendEstimador("ibm", resiliencia=3)


def test_tarefa_assincrona_e_fabrica():
    a, X, p = _caso()
    b = criar_backend("estimador-local")
    t = b.submeter(X, a, p)
    assert t.resultado(timeout=120).shape == (3, 10) and t.status() == "concluida"
    assert criar_backend("ibm-estimador").modo == "ibm"
