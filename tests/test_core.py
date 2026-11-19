import numpy as np
import pytest

from quirios_engine import QuantumConfig, apply_gate, apply_perm, gate_H
from quirios_engine.core import topologia_todos_pares
from quirios_engine.vqc import estado


def _bijetora(cfg):
    return all(len(set(p.tolist())) == cfg.dim for p in cfg.camadas_cnot)


def test_par_de_bell_gera_estado_esperado():
    c2 = QuantumConfig(n_qubits=2, topologia="linear", n_camadas_entrelacamento=1)
    st = np.zeros(4, complex); st[0] = 1
    apply_gate(st, gate_H(), 0, 2); apply_perm(st, c2.camadas_cnot[0])
    esperado = np.array([1, 0, 0, 1]) / np.sqrt(2)
    assert np.allclose(st, esperado)


@pytest.mark.parametrize("n, topo", [(4, "anel"), (6, "linear"), (10, "anel"),
                                     (12, "blocos"), (14, "blocos")])
def test_permutacoes_bijetoras(n, topo):
    cfg = QuantumConfig(n_qubits=n, topologia=topo, n_camadas_entrelacamento=1, tam_bloco=6)
    assert _bijetora(cfg)


@pytest.mark.parametrize("n, camadas", [(2, 2), (2, 5), (3, 3), (4, 4), (4, 9)])
def test_anel_com_muitas_camadas_continua_unitario(n, camadas):
    # Regressão v0.3: n_camadas multiplo de n_qubits gerava pares (i, i)
    cfg = QuantumConfig(n, "anel", camadas)
    assert all(c != t for pares in cfg.pares_camadas for c, t in pares)
    assert _bijetora(cfg)


def test_anel_preserva_offsets_da_v03_quando_camadas_menor_que_n():
    cfg = QuantumConfig(8, "anel", 3)
    for c, pares in enumerate(cfg.pares_camadas):
        assert pares == [(i, (i + c + 1) % 8) for i in range(8)]


def test_norma_do_estado_igual_a_1():
    cfg8 = QuantumConfig(8, "anel", 2)
    r = np.random.default_rng(0)
    s = estado(r.uniform(0, 1, 8), r.uniform(-1, 1, 8), r.uniform(-1, 1, 8),
               r.uniform(-1, 1, 8), cfg8)
    assert abs(np.sum(np.abs(s) ** 2) - 1) < 1e-9


@pytest.mark.parametrize("kwargs", [{"n_qubits": 1}, {"topologia": "estrela"}])
def test_config_invalida_levanta_erro(kwargs):
    with pytest.raises(ValueError):
        QuantumConfig(**kwargs)


def test_todos_pares_bloqueado_acima_do_limite():
    with pytest.raises(ValueError):
        topologia_todos_pares(9)
