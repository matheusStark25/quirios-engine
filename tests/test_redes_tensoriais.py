import math

import numpy as np
import pytest

from quirios_engine import redes_tensoriais as rt


def test_fatorar():
    assert rt.fatorar(784, 4) == [4, 4, 7, 7] and rt.fatorar(1024, 4) == [4, 4, 8, 8]
    assert rt.fatorar(10, 4) == [1, 1, 2, 5] and math.prod(rt.fatorar(4096, 3)) == 4096


def test_tt_matriz_ida_e_volta_e_parametros():
    r = np.random.default_rng(0)
    W = r.normal(size=(64, 48))
    ms, ns = rt.fatorar(64, 3), rt.fatorar(48, 3)
    assert np.abs(rt.tt_para_matriz(rt.tt_svd_matriz(W, ms, ns, 10 ** 6)) - W).max() < 1e-12
    G = rt.tt_svd_matriz(W, ms, ns, 3)
    assert sum(g.size for g in G) == rt.params_tt(ms, ns, 3)
    assert rt.params_tt(rt.fatorar(1024, 4), rt.fatorar(1024, 4), 4) == 1600          # contra 1.048.576 densos
    with pytest.raises(ValueError):
        rt.tt_svd_matriz(W, [4, 4], [6, 8], 2)


def test_mps_e_circuito_preparam_o_estado_exato():
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector
    r = np.random.default_rng(1)
    for n, chi in ((3, 2), (5, 4), (6, 8)):
        v = r.normal(size=2 ** n) + 1j * r.normal(size=2 ** n)
        v /= np.linalg.norm(v)
        assert rt.fidelidade(rt.mps_para_vetor(rt.vetor_para_mps(v, 2 ** n)), v) > 1 - 1e-12   # sem truncar
        mps = rt.vetor_para_mps(v, chi)
        sv = Statevector(rt.mps_para_circuito(mps)).data.reshape(-1, 2 ** n)
        assert rt.fidelidade(sv[0], rt.mps_para_vetor(mps)) > 1 - 1e-10                      # dados = MPS
        assert np.linalg.norm(sv[1:]) < 1e-8                                                  # auxiliares em |0>
    produto = np.kron([1, 0], [0.6, 0.8])
    assert rt.fidelidade(Statevector(rt.mps_para_circuito(rt.vetor_para_mps(produto, 1))).data, produto) > 1 - 1e-10


def test_ligacao_maior_aproxima_melhor():
    r = np.random.default_rng(2)
    v = np.cumsum(r.normal(size=256))                       # sinal suave, como uma imagem
    v /= np.linalg.norm(v)
    fids = [rt.fidelidade(rt.mps_para_vetor(rt.vetor_para_mps(v, chi)), v) for chi in (1, 2, 4, 16)]
    assert fids == sorted(fids) and fids[-1] > 1 - 1e-12
