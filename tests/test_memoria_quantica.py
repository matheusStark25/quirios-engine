import numpy as np
import pytest

from quirios_engine import memoria_quantica as mq


def test_paulis_comutacao_e_simetria():
    PX, PZ = mq.todos_paulis(2)
    assert len(PX) == 15
    nomes = {mq.nome_pauli(x, z, 2) for x, z in zip(PX, PZ)}
    assert "II" not in nomes and {"XI", "ZZ", "YX"} <= nomes
    x, z = 1, 0                                                   # X no qubit 0
    assert mq.comuta(x, z, 0, 1) == -1 and mq.comuta(x, z, 1, 0) == 1     # X anticomuta com Z
    assert mq.simetria_T(1, 1) == -1 and mq.simetria_T(3, 3) == 1        # Y: -1; YY: +1


def test_circuitos_reproduzem_a_teoria_exatamente():
    # Preparo: média sobre os bits sorteados = (I + eps P)/2^n.
    # Medida de Bell em rho x rho: p(Q) = 4^-n (1 + eps^2 s c(P,Q)), para todo P de 2 qubits.
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import DensityMatrix, Pauli, Statevector
    n, eps = 2, 0.9
    for x, z in zip(*mq.todos_paulis(n)):
        x, z = int(x), int(z)
        alvo = (np.eye(2 ** n) + eps * Pauli(mq.nome_pauli(x, z, n)[::-1]).to_matrix()) / 2 ** n
        k = [q for q in range(n) if ((x | z) >> q) & 1][0]
        rho = np.zeros((2 ** n, 2 ** n), complex)
        for bits in range(2 ** n):
            w = ((1 + eps) / 2 if not (bits >> k) & 1 else (1 - eps) / 2) / 2 ** (n - 1)
            qc = QuantumCircuit(n)
            mq._preparar(qc, list(range(n)), x, z, bits)
            v = Statevector(qc).data
            rho += w * np.outer(v, v.conj())
        assert np.abs(rho - alvo).max() < 1e-12
        med = QuantumCircuit(2 * n)
        for i in range(n):
            med.cx(i, n + i)
            med.h(i)
        s = int(mq.simetria_T(x, z))
        for chave, p in DensityMatrix(np.kron(rho, rho)).evolve(med).probabilities_dict().items():
            qx, qz = mq.pauli_de_bell(chave, n)
            assert abs(p - (1 + eps ** 2 * s * int(mq.comuta(x, z, qx, qz))) / 4 ** n) < 1e-12


def test_amostradores_respeitam_a_estrutura():
    rng = np.random.default_rng(0)
    x, z = 0b011, 0b110                                           # X Y Z em 3 qubits: um Y -> antissimétrico
    Q = mq.amostras_bell(3, x, z, 1.0, 50, rng)
    assert all(mq.comuta(x, z, a, b) * mq.simetria_T(x, z) == 1 for a, b in Q)
    xb, zb, bits = mq.amostras_uma_copia(3, x, z, 1.0, 400, rng)
    sup = x | z
    casa = (sup & ((x ^ xb) | (z ^ zb))) == 0
    assert casa.any() and np.all(mq._paridade(bits[casa] & sup) == 0)   # com eps = 1 a paridade é sempre +1


def test_memoria_quantica_vence_com_o_mesmo_numero_de_copias():
    r = mq.comparar(n_qubits=(4,), copias=(32,), instancias=80, seed=1)
    q = next(l for l in r["linhas"] if "com" in l["metodo"])["acuracia"]
    c = next(l for l in r["linhas"] if "sem" in l["metodo"])["acuracia"]
    assert q > 0.8 and q - c > 0.15
    assert "cópias para 90%" in mq.texto_comparacao(r)


def test_memoria_na_qpu_simulada():
    pytest.importorskip("qiskit_aer")
    from quirios_engine.backends import BackendQiskit
    from quirios_engine.qpu import memoria_na_qpu
    r = memoria_na_qpu(BackendQiskit("ideal", shots=1, seed=3), n_qubits=2, instancias=12, copias=16, seed=0)
    assert [j["tipo"] for j in r["jobs"]] == ["memoria-bell", "memoria-uma-copia"]
    assert [j["circuitos"] for j in r["jobs"]] == [12 * 8, 12 * 16]
    assert r["copias_por_estado"] == {"com_memoria": 16, "sem_memoria": 16}
    assert 0 <= r["acuracia"]["sem_memoria"] <= 1 and r["acuracia"]["com_memoria_quantica"] >= 0.75
    assert "calibracao" not in r and r["eps_decodificacao"] == 0.9                 # sem calibração: igual a antes


def test_calibracao_usa_estados_separados_e_o_teste_pareado():
    pytest.importorskip("qiskit_aer")
    from quirios_engine.backends import BackendQiskit
    from quirios_engine.qpu import GRADE_EPS, memoria_na_qpu
    b = BackendQiskit("parametrico", shots=1, seed=3, erros={"erro_2q": 0.08, "erro_1q": 0.01, "erro_leitura": 0.05})
    r = memoria_na_qpu(b, n_qubits=2, instancias=12, copias=16, seed=1, calibracao=10)
    assert [j["circuitos"] for j in r["jobs"]] == [22 * 8, 22 * 16]            # calibração + teste no mesmo job
    assert len(r["por_estado"]) == 12                                          # a nota sai só do teste
    c = r["calibracao"]
    assert c["estados"] == 10 and c["eps_com_memoria"] in GRADE_EPS and c["eps_sem_memoria"] in GRADE_EPS
    assert c["eps_com_memoria"] < 0.9                                          # o ruído apagou parte do sinal
    m = r["acuracia"]["mcnemar"]
    assert m["so_com_memoria_acertou"] + m["so_sem_memoria_acertou"] <= 12 and 0 < m["p_valor"] <= 1


def test_mcnemar_exato():
    from quirios_engine.qpu import mcnemar
    assert mcnemar([True] * 10, [False] * 10) == (10, 0, 2 / 2 ** 10)          # 10 discordâncias, todas para um lado
    assert mcnemar([True, False], [False, True])[2] == 1.0
    assert mcnemar([True] * 5, [True] * 5) == (0, 0, 1.0)                      # concordam sempre: nada a testar


def test_linha_de_comando(capsys):
    mq.main(["--n-qubits", "2", "--copias", "4", "8", "--instancias", "10"])
    assert "n=2" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        mq.main(["--n-qubits", "9"])
