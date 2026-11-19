import numpy as np
import pytest

from quirios_engine.dados_quanticos import (_transformada_hadamard, comparar, estado_fundamental, fases_ising,
                                            medir, texto_comparacao)


def test_fisica_das_duas_fases():
    r = np.random.default_rng(0)
    ferro = medir(estado_fundamental(np.ones(6), 0.2), 6, 4000, r)
    para = medir(estado_fundamental(np.ones(6), 3.0), 6, 4000, r)
    assert ferro[:6].mean() > 0.95 and ferro[-1] > 0.95          # <ZZ> e |M| altos
    assert para[6:12].mean() > 0.9                                # <X> alto
    assert para[:6].mean() < 0.4


def test_estado_fundamental_contra_diagonalizacao_densa():
    J, h = np.array([1.1, 0.9, 1.0, 1.2]), 0.8
    psi = estado_fundamental(J, h)
    from quirios_engine.dados_quanticos import hamiltoniano_ising
    H = hamiltoniano_ising(J, h)
    H = H.toarray() if hasattr(H, "toarray") else H
    e0 = np.linalg.eigvalsh(H)[0]
    assert abs(psi @ H @ psi - e0) < 1e-9
    assert np.all(psi >= -1e-12)                                  # Perron-Frobenius


def test_estado_fundamental_em_varias_threads():
    # Sem a trava, eigh simultâneo derrubava o processo (access violation no OpenBLAS do Windows).
    # Roda num subprocesso: se voltar a quebrar, o teste falha em vez de derrubar o pytest.
    import subprocess
    import sys
    codigo = ("import threading, numpy as np\n"
              "from quirios_engine.dados_quanticos import estado_fundamental\n"
              "def t(s):\n"
              "    r = np.random.default_rng(s)\n"
              "    for _ in range(60): estado_fundamental(1 + 0.3 * r.uniform(-1, 1, 8), 1.0)\n"
              "ts = [threading.Thread(target=t, args=(s,)) for s in range(6)]\n"
              "[x.start() for x in ts]; [x.join() for x in ts]\n")
    from pathlib import Path
    raiz = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "-c", codigo], cwd=raiz, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]


def test_transformada_de_hadamard():
    r = np.random.default_rng(1)
    v = r.normal(size=8); v /= np.linalg.norm(v)
    H1 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    Hn = np.kron(np.kron(H1, H1), H1)
    assert np.allclose(_transformada_hadamard(v, 3), Hn @ v)


def test_conjunto_de_dados():
    d = fases_ising(n_spins=5, n_amostras=30, shots=20, seed=2)
    assert d["X_estado"].shape == (30, 32) and d["X_medidas"].shape == (30, 11)
    assert set(np.unique(d["y"])) <= {0.0, 1.0}
    assert np.allclose(np.linalg.norm(d["X_estado"], axis=1), 1)
    with pytest.raises(ValueError):
        fases_ising(n_spins=20)


def test_comparacao_justa_e_o_padrao():
    d = fases_ising(n_spins=5, n_amostras=60, shots=10, seed=3)
    res = comparar(d, n_treino=24, repeticoes=2, incluir_svm=False)
    assert res["justo"] is True
    assert {"logistica", "rff", "vqc (20 medições)", "kernel-projetado (20 medições)"} == set(res["modelos"])
    assert all(0 <= v["media"] <= 1 for v in res["modelos"].values())
    assert "mesmo orçamento de medições" in texto_comparacao(res)
    teto = comparar(d, n_treino=24, repeticoes=2, incluir_svm=False, justo=False)
    assert "teto" in teto["comparacao"]["melhor_quantico"] and "TETO TEÓRICO" in texto_comparacao(teto)


def test_preparo_pelo_circuito():
    d = fases_ising(n_spins=6, n_amostras=12, shots=10, seed=4, preparo="circuito")
    assert d["X_hamiltoniano"].shape == (12, 7)
    assert d["fidelidade"].mean() > 0.85                   # cronograma otimizado de 6 passos
    exato = fases_ising(n_spins=6, n_amostras=12, shots=10, seed=4)
    assert np.allclose(exato["fidelidade"], 1.0)
    with pytest.raises(ValueError):
        fases_ising(preparo="magico")


def test_codificacao_ising_no_qiskit_e_no_mps():
    qi = pytest.importorskip("qiskit.quantum_info")
    from quirios_engine import Ansatz, BackendExato, QuantumConfig
    a = Ansatz(QuantumConfig(5, "anel", 1), codificacao="ising", emaranhador="rzz", leitura="pauli")
    assert a.n_features == 6 and a.portas_2q() == 6 * 5 + 5
    r = np.random.default_rng(5)
    x = np.concatenate([1 + 0.3 * r.uniform(-1, 1, 5), [0.9]])
    p = a.params_iniciais(5)
    sv = qi.Statevector(a.circuito_qiskit(x, p, medir=False)).data
    assert abs(np.vdot(a.estado(x, p), sv)) ** 2 > 1 - 1e-10
    with pytest.raises(ValueError, match="ising"):
        Ansatz(QuantumConfig(5, "anel", 1), codificacao="ising", n_features=5)
    pytest.importorskip("qiskit_aer")
    from quirios_engine import BackendMPS
    X = np.vstack([x, np.concatenate([np.ones(5), [1.2]])])
    assert np.abs(BackendMPS().features(X, a, p) - BackendExato().features(X, a, p)).max() < 1e-8


def test_cronograma_otimizado_reproduzivel():
    pytest.importorskip("scipy")
    from quirios_engine.dados_quanticos import otimizar_cronograma
    cron, fid = otimizar_cronograma(n_spins=4, passos=3, n_instancias=6, tentativas=1)
    assert len(cron[0]) == 3 and fid > 0.8


def test_backend_shots_estatistica():
    from quirios_engine import Ansatz, BackendExato, QuantumConfig
    from quirios_engine.backends import BackendShots
    a = Ansatz(QuantumConfig(4, "anel", 1), leitura="pauli")
    X = np.random.default_rng(6).uniform(0, 1, (3, 4)); p = a.params_iniciais(6)
    ex = BackendExato().features(X, a, p)
    poucos = np.abs(BackendShots(20, seed=1).features(X, a, p) - ex).mean()
    muitos = np.abs(BackendShots(20000, seed=1).features(X, a, p) - ex).mean()
    assert muitos < 0.02 < poucos
    with pytest.raises(ValueError):
        BackendShots(0)
