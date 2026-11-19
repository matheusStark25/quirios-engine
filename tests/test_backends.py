import numpy as np
import pytest

from quirios_engine import (Ansatz, Backend, BackendExato, QuantumConfig, criar_backend,
                            dispositivos_disponiveis)
from quirios_engine.vqc import estado


def _params(n, seed):
    r = np.random.default_rng(seed)
    return [r.uniform(-1, 1, n) for _ in range(3)]


def test_backend_exato_formato_das_features():
    cfg = QuantumConfig(6, "anel", 1)
    ans = Ansatz(cfg)
    X = np.random.default_rng(0).uniform(0, 1, (3, 6))
    F = BackendExato().features(X, ans, ans.empacotar(*_params(6, 1)))
    assert F.shape == (3, 6 + 6)           # n marginais + n pares ZZ do anel
    assert np.all((F[:, :6] >= 0) & (F[:, :6] <= 1))
    assert np.all(np.abs(F[:, 6:]) <= 1 + 1e-12)


def test_backend_exato_rejeita_acima_do_limite_vetorial():
    ans = Ansatz(QuantumConfig(30, "linear", 1))
    with pytest.raises(ValueError, match="BackendMPS"):
        BackendExato().features(np.zeros((1, 30)), ans, ans.params_iniciais(0))


def test_backend_valida_colunas_de_X():
    ans = Ansatz(QuantumConfig(4, "anel", 1))
    with pytest.raises(ValueError, match="colunas"):
        BackendExato().features(np.zeros((2, 3)), ans, ans.params_iniciais(0))


def test_submeter_devolve_tarefa_com_mesmo_resultado():
    cfg = QuantumConfig(4, "anel", 1)
    ans = Ansatz(cfg)
    X = np.random.default_rng(1).uniform(0, 1, (5, 4))
    p = ans.params_iniciais(2)
    t = BackendExato().submeter(X, ans, p)
    assert t.id and t.origem == "exato"
    F = t.resultado(timeout=60)
    assert t.status() == "concluida"
    assert np.allclose(F, BackendExato().features(X, ans, p))


def test_contrato_permite_backend_novo():
    class Constante(Backend):
        nome = "constante"
        def _calcular(self, X, ansatz, params):
            return np.full((len(X), 2 * ansatz.n_qubits), 0.5)

    ans = Ansatz(QuantumConfig(3, "anel", 1))
    b = Constante()
    assert b.features(np.zeros((2, 3)), ans, ans.params_iniciais(0)).shape == (2, 6)
    assert b.submeter(np.zeros((2, 3)), ans, ans.params_iniciais(0)).resultado(timeout=30).shape == (2, 6)
    assert b.capacidades()["assincrono_nativo"] is False


@pytest.mark.parametrize("n, topo, cam, kw", [(4, "anel", 1, {}), (6, "anel", 2, {}),
                                              (8, "linear", 1, {}), (10, "blocos", 1, {"tam_bloco": 5})])
def test_circuito_qiskit_identico_ao_simulador(n, topo, cam, kw):
    qi = pytest.importorskip("qiskit.quantum_info")
    from quirios_engine.backends import circuito_qiskit
    cfg = QuantumConfig(n, topo, cam, **kw)
    r = np.random.default_rng(n)
    a_ = [r.uniform(0, 1, n), r.uniform(-3, 3, n), r.uniform(-3, 3, n), r.uniform(-3, 3, n)]
    sv = qi.Statevector(circuito_qiskit(*a_, cfg, medir=False)).data
    assert abs(np.vdot(estado(*a_, cfg), sv)) ** 2 > 1 - 1e-10


def test_features_por_shots_convergem_para_exato():
    pytest.importorskip("qiskit_aer")
    from quirios_engine import BackendQiskit
    cfg = QuantumConfig(6, "anel", 1)
    ans = Ansatz(cfg)
    r = np.random.default_rng(9)
    Xq = r.uniform(0, 1, (5, 6)); p = ans.empacotar(*[r.uniform(-1, 1, 6) for _ in range(3)])
    err = np.abs(BackendQiskit("ideal", shots=20000, seed=1).features(Xq, ans, p)
                 - BackendExato().features(Xq, ans, p)).max()
    assert err < 0.03


def test_backend_qiskit_tarefa_assincrona():
    pytest.importorskip("qiskit_aer")
    from quirios_engine import BackendQiskit
    ans = Ansatz(QuantumConfig(4, "anel", 1))
    b = BackendQiskit("ideal", shots=500, seed=3)
    t = b.submeter(np.full((3, 4), 0.3), ans, ans.params_iniciais(0))
    assert b.ultimo_job_id == t.id
    assert t.resultado(timeout=120).shape == (3, 8)
    assert t.status() == "concluida"
    assert b.capacidades()["assincrono_nativo"] is True


def test_modo_ruidoso_aplica_ruido_de_verdade():
    # Regressão v0.3: o SamplerV2 do Aer era criado sem o modelo de ruído
    pytest.importorskip("qiskit_ibm_runtime")
    from quirios_engine import BackendQiskit
    cfg = QuantumConfig(6, "anel", 2)
    ans = Ansatz(cfg)
    r = np.random.default_rng(4)
    X = r.uniform(0, 1, (4, 6)); p = ans.params_iniciais(1)
    exato = BackendExato().features(X, ans, p)
    erro_ideal = np.abs(BackendQiskit("ideal", shots=20000, seed=1).features(X, ans, p) - exato).mean()
    erro_ruido = np.abs(BackendQiskit("ruidoso", shots=20000, seed=1).features(X, ans, p) - exato).mean()
    assert erro_ruido > 2 * erro_ideal


def test_backend_qiskit_rejeita_modo_invalido():
    from quirios_engine import BackendQiskit
    with pytest.raises(ValueError):
        BackendQiskit("nuvem")


def test_gpu_indisponivel_da_erro_claro():
    if "GPU" in dispositivos_disponiveis():
        pytest.skip("máquina com GPU")
    pytest.importorskip("qiskit_aer")
    from quirios_engine import BackendMPS
    with pytest.raises(ValueError, match="GPU"):
        BackendMPS(device="GPU")


def test_criar_backend_por_nome():
    assert criar_backend("exato").nome == "exato"
    with pytest.raises(ValueError):
        criar_backend("quantico-magico")


# ── MPS ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n, topo, cam, kw", [(6, "anel", 2, {}), (8, "linear", 1, {}),
                                              (10, "blocos", 1, {"tam_bloco": 5})])
def test_mps_igual_ao_exato(n, topo, cam, kw):
    pytest.importorskip("qiskit_aer")
    from quirios_engine import BackendMPS
    ans = Ansatz(QuantumConfig(n, topo, cam, **kw))
    r = np.random.default_rng(n)
    X = r.uniform(0, 1, (3, n)); p = ans.params_iniciais(n)
    assert np.abs(BackendMPS().features(X, ans, p) - BackendExato().features(X, ans, p)).max() < 1e-8


def test_mps_roda_100_qubits():
    pytest.importorskip("qiskit_aer")
    from quirios_engine import BackendMPS
    ans = Ansatz(QuantumConfig(100, "linear", 1))
    X = np.random.default_rng(0).uniform(0, 1, (2, 100))
    F = BackendMPS().features(X, ans, ans.params_iniciais(0))
    assert F.shape == (2, 200)
    assert np.all((F[:, :100] >= -1e-9) & (F[:, :100] <= 1 + 1e-9))


def test_mps_truncado_e_aproximado():
    pytest.importorskip("qiskit_aer")
    from quirios_engine import BackendMPS
    b = BackendMPS(max_bond=4)
    assert b.exato is False and "chi=4" in b.nome
    ans = Ansatz(QuantumConfig(8, "anel", 2))
    X = np.random.default_rng(5).uniform(0, 1, (2, 8)); p = ans.params_iniciais(5)
    err = np.abs(b.features(X, ans, p) - BackendExato().features(X, ans, p)).max()
    assert err < 0.5      # aproximação controlada, não exata
