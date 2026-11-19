"""Fase 5: genoma do circuito (portas, emaranhadores, IQP, leituras), medição
por bases e sombras clássicas, mitigação de erro, hardware futuro e plugins."""
import numpy as np
import pytest

from quirios_engine import (Ansatz, Backend, BackendExato, QuantumConfig, VQCClassifier,
                            backends_registrados, criar_backend, dobrar_portas)

GENOMAS = [
    dict(codificacao="angulo", rotacoes=("rz", "ry", "rz"), emaranhador="cz"),
    dict(codificacao="iqp", rotacoes=("ry",), emaranhador="rzz"),
    dict(codificacao="angulo", rotacoes=("rx", "ry"), emaranhador="rzz", camada_pos=False),
    dict(codificacao="amplitude", rotacoes=("ry", "rz"), emaranhador="cz"),
]


def _ans(g, n=4, cam=2, L=2, **kw):
    d = 6 if g["codificacao"] == "amplitude" else n
    return Ansatz(QuantumConfig(n, "anel", cam), n_features=d, n_camadas=L, **g, **kw)


def _x(a, seed):
    r = np.random.default_rng(seed)
    return r.normal(0, 1, a.n_features) if a.codificacao == "amplitude" else r.uniform(0, 1, a.n_features)


# ── circuito ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("g", GENOMAS)
def test_genoma_qiskit_identico_ao_simulador(g):
    qi = pytest.importorskip("qiskit.quantum_info")
    a = _ans(g)
    x, p = _x(a, 1), a.params_iniciais(1)
    sv = qi.Statevector(a.circuito_qiskit(x, p, medir=False)).data
    assert abs(np.vdot(a.estado(x, p), sv)) ** 2 > 1 - 1e-10
    assert abs(np.sum(np.abs(a.estado(x, p)) ** 2) - 1) < 1e-10


def test_genoma_padrao_e_compativel():
    a = Ansatz(QuantumConfig(4, "anel", 1))
    assert a.padrao and a.leitura == "z_zz" and a.n_params == 3 * 4
    assert not Ansatz(QuantumConfig(4, "anel", 1), emaranhador="cz").padrao
    with pytest.raises(ValueError, match="blocos"):
        Ansatz(QuantumConfig(4, "anel", 1), rotacoes=("ry",)).desempacotar(np.zeros(8))


def test_rzz_tem_um_parametro_por_par_e_camada():
    a = Ansatz(QuantumConfig(4, "anel", 2), n_camadas=3, emaranhador="rzz")
    pares = len(a.pares_emaranhamento())
    assert pares == 8
    assert a.n_params == 2 * 3 * 4 + 3 * pares + 4
    assert a.blocos(np.zeros(a.n_params))["rzz"].shape == (3, pares)
    m = a.mascara_camada(1)
    assert m.sum() == 2 * 4 + pares + 4
    assert a.portas_2q() == 3 * pares


def test_iqp_custa_portas_de_dados():
    a = Ansatz(QuantumConfig(4, "anel", 1), codificacao="iqp")
    assert a.portas_2q() == len(a.pares_emaranhamento()) + len(a.pares_iqp())


def test_validacao_dos_genes():
    cfg = QuantumConfig(3, "anel", 1)
    for kw in [dict(emaranhador="swap"), dict(leitura="w"), dict(rotacoes=()), dict(rotacoes=("rw",))]:
        with pytest.raises(ValueError):
            Ansatz(cfg, **kw)


# ── leitura ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("leitura, tamanho", [("z", 4), ("z_zz", 8), ("pauli", 16), ("pauli2", 48)])
def test_leituras_exatas_batem_com_qiskit(leitura, tamanho):
    qi = pytest.importorskip("qiskit.quantum_info")
    a = Ansatz(QuantumConfig(4, "anel", 1), leitura=leitura, emaranhador="rzz", codificacao="iqp")
    x, p = _x(a, 2), a.params_iniciais(2)
    F = a.medir(a.estado(x, p))
    assert F.shape == (tamanho,) == (a.n_saida,)
    sv = qi.Statevector(a.circuito_qiskit(x, p, medir=False))
    evs = [sv.expectation_value(qi.SparsePauliOp.from_sparse_list([(o, list(q), 1)], 4)).real
           for o, q in a.observaveis()]
    assert np.allclose(F, a.converter(evs), atol=1e-10)


@pytest.mark.parametrize("g", GENOMAS[:3])
def test_mps_igual_exato_com_leitura_pauli2(g):
    pytest.importorskip("qiskit_aer")
    from quirios_engine import BackendMPS
    a = _ans(g, n=5, cam=1, leitura="pauli2")
    X = np.array([_x(a, s) for s in range(3)]); p = a.params_iniciais(3)
    assert np.abs(BackendMPS().features(X, a, p) - BackendExato().features(X, a, p)).max() < 1e-8


# ── treino com genoma ───────────────────────────────────────────────────────

def test_parameter_shift_exato_com_rzz_rx_iqp_e_pauli():
    r = np.random.default_rng(4)
    X = r.uniform(0, 1, (6, 3)); y = np.array([0, 1, 0, 1, 1, 0], float)
    clf = VQCClassifier(QuantumConfig(3, "anel", 1), epocas=1, seed=0, codificacao="iqp",
                        rotacoes=("rx", "ry"), emaranhador="rzz", leitura="pauli").fit(X, y)
    p = clf.params
    g, _ = clf._grad_shift(p, X, y, np.ones(len(p), bool))
    eps, dmax = 1e-6, 0.0
    for i in range(len(p)):
        e = np.zeros_like(p); e[i] = eps
        fd = (clf._perda(p + e, X, y) - clf._perda(p - e, X, y)) / (2 * eps)
        dmax = max(dmax, abs(fd - g[i]))
    assert dmax < 1e-5


def test_vqc_treina_com_genoma_emergente():
    r = np.random.default_rng(5)
    X = r.uniform(0, 1, (40, 4)); y = (X[:, 0] + X[:, 1] > 1).astype(float)
    clf = VQCClassifier(QuantumConfig(4, "anel", 1), epocas=12, seed=0, otimizador="spsa",
                        codificacao="iqp", emaranhador="rzz", leitura="pauli").fit(X, y)
    assert clf.historico_loss[-1] < clf.historico_loss[0]
    assert clf.ansatz.n_saida == 16


# ── medição com shots: bases e sombras clássicas ────────────────────────────

def test_leitura_pauli_por_bases_converge():
    pytest.importorskip("qiskit_aer")
    from quirios_engine import BackendQiskit
    a = Ansatz(QuantumConfig(3, "anel", 1), leitura="pauli")
    X = np.random.default_rng(6).uniform(0, 1, (3, 3)); p = a.params_iniciais(6)
    err = np.abs(BackendQiskit("ideal", shots=20000, seed=1).features(X, a, p)
                 - BackendExato().features(X, a, p)).max()
    assert err < 0.04


def test_pauli2_por_bases_pede_sombras():
    pytest.importorskip("qiskit_aer")
    from quirios_engine import BackendQiskit
    a = Ansatz(QuantumConfig(3, "anel", 1), leitura="pauli2")
    with pytest.raises(ValueError, match="sombras"):
        BackendQiskit("ideal", shots=100).features(np.zeros((1, 3)), a, a.params_iniciais(0))


def test_sombras_classicas_estimam_paulis_de_2_qubits():
    pytest.importorskip("qiskit_aer")
    from quirios_engine import BackendQiskit
    a = Ansatz(QuantumConfig(3, "anel", 1), leitura="pauli2", emaranhador="cz")
    X = np.random.default_rng(7).uniform(0, 1, (2, 3)); p = a.params_iniciais(7)
    b = BackendQiskit("ideal", shots=30000, n_sombras=300, medicao="sombras", seed=2)
    err = np.abs(b.features(X, a, p) - BackendExato().features(X, a, p))
    assert err.mean() < 0.03 and err.max() < 0.15


# ── mitigação de erro ───────────────────────────────────────────────────────

def _erro_medio(backend, a, X, p):
    return float(np.abs(backend.features(X, a, p) - BackendExato().features(X, a, p)).mean())


def test_mitigacao_de_leitura_remove_o_erro_de_medida():
    pytest.importorskip("qiskit_aer")
    from quirios_engine import BackendQiskit
    a = Ansatz(QuantumConfig(4, "anel", 1))
    X = np.random.default_rng(8).uniform(0, 1, (4, 4)); p = a.params_iniciais(8)
    erros = {"erro_1q": 0.0, "erro_2q": 0.0, "erro_leitura": 0.08}
    bruto = _erro_medio(BackendQiskit("parametrico", erros=erros, shots=20000, seed=3), a, X, p)
    mitigado = _erro_medio(BackendQiskit("parametrico", erros=erros, shots=20000, seed=3,
                                         mitigacao=("leitura",)), a, X, p)
    assert mitigado < 0.4 * bruto


def test_zne_reduz_o_erro_de_portas():
    pytest.importorskip("qiskit_aer")
    from quirios_engine import BackendQiskit
    a = Ansatz(QuantumConfig(4, "anel", 2))
    X = np.random.default_rng(9).uniform(0, 1, (4, 4)); p = a.params_iniciais(9)
    erros = {"erro_1q": 0.0, "erro_2q": 0.03, "erro_leitura": 0.0}
    bruto = _erro_medio(BackendQiskit("parametrico", erros=erros, shots=40000, seed=4), a, X, p)
    zne = _erro_medio(BackendQiskit("parametrico", erros=erros, shots=40000, seed=4,
                                    mitigacao=("zne",)), a, X, p)
    assert zne < 0.7 * bruto


def test_dobrar_portas():
    pytest.importorskip("qiskit")
    a = Ansatz(QuantumConfig(3, "anel", 1), emaranhador="rzz")
    qc = a.circuito_qiskit(np.full(3, 0.3), a.params_iniciais(0), medir=False)
    conta = lambda c: sum(1 for i in c.data if i.operation.num_qubits == 2 and i.operation.name != "barrier")
    assert conta(dobrar_portas(qc, 3)) == 3 * conta(qc)
    assert dobrar_portas(qc, 1) is qc
    with pytest.raises(ValueError):
        dobrar_portas(qc, 2)


# ── hardware futuro ─────────────────────────────────────────────────────────

def test_geracoes_de_hardware():
    pytest.importorskip("qiskit_aer")
    from quirios_engine import BackendQiskit
    from quirios_engine.futuro import resolver_erros
    assert "p2q=3e-03" in BackendQiskit("parametrico", erros="hoje").nome
    with pytest.raises(ValueError):
        resolver_erros("2099")
    with pytest.raises(ValueError):
        resolver_erros({"erro_1q": 0.1})


def test_projetar_hardware_melhora_com_a_geracao():
    pytest.importorskip("qiskit_aer")
    from quirios_engine import projetar_hardware
    r = np.random.default_rng(10)
    X = r.uniform(0, 1, (24, 3)); y = (X[:, 0] > 0.5).astype(float)
    a = Ansatz(QuantumConfig(3, "anel", 2), n_features=3)
    linhas = projetar_hardware(X, y, a, a.params_iniciais(0), geracoes=["hoje", "tolerante_a_falhas"],
                               shots=4000)
    assert [l["geracao"] for l in linhas] == ["ideal", "hoje", "tolerante_a_falhas"]
    assert linhas[0]["erro_features"] == 0.0
    assert linhas[2]["erro_features"] < linhas[1]["erro_features"]


# ── plugins de backend ──────────────────────────────────────────────────────

def test_plugin_de_backend_por_entry_point(monkeypatch):
    import quirios_engine.backends as bk

    class Fake(Backend):
        nome = "provedor-x"
        def _calcular(self, X, ansatz, params):
            return np.zeros((len(X), ansatz.n_saida))

    class EP:
        def __init__(self, obj): self.obj = obj
        def load(self): return self.obj

    monkeypatch.setattr(bk, "_plugins", lambda: {"provedor-x": EP(Fake), "quebrado": EP(dict)})
    assert "provedor-x" in backends_registrados()
    assert criar_backend("provedor-x").nome == "provedor-x"
    with pytest.raises(TypeError):
        criar_backend("quebrado")
    with pytest.raises(ValueError, match="disponíveis"):
        criar_backend("inexistente")


def test_recuperar_so_existe_na_ibm():
    pytest.importorskip("qiskit_aer")
    from quirios_engine import BackendQiskit
    with pytest.raises(ValueError):
        BackendQiskit("ideal").recuperar("abc", 4)
