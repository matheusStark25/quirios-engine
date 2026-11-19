"""Evolução quântica: a população vive em superposição num circuito gerador."""
import numpy as np
import pytest

from quirios_engine import QuantumConfig
from quirios_engine.evolucao import ESPACO_PADRAO, GENOMA_PADRAO, RegistroCampeoes, chave_genoma, espaco_valido
from quirios_engine.evolucao_quantica import CodificacaoGenoma, EvolucaoQuantica, GeradorQuantico


def _dados(n=48, seed=0):
    r = np.random.default_rng(seed)
    X = r.uniform(0, 1, (n, 4))
    return X, ((X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)).astype(float)


def _eq(**kw):
    base = dict(n_qubits=4, populacao=6, geracoes=3, dobras=3, repeticoes=1, seed=0)
    base.update(kw)
    return EvolucaoQuantica(**base)


def _gerador(n=6, camadas=2, seed=0):
    return GeradorQuantico(n, [(i, (i + 1) % n) for i in range(n)], camadas)


# ── codificação genoma <-> qubits ───────────────────────────────────────────

def test_codificacao_ida_e_volta():
    cod = CodificacaoGenoma(espaco_valido(ESPACO_PADRAO, 4))
    assert cod.n_bits == 16 and cod.n_genomas == 15552
    r = np.random.default_rng(0)
    for _ in range(50):
        g = cod.decodificar(int(r.integers(1 << cod.n_bits)))
        assert chave_genoma(cod.decodificar(cod.codificar(g))) == chave_genoma(g)
    assert chave_genoma(cod.decodificar(cod.codificar(GENOMA_PADRAO))) == chave_genoma(GENOMA_PADRAO)


# ── circuito gerador ────────────────────────────────────────────────────────

def test_gerador_comeca_com_todos_os_genomas_em_superposicao():
    g = _gerador()
    p = g.probabilidades(g.params_iniciais(seed=3))
    assert np.allclose(p, 1 / 64)
    assert abs(g.entropia(g.params_iniciais(seed=3)) - 6) < 1e-9


def test_gerador_qiskit_identico():
    qi = pytest.importorskip("qiskit.quantum_info")
    g = _gerador()
    p = g.params_iniciais(1) + np.random.default_rng(1).normal(0, 0.4, g.n_params)
    qc = g.circuito_qiskit(p)
    qc.remove_final_measurements()
    assert np.allclose(qi.Statevector(qc).probabilities(), g.probabilidades(p), atol=1e-10)


@pytest.mark.parametrize("temperatura", [0.0, 0.2])
def test_gradiente_adjunto_igual_parameter_shift(temperatura):
    g = _gerador()
    p = g.params_iniciais(2) + np.random.default_rng(2).normal(0, 0.5, g.n_params)
    xs, w = [3, 17, 40], [0.5, 0.3, 0.2]
    _, dL = g.objetivo(p, xs, w, temperatura)
    assert np.abs(g.gradiente(p, dL) - g.gradiente_parameter_shift(p, xs, w, temperatura)).max() < 1e-10


def test_gerador_aprende_a_sortear_a_elite():
    g = _gerador()
    p = g.params_iniciais(0)
    alvo = [5, 7]                                    # diferem em 1 bit
    massa0 = g.probabilidades(p)[alvo].sum()
    for _ in range(80):
        _, grad = g.nll_e_gradiente(p, alvo, [0.5, 0.5])
        p = p - 0.3 * grad
    assert g.probabilidades(p)[alvo].sum() > 20 * massa0


def test_gerador_sem_emaranhamento_nao_cria_ligacao():
    X, y = _dados()
    res = _eq(emaranhamento=False).executar(X, y)
    assert all(h["ligacao_max_bits"] == 0.0 for h in res["historico"])


def test_rzz_nao_comeca_no_ponto_de_sela():
    g = _gerador()
    p = g.params_iniciais(0)
    p[g.n:] = 0.0                                    # RZZ em zero: estado real
    p[-g.n:] = 0.3
    _, dL = g.objetivo(p, [3], [1.0])
    grad = g.gradiente(p, dL)
    rzz = [k for t, _, k in g._portas() if t == "rzz"]
    assert np.abs(grad[rzz]).max() < 1e-12            # gradiente nulo: por isso começam aleatórias
    assert any(g.params_iniciais(0)[k] != 0 for k in rzz)


# ── evolução ────────────────────────────────────────────────────────────────

def test_evolucao_quantica_converge_e_cria_ligacoes():
    X, y = _dados()
    res = _eq(geracoes=4).executar(X, y)
    h = res["historico"]
    assert res["motor"] == "quantico" and res["n_qubits_gerador"] == 16
    assert h[0]["entropia_bits"] == pytest.approx(16.0)
    assert h[-1]["entropia_bits"] < h[0]["entropia_bits"] - 4          # a população concentrou
    assert h[0]["ligacao_max_bits"] == pytest.approx(0.0, abs=1e-9)    # uniforme: nenhuma ligação
    assert max(x["ligacao_max_bits"] for x in h) > 0.01                # o emaranhamento criou ligações
    assert res["melhor"]["fitness"] >= res["padrao"]["fitness"]
    assert [x["melhor"] for x in h] == sorted(x["melhor"] for x in h)


def test_evolucao_quantica_usa_o_registro_com_promocao_segura(tmp_path):
    X, y = _dados(n=80)
    reg = RegistroCampeoes(tmp_path / "c.json")
    r1 = _eq(registro=reg, chave="xor", dobras=4, repeticoes=3).executar(X, y)
    assert reg.historico("xor")[0]["motor"] == "quantico"
    assert "circuito padrão" in r1["motivo"]                 # sem campeão, o titular é o padrão
    if r1["promovido"]:
        assert reg.campeao("xor")["motor"] == "quantico"
        r2 = _eq(registro=reg, chave="xor", dobras=4, repeticoes=3).executar(X, y)
        assert r2["promovido"] is False


def test_populacao_sorteada_numa_qpu_simulada():
    pytest.importorskip("qiskit_aer")
    from quirios_engine import BackendQiskit
    X, y = _dados(n=30)
    res = _eq(geracoes=1, backend_gerador=BackendQiskit("ruidoso", seed=1)).executar(X, y)
    assert res["backend_gerador"].startswith("qiskit-ruidoso")
    assert res["historico"][0]["genomas_distintos"] >= 3


def test_evoluir_continuamente_com_motor_quantico(tmp_path):
    from quirios_engine.evolucao import evoluir_continuamente
    X, y = _dados()
    reg = RegistroCampeoes(tmp_path / "c.json")
    res = evoluir_continuamente(X, y, 4, reg, "xor", rodadas=2, motor="quantico", populacao=5, geracoes=1,
                                dobras=3, repeticoes=1)
    assert len(res) == 2 and all(r["motor"] == "quantico" for r in res)
    with pytest.raises(ValueError):
        evoluir_continuamente(X, y, 4, reg, "xor", rodadas=1, motor="magico")
