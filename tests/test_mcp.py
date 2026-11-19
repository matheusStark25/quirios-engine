import asyncio
import json

import numpy as np
import pytest

pytest.importorskip("mcp")
from quirios_engine import mcp_servidor as m  # noqa: E402

FERRAMENTAS = {"status_quirios", "treinar_vqc", "prever", "orquestrar", "evoluir_circuito",
               "campeoes", "projetar_hardware", "diagnosticar_kernel", "comparar_dados_quanticos"}


def _dados(n=24, d=3, seed=0):
    r = np.random.default_rng(seed)
    X = r.uniform(-5, 5, (n, d))
    return X.tolist(), (X[:, 0] > 0).astype(int).tolist()


def test_ferramentas_publicadas():
    nomes = {t.name for t in asyncio.run(m.servidor.list_tools())}
    assert FERRAMENTAS <= nomes


def test_chamada_pelo_protocolo():
    r = asyncio.run(m.servidor.call_tool("status_quirios", {}))
    corpo = json.loads(r.content[0].text)
    assert corpo["versao"] and "exato" in corpo["backends"]


def test_treinar_e_prever():
    X, y = _dados()
    t = m.treinar_vqc(X, y, epocas=3)
    assert t["parametros"] > 0 and 0 <= t["acuracia_treino"] <= 1
    p = m.prever(t["modelo_id"], X[:4])
    assert len(p["predicoes"]) == 4
    with pytest.raises(ValueError):
        m.prever("nao-existe", X[:1])


def test_orquestrar_e_kernel():
    X, y = _dados(n=30, seed=1)
    o = m.orquestrar(X, y, incluir_kernel=True)
    assert o["relatorio"]["escolhido"] and "kernel-projetado" in o["texto"]
    k = m.diagnosticar_kernel(X, y)
    assert -1 <= k["alinhamento"] <= 1 and "variancia" in k


def test_evoluir_e_campeoes(tmp_path, monkeypatch):
    monkeypatch.setenv("QURIOS_REGISTRO", str(tmp_path / "c.json"))
    X, y = _dados(n=24, seed=2)
    e = m.evoluir_circuito(X, y, chave="mcp-teste", geracoes=1, populacao=3)
    assert e["motor"] == "quantico" and "superposição" in e["resumo"] and "promovido" in e
    c = m.evoluir_circuito(X, y, chave="mcp-classico", geracoes=1, populacao=3, motor="classico")
    assert c["motor"] == "classico"
    m._registro().registrar("mcp-teste", {"genoma": {"emaranhador": "cz"}, "descricao": "teste"}, promover=True)
    assert m.campeoes()["mcp-teste"]["descricao"] == "teste"


def test_dados_quanticos_pelo_mcp():
    r = m.comparar_dados_quanticos(n_spins=4, n_amostras=48, shots=10)
    assert r["comparacao"]["melhor_quantico"] and "Ising" in r["texto"]


def test_validacao_de_entrada():
    with pytest.raises(ValueError):
        m.treinar_vqc([[0.1, 0.2]], [1])
    X, _ = _dados()
    with pytest.raises(ValueError):
        m.treinar_vqc(X, [0] * len(X))
