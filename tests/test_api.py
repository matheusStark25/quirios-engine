import time

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from quirios_engine import api  # noqa: E402

CHAVE_TESTE = "chave-dos-testes-de-unidade-0123"
cliente = TestClient(api.app, headers={"X-API-Key": CHAVE_TESTE})


@pytest.fixture(autouse=True, scope="module")
def _chave_configurada():
    """Sem chave o servidor não sobe: os testes rodam com uma chave, como em produção.
    Escopo de módulo: vale também para as fixtures de módulo (como modelo_vqc)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("QURIOS_API_KEY", CHAVE_TESTE)
        mp.delenv("QURIOS_CHAVES_API", raising=False)
        yield


def _dados(n=40, q=4, seed=0):
    r = np.random.default_rng(seed)
    X = r.uniform(0, 1, (n, q))
    y = (X[:, 0] > 0.5).astype(int)
    return X.tolist(), y.tolist()


def _esperar(tid, limite=120):
    t0 = time.time()
    while time.time() - t0 < limite:
        r = cliente.get(f"/tarefas/{tid}").json()
        if r["status"] in ("concluida", "erro", "cancelada"):
            return r
        time.sleep(0.2)
    raise TimeoutError(tid)


@pytest.fixture(scope="module")
def modelo_vqc():
    X, y = _dados()
    resp = cliente.post("/modelos", json={"tipo": "vqc", "circuito": {"n_qubits": 4, "n_camadas": 1},
                                          "X": X, "y": y, "epocas": 5})
    assert resp.status_code == 201, resp.text
    return resp.json()


def test_saude():
    resp = cliente.get("/saude")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_backends_disponiveis():
    corpo = cliente.get("/backends").json()
    nomes = [b["nome"] for b in corpo["backends"]]
    assert "exato" in nomes and "CPU" in corpo["dispositivos"]


def test_treinar_vqc_retorna_resumo(modelo_vqc):
    assert modelo_vqc["tipo"] == "vqc"
    assert len(modelo_vqc["historico_loss"]) >= 1
    assert 0 <= modelo_vqc["acuracia_treino"] <= 1
    assert modelo_vqc["circuitos_executados"] > 0


def test_prever(modelo_vqc):
    X, _ = _dados(n=5, seed=1)
    resp = cliente.post(f"/modelos/{modelo_vqc['id']}/prever", json={"X": X})
    assert resp.status_code == 200
    corpo = resp.json()
    assert len(corpo["predicoes"]) == 5
    assert all(0 <= p <= 1 for p in corpo["probabilidades"])


def test_prever_com_numero_errado_de_features(modelo_vqc):
    resp = cliente.post(f"/modelos/{modelo_vqc['id']}/prever", json={"X": [[0.1, 0.2, 0.3]]})
    assert resp.status_code == 422


def test_entropia(modelo_vqc):
    resp = cliente.post(f"/modelos/{modelo_vqc['id']}/entropia",
                        json={"x": [0.1, 0.5, 0.9, 0.3], "subsistema": [0, 1]})
    assert resp.status_code == 200
    corpo = resp.json()
    assert 0 <= corpo["entropia_emaranhamento"] <= 2 + 1e-9
    assert 0 < corpo["pureza"] <= 1


def test_listar_e_detalhar(modelo_vqc):
    ids = [m["id"] for m in cliente.get("/modelos").json()]
    assert modelo_vqc["id"] in ids
    assert cliente.get(f"/modelos/{modelo_vqc['id']}").status_code == 200


def test_treinar_sistema():
    X, y = _dados(seed=2)
    resp = cliente.post("/modelos", json={"tipo": "sistema", "circuito": {"n_qubits": 4, "n_camadas": 1},
                                          "X": X, "y": y})
    assert resp.status_code == 201, resp.text
    corpo = resp.json()
    assert set(corpo["relatorio_cv"]) == {"brutos", "rff(escala=1)", "quantico[exato]"}
    resp = cliente.post(f"/modelos/{corpo['id']}/entropia", json={"x": [0.1, 0.2, 0.3, 0.4]})
    assert resp.status_code == 422


def test_remover():
    X, y = _dados(n=8, seed=3)
    mid = cliente.post("/modelos", json={"circuito": {"n_qubits": 4, "n_camadas": 1},
                                         "X": X, "y": y, "epocas": 1}).json()["id"]
    assert cliente.delete(f"/modelos/{mid}").status_code == 204
    assert cliente.get(f"/modelos/{mid}").status_code == 404
    assert cliente.delete(f"/modelos/{mid}").status_code == 404


@pytest.mark.parametrize("ajuste", [
    lambda c: c.update(X=[[0.1, 0.2, 0.3, 0.4]] * 7 + [[0.1, 0.2]]),   # linhas de tamanhos diferentes
    lambda c: c.update(X=[[1.5, 0.2, 0.3, 0.4]] * 8),                  # fora de [0, 1] (ângulo)
    lambda c: c.update(y=[0] * 8),                                     # uma classe só
    lambda c: c.update(y=[0, 1] * 3),                                  # tamanhos diferentes
    lambda c: c["circuito"].update(n_qubits=20),                       # acima do limite do exato
    lambda c: c.update(codificacao="amplitude", X=[[0.5] * 17] * 8),   # 17 > 2^4 amplitudes
    lambda c: (c["circuito"].update(topologia="todos_pares", n_qubits=10),   # O(n^2) bloqueado
               c.update(X=[[0.5] * 10] * 8)),
])
def test_validacao_de_entrada(ajuste):
    X, y = _dados(n=8, seed=4)
    corpo = {"circuito": {"n_qubits": 4, "n_camadas": 1}, "X": X, "y": y, "epocas": 1}
    ajuste(corpo)
    assert cliente.post("/modelos", json=corpo).status_code == 422


def test_nan_e_infinito_sao_recusados_com_422():
    # Achado pelo teste de estresse: o erro 422 repetia o NaN e a conexão caía sem resposta.
    X, y = _dados(n=8, seed=4)
    X[0][0] = float("nan")
    corpo = {"circuito": {"n_qubits": 4, "n_camadas": 1}, "X": X, "y": y, "epocas": 1}
    for texto in (__import__("json").dumps(corpo), '{"x": [Infinity], "subsistema": [0]}'):
        rota = "/modelos" if texto.startswith('{"circuito"') else "/modelos/abc/entropia"
        r = cliente.post(rota, content=texto, headers={"Content-Type": "application/json"})
        assert r.status_code == 422 and r.json()["detail"]


def test_servidor_usa_blas_em_uma_thread():
    # Achado pelo teste de estresse: OpenBLAS multithread + threads da API = access violation.
    threadpoolctl = pytest.importorskip("threadpoolctl")
    blas = [p for p in threadpoolctl.threadpool_info() if p["user_api"] == "blas"]
    assert blas and all(p["num_threads"] == 1 for p in blas)


def test_modelo_inexistente():
    assert cliente.post("/modelos/nao-existe/prever", json={"X": [[0, 0, 0, 0]]}).status_code == 404


# ── fases 1-3 pela API ──────────────────────────────────────────────────────

def test_treino_assincrono():
    X, y = _dados(n=16, seed=5)
    resp = cliente.post("/modelos", json={"circuito": {"n_qubits": 4, "n_camadas": 1}, "X": X, "y": y,
                                          "epocas": 2, "assincrono": True})
    assert resp.status_code == 202
    tid = resp.json()["tarefa"]["id"]
    r = _esperar(tid)
    assert r["status"] == "concluida", r
    mid = r["resultado"]["id"]
    assert cliente.post(f"/modelos/{mid}/prever", json={"X": X[:2]}).status_code == 200
    assert tid in [t["id"] for t in cliente.get("/tarefas").json()]


def test_tarefa_inexistente():
    assert cliente.get("/tarefas/xyz").status_code == 404


def test_treino_mps_com_spsa():
    pytest.importorskip("qiskit_aer")
    X, y = _dados(n=16, seed=6)
    resp = cliente.post("/modelos", json={"circuito": {"n_qubits": 4, "topologia": "linear", "n_camadas": 1},
                                          "backend": "mps", "otimizador": "spsa", "X": X, "y": y, "epocas": 2})
    assert resp.status_code == 201, resp.text
    assert resp.json()["backend"] == "mps"


def test_reuploading_e_amplitude_pela_api():
    r = np.random.default_rng(7)
    X8 = r.uniform(0, 1, (16, 8)); y = (X8[:, 6] > 0.5).astype(int)
    resp = cliente.post("/modelos", json={"circuito": {"n_qubits": 4, "n_camadas": 1}, "X": X8.tolist(),
                                          "y": y.tolist(), "epocas": 2})
    assert resp.status_code == 201, resp.text
    corpo = resp.json()
    assert corpo["ansatz"]["n_camadas"] == 2 and corpo["n_features"] == 8
    assert cliente.post(f"/modelos/{corpo['id']}/prever", json={"X": X8[:2].tolist()}).status_code == 200

    Xa = r.normal(0, 1, (16, 8))
    resp = cliente.post("/modelos", json={"circuito": {"n_qubits": 4, "n_camadas": 1}, "codificacao": "amplitude",
                                          "X": Xa.tolist(), "y": y.tolist(), "epocas": 2})
    assert resp.status_code == 201, resp.text
    assert resp.json()["ansatz"]["codificacao"] == "amplitude"


# ── fase 4: orquestrador pela API ───────────────────────────────────────────

def test_orquestrar():
    X, y = _dados(n=30, seed=8)
    resp = cliente.post("/orquestrar", json={"circuito": {"n_qubits": 4, "n_camadas": 1}, "X": X, "y": y,
                                             "repeticoes": 1})
    assert resp.status_code == 201, resp.text
    corpo = resp.json()
    rel = corpo["relatorio"]
    assert rel["escolhido"] and rel["comparacao"]["melhor_quantico"] == "quantico[exato]"
    assert cliente.post(f"/modelos/{corpo['id']}/prever", json={"X": X[:3]}).status_code == 200


# ── fase 5 pela API ─────────────────────────────────────────────────────────

def test_treino_com_genoma_emergente():
    X, y = _dados(n=16, seed=10)
    resp = cliente.post("/modelos", json={"circuito": {"n_qubits": 4, "n_camadas": 1}, "X": X, "y": y,
                                          "epocas": 2, "codificacao": "iqp", "emaranhador": "rzz",
                                          "rotacoes": ["rx", "ry"], "leitura": "pauli"})
    assert resp.status_code == 201, resp.text
    a = resp.json()["ansatz"]
    assert (a["codificacao"], a["emaranhador"], a["leitura"]) == ("iqp", "rzz", "pauli")


def test_evoluir_e_campeoes(monkeypatch, tmp_path):
    monkeypatch.setenv("QURIOS_REGISTRO", str(tmp_path / "c.json"))
    X, y = _dados(n=24, seed=11)
    resp = cliente.post("/evoluir", json={"circuito": {"n_qubits": 4}, "X": X, "y": y, "chave": "api-teste",
                                          "geracoes": 1, "populacao": 3})
    assert resp.status_code == 202
    r = _esperar(resp.json()["tarefa"]["id"], limite=300)
    assert r["status"] == "concluida", r
    assert r["resultado"]["motor"] == "quantico" and r["resultado"]["entropia_inicial_bits"] == 16.0
    if not r["resultado"]["promovido"]:          # sem evidência sobre o padrão: registra um campeão à mão
        api._registro_campeoes().registrar("api-teste", {"genoma": {"emaranhador": "cz"}, "descricao": "t"},
                                           promover=True)
    c = cliente.get("/campeoes").json()
    assert "api-teste" in c and c["api-teste"]["tentativas"] >= 1
    o = cliente.post("/orquestrar", json={"circuito": {"n_qubits": 4, "n_camadas": 1}, "X": X, "y": y,
                                          "repeticoes": 1, "chave_campeao": "api-teste",
                                          "incluir_kernel": True})
    assert o.status_code == 201, o.text
    nomes = [cand["nome"] for cand in o.json()["relatorio"]["candidatos"]]
    assert "quantico[campeao]" in nomes and "kernel-projetado" in nomes


def test_dados_quanticos_pela_api():
    resp = cliente.post("/dados-quanticos", json={"n_spins": 4, "n_amostras": 40, "shots": [10],
                                                  "n_treino": 16, "repeticoes": 1})
    assert resp.status_code == 202
    r = _esperar(resp.json()["tarefa"]["id"], limite=300)
    assert r["status"] == "concluida", r
    res = r["resultado"]["resultados"][0]
    assert res["shots"] == 10 and "melhor_quantico" in res["comparacao"]
    ruim = cliente.post("/dados-quanticos", json={"h_min": 1.5, "h_max": 1.0})
    assert ruim.status_code == 422


def test_projetar_hardware_pela_api():
    pytest.importorskip("qiskit_aer")
    X, y = _dados(n=16, q=3, seed=12)
    resp = cliente.post("/projetar", json={"circuito": {"n_qubits": 3, "n_camadas": 1}, "X": X, "y": y,
                                           "geracoes": ["hoje"], "shots": 500})
    assert resp.status_code == 200, resp.text
    assert [l["geracao"] for l in resp.json()["linhas"]] == ["ideal", "hoje"]
    ruim = cliente.post("/projetar", json={"circuito": {"n_qubits": 3, "n_camadas": 1}, "X": X, "y": y,
                                           "geracoes": ["2099"]})
    assert ruim.status_code == 422


def test_modelos_assinados_com_ml_dsa(monkeypatch, tmp_path):
    pytest.importorskip("cryptography.hazmat.primitives.asymmetric.mldsa")
    from quirios_engine.seguranca import gerar_chaves
    priv, pub = gerar_chaves(tmp_path / "chaves")
    monkeypatch.setenv("QURIOS_DIR_MODELOS", str(tmp_path / "modelos"))
    monkeypatch.setenv("QURIOS_CHAVE_PQC", str(priv))
    monkeypatch.setenv("QURIOS_CHAVE_PQC_PUBLICA", str(pub))
    X, y = _dados(n=12, seed=13)
    mid = cliente.post("/modelos", json={"circuito": {"n_qubits": 4, "n_camadas": 1}, "X": X, "y": y,
                                         "epocas": 1}).json()["id"]
    assert (tmp_path / "modelos" / f"{mid}.json.sig").exists()
    with api._trava:
        api._modelos.pop(mid)
    assert api.carregar_persistidos(tmp_path / "modelos") >= 1
    arq = tmp_path / "modelos" / f"{mid}.json"
    arq.write_text(arq.read_text(encoding="utf-8").replace('"b": ', '"b": 1+'), encoding="utf-8")
    with api._trava:
        api._modelos.pop(mid)
    api.carregar_persistidos(tmp_path / "modelos")        # adulterado: recusado, sem derrubar o servidor
    assert mid not in api._modelos and "assinatura" in api._rejeitados[mid]
    assert mid in cliente.get("/saude").json()["modelos_rejeitados"]
    api._rejeitados.clear()


# ── infraestrutura: chave de acesso e persistência ──────────────────────────

def test_chave_de_api(monkeypatch):
    monkeypatch.setenv("QURIOS_API_KEY", "segredo-teste-0123456789")
    assert cliente.get("/modelos").status_code == 401
    assert cliente.get("/modelos", headers={"X-API-Key": "errada"}).status_code == 401
    assert cliente.get("/modelos", headers={"X-API-Key": "segredo-teste-0123456789"}).status_code == 200
    assert cliente.get("/saude").status_code == 200


def test_sem_chave_o_servidor_nao_sobe(monkeypatch, tmp_path):
    from quirios_engine.protecao import CofreDeChaves
    monkeypatch.delenv("QURIOS_API_KEY")
    with pytest.raises(RuntimeError, match="não sobe"):
        with TestClient(api.app):                                          # a partida roda a checagem
            pass
    monkeypatch.setenv("QURIOS_API_KEY", "curta")
    with pytest.raises(RuntimeError, match="curta demais"):
        with TestClient(api.app):
            pass
    monkeypatch.delenv("QURIOS_API_KEY")
    arq = tmp_path / "chaves.json"
    monkeypatch.setenv("QURIOS_CHAVES_API", str(arq))
    with pytest.raises(RuntimeError, match="nenhuma chave valendo"):       # arquivo que não existe
        with TestClient(api.app):
            pass
    _, ident = CofreDeChaves(arq).criar(["leitura"])
    CofreDeChaves(arq).revogar(ident)
    with pytest.raises(RuntimeError, match="nenhuma chave valendo"):       # só chaves revogadas
        with TestClient(api.app):
            pass
    chave, _ = CofreDeChaves(arq).criar(["leitura"], dias=30)
    with TestClient(api.app, headers={"X-API-Key": chave}) as c:            # com uma chave valendo, sobe
        assert c.get("/saude").json()["protecao"]["chaves"] == "cofre de chaves"
        assert c.get("/modelos").status_code == 200


def test_sem_chave_ninguem_entra_mesmo_sem_a_partida(monkeypatch):
    # segunda trava: se a checagem da partida fosse pulada, nenhuma rota protegida responde
    monkeypatch.delenv("QURIOS_API_KEY")
    monkeypatch.setattr(api, "_bloqueio", api.BloqueioForcaBruta(max_falhas=100))
    assert TestClient(api.app).get("/modelos").status_code == 401
    assert cliente.get("/backends").status_code == 401
    assert "recusa tudo" in cliente.get("/saude").json()["protecao"]["chaves"]


def test_persistencia_de_modelos(monkeypatch, tmp_path):
    monkeypatch.setenv("QURIOS_DIR_MODELOS", str(tmp_path))
    X, y = _dados(n=12, seed=9)
    mid = cliente.post("/modelos", json={"circuito": {"n_qubits": 4, "n_camadas": 1}, "X": X, "y": y,
                                         "epocas": 1}).json()["id"]
    assert (tmp_path / f"{mid}.json").exists()
    antes = cliente.post(f"/modelos/{mid}/prever", json={"X": X[:3]}).json()["probabilidades"]
    with api._trava:
        api._modelos.pop(mid)                      # simula reinício do servidor
    assert api.carregar_persistidos(tmp_path) >= 1
    depois = cliente.post(f"/modelos/{mid}/prever", json={"X": X[:3]}).json()["probabilidades"]
    assert np.allclose(antes, depois)
    assert cliente.delete(f"/modelos/{mid}").status_code == 204
    assert not (tmp_path / f"{mid}.json").exists()
