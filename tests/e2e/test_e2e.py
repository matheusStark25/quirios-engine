"""Testes end-to-end: o QURIOS de verdade -- servidor HTTP real, linhas de
comando reais, cliente MCP real pelo protocolo. Rodar com: pytest -m e2e"""
import asyncio
import json
import os
import subprocess
import sys

import pytest

from .conftest import CHAVE_API, RAIZ, Servidor, dados, esperar_tarefa

pytestmark = pytest.mark.e2e


# ── API HTTP: fluxo completo de um usuário ──────────────────────────────────

def test_autenticacao(servidor):
    assert servidor.cliente(chave=None).get("/saude").status_code == 200
    assert servidor.cliente(chave=None).get("/modelos").status_code == 401
    assert servidor.cliente(chave="errada").get("/modelos").status_code == 401


def test_uvicorn_de_verdade_nao_sobe_sem_chave(tmp_path):
    from .conftest import porta_livre
    env = {k: v for k, v in os.environ.items() if k not in ("QURIOS_API_KEY", "QURIOS_CHAVES_API")}
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run([sys.executable, "-m", "uvicorn", "--app-dir", str(RAIZ), "quirios_engine.api:app",
                        "--port", str(porta_livre())], cwd=tmp_path, env=env, capture_output=True,
                       timeout=120, encoding="utf-8", errors="replace")
    assert r.returncode != 0                                          # saiu sozinho, sem nunca atender
    assert "não sobe" in r.stdout + r.stderr and "protecao chave criar" in r.stdout + r.stderr


def test_treinar_prever_orquestrar(servidor):
    cli = servidor.cliente()
    X, y = dados(seed=1)
    m = cli.post("/modelos", json={"circuito": {"n_qubits": 4, "n_camadas": 1}, "X": X, "y": y, "epocas": 3,
                                   "codificacao": "iqp", "emaranhador": "rzz", "leitura": "pauli"})
    assert m.status_code == 201, m.text
    p = cli.post(f"/modelos/{m.json()['id']}/prever", json={"X": X[:5]})
    assert p.status_code == 200 and len(p.json()["predicoes"]) == 5
    o = cli.post("/orquestrar", json={"circuito": {"n_qubits": 4, "n_camadas": 1}, "X": X, "y": y,
                                      "repeticoes": 1, "incluir_kernel": True})
    assert o.status_code == 201, o.text
    assert cli.post(f"/modelos/{o.json()['id']}/prever", json={"X": X[:3]}).status_code == 200


def test_treino_assincrono_com_mps(servidor):
    cli = servidor.cliente()
    X, y = dados(seed=2)
    r = cli.post("/modelos", json={"circuito": {"n_qubits": 4, "topologia": "linear", "n_camadas": 1},
                                   "backend": "mps", "otimizador": "spsa", "X": X, "y": y, "epocas": 2,
                                   "assincrono": True})
    assert r.status_code == 202
    t = esperar_tarefa(cli, r.json()["tarefa"]["id"])
    assert t["status"] == "concluida", t
    assert cli.post(f"/modelos/{t['resultado']['id']}/prever", json={"X": X[:2]}).status_code == 200


def test_evolucao_quantica_e_campeoes(servidor):
    cli = servidor.cliente()
    X, y = dados(n=64, seed=3, xor=True)
    r = cli.post("/evoluir", json={"circuito": {"n_qubits": 4}, "X": X, "y": y, "chave": "e2e-xor",
                                   "geracoes": 2, "populacao": 6})
    t = esperar_tarefa(cli, r.json()["tarefa"]["id"])
    assert t["status"] == "concluida", t
    res = t["resultado"]
    assert res["motor"] == "quantico" and res["entropia_inicial_bits"] == 16.0
    assert res["entropia_final_bits"] < 16.0
    assert isinstance(cli.get("/campeoes").json(), dict)


def test_dados_quanticos_e_projecao(servidor):
    cli = servidor.cliente()
    r = cli.post("/dados-quanticos", json={"n_spins": 4, "n_amostras": 40, "shots": [10], "n_treino": 16,
                                           "repeticoes": 1})
    t = esperar_tarefa(cli, r.json()["tarefa"]["id"])
    assert t["status"] == "concluida", t
    assert t["resultado"]["resultados"][0]["justo"] is True
    X, y = dados(n=16, q=3, seed=4)
    p = cli.post("/projetar", json={"circuito": {"n_qubits": 3, "n_camadas": 1}, "X": X, "y": y,
                                    "geracoes": ["hoje"], "shots": 500})
    assert p.status_code == 200 and len(p.json()["linhas"]) == 2


def test_painel_ao_vivo_pelo_fluxo_de_eventos(servidor):
    cli = servidor.cliente()
    assert "Evolução quântica ao vivo" in cli.get("/painel").text
    sid = cli.post("/painel/sessoes", json={"n_qubits": 4, "geracoes": 2, "populacao": 6}).json()["id"]
    tipos = []
    with cli.stream("GET", f"/painel/sessoes/{sid}/eventos") as r:
        for linha in r.iter_lines():
            if linha.startswith("data: "):
                tipos.append(json.loads(linha[6:])["tipo"])
    assert tipos[0] == "inicio" and tipos.count("geracao") == 3 and tipos[-1] == "fim"


def test_persistencia_assinada_sobrevive_a_reinicio_e_recusa_adulteracao(tmp_path):
    s = Servidor(tmp_path).iniciar()
    try:
        cli = s.cliente()
        X, y = dados(seed=5)
        ids = [cli.post("/modelos", json={"circuito": {"n_qubits": 4, "n_camadas": 1}, "X": X, "y": y,
                                          "epocas": 1}).json()["id"] for _ in range(2)]
        antes = cli.post(f"/modelos/{ids[0]}/prever", json={"X": X[:4]}).json()["probabilidades"]
        assert (tmp_path / "modelos" / f"{ids[0]}.json.sig").exists()
        s.parar()
        arq = tmp_path / "modelos" / f"{ids[1]}.json"                  # adultera o segundo modelo
        arq.write_text(arq.read_text(encoding="utf-8").replace('"b": ', '"b": 1+'), encoding="utf-8")
        s.iniciar()
        cli = s.cliente()
        saude = cli.get("/saude").json()
        assert saude["modelos_rejeitados"] == [ids[1]]
        depois = cli.post(f"/modelos/{ids[0]}/prever", json={"X": X[:4]}).json()["probabilidades"]
        assert antes == pytest.approx(depois)
        assert cli.get(f"/modelos/{ids[1]}").status_code == 404
    finally:
        s.parar()


# ── linhas de comando ───────────────────────────────────────────────────────

def _rodar(args, cwd, timeout=900):
    r = subprocess.run([sys.executable, "-m", *args], cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout,
                       env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r.stdout


def test_cli_evolucao_dados_benchmark_agendamento(tmp_path):
    pytest.importorskip("sklearn")
    out = _rodar(["quirios_engine.evolucao", "--dataset", "iris", "--n-qubits", "4", "--n-treino", "40",
                  "--populacao", "4", "--geracoes", "1", "--registro", str(tmp_path / "c.json")], tmp_path)
    assert "superposição" in out and "padrão QURIOS" in out
    out = _rodar(["quirios_engine.dados_quanticos", "--n-spins", "4", "--amostras", "40", "--shots", "10",
                  "--n-treino", "16", "--repeticoes", "1"], tmp_path)
    assert "mesmo orçamento de medições" in out
    out = _rodar(["quirios_engine.memoria_quantica", "--n-qubits", "3", "--copias", "8", "16",
                  "--instancias", "20"], tmp_path)
    assert "mesmo número de cópias" in out and "n=3" in out
    if __import__("importlib").util.find_spec("torch"):
        out = _rodar(["quirios_engine.ia_tensorial", "velocidade", "--larguras", "256", "--posto", "4",
                      "--nucleos", "4"], tmp_path)
        assert "L=  256" in out and "posto baixo" in out
    out = _rodar(["quirios_engine.benchmark", "rodar", "--dataset", "iris", "--n-treino", "12", "--sorteios", "1",
                  "--n-qubits", "4", "--candidatos", "logistica", "quantico", "--historico",
                  str(tmp_path / "h.jsonl")], tmp_path)
    assert "iris" in out
    out = _rodar(["quirios_engine.agendamento", "mostrar", "--pasta", str(tmp_path / "ag")], tmp_path)
    assert "quirios_engine.evolucao" in out


def test_cli_qpu_com_comprovante_assinado(tmp_path):
    pytest.importorskip("qiskit_ibm_runtime")
    from quirios_engine.seguranca import gerar_chaves
    priv, pub = gerar_chaves(tmp_path / "chaves")
    comp = tmp_path / "comp_evolucao.json"
    out = _rodar(["quirios_engine.qpu", "evolucao", "--modo", "ruidoso", "--dataset", "iris", "--n-qubits", "4",
                  "--n-treino", "40", "--populacao", "8", "--geracoes", "1", "--saida", str(comp),
                  "--chave-privada", str(priv)], tmp_path)
    assert "jobs:" in out and comp.exists() and (tmp_path / "comp_evolucao.json.sig").exists()
    ver = json.loads(_rodar(["quirios_engine.qpu", "verificar", str(comp), "--chave-publica", str(pub)], tmp_path))
    assert ver["hash_ok"] and ver["assinatura_ok"] and len(ver["jobs"]) == 2       # 2 gerações = 2 jobs
    d = json.loads(comp.read_text(encoding="utf-8"))
    d["melhor"]["acuracia_cv"] = 1.0                                               # adulteração
    comp.write_text(json.dumps(d), encoding="utf-8")
    ver = json.loads(_rodar(["quirios_engine.qpu", "verificar", str(comp), "--chave-publica", str(pub)], tmp_path))
    assert ver["hash_ok"] is False and ver["assinatura_ok"] is False

    comp2 = tmp_path / "comp_dados.json"
    _rodar(["quirios_engine.qpu", "dados", "--modo", "ruidoso", "--n-spins", "4", "--sistemas", "12",
            "--shots", "30", "--saida", str(comp2)], tmp_path)
    d2 = json.loads(comp2.read_text(encoding="utf-8"))
    assert len(d2["jobs"]) == 2 and set(d2["acuracia_cv"]) == {
        "classico_medidas_na_qpu", "kernel_quantico_na_qpu", "kernel_quantico_teto_sem_ruido_infinitos_shots"}
    assert d2["medicoes_por_sistema"]["classico"] == d2["medicoes_por_sistema"]["quantico"] == 60

    comp3 = tmp_path / "comp_memoria.json"
    out = _rodar(["quirios_engine.qpu", "memoria", "--modo", "ruidoso", "--n-qubits", "2", "--instancias", "4",
                  "--copias", "8", "--saida", str(comp3)], tmp_path)
    d3 = json.loads(comp3.read_text(encoding="utf-8"))
    assert [j["circuitos"] for j in d3["jobs"]] == [4 * 4, 4 * 8] and "com_memoria_quantica" in out
    assert d3["copias_por_estado"]["com_memoria"] == d3["copias_por_estado"]["sem_memoria"] == 8
    assert json.loads(_rodar(["quirios_engine.qpu", "verificar", str(comp3)], tmp_path))["hash_ok"]


# ── MCP pelo protocolo (processo separado, stdio) ───────────────────────────

def test_mcp_pelo_protocolo_real(tmp_path):
    pytest.importorskip("mcp")
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    async def sessao():
        params = StdioServerParameters(command=sys.executable, args=["-m", "quirios_engine.mcp_servidor"],
                                       cwd=str(tmp_path), env={**__import__("os").environ,
                                                               "QURIOS_REGISTRO": str(tmp_path / "c.json")})
        async with stdio_client(params) as (leitura, escrita):
            async with ClientSession(leitura, escrita) as s:
                await s.initialize()
                nomes = {t.name for t in (await s.list_tools()).tools}
                status = json.loads((await s.call_tool("status_quirios", {})).content[0].text)
                X, y = dados(n=24, q=3, seed=6)
                treino = json.loads((await s.call_tool("treinar_vqc", {"X": X, "y": y, "epocas": 2}))
                                    .content[0].text)
                pred = json.loads((await s.call_tool("prever", {"modelo_id": treino["modelo_id"], "X": X[:3]}))
                                  .content[0].text)
                return nomes, status, pred
    nomes, status, pred = asyncio.run(sessao())
    assert {"evoluir_circuito", "comparar_dados_quanticos", "treinar_vqc"} <= nomes
    assert status["versao"] and len(pred["predicoes"]) == 3
