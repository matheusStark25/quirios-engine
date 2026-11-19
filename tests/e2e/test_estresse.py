"""Testes de estresse contra o servidor real. Rodar com: pytest -m stress -s
Cada teste imprime as métricas medidas (latência, vazão, erros)."""
import json
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from .conftest import dados, esperar_tarefa

pytestmark = pytest.mark.stress


def _relatorio(nome, **m):
    print(f"\n[estresse] {nome}: " + ", ".join(f"{k}={v}" for k, v in m.items()), flush=True)


def _latencias(ts):
    ts = np.array(ts)
    return {"p50_s": round(float(np.percentile(ts, 50)), 3), "p95_s": round(float(np.percentile(ts, 95)), 3),
            "max_s": round(float(ts.max()), 3)}


def test_concorrencia_mista_sem_erros_5xx(servidor):
    cli = servidor.cliente()
    X, y = dados(n=24, seed=10)
    base = cli.post("/modelos", json={"circuito": {"n_qubits": 4, "n_camadas": 1}, "X": X, "y": y,
                                      "epocas": 1}).json()["id"]

    def chamada(i):
        c = servidor.cliente()
        t0 = time.perf_counter()
        if i % 4 == 0:
            r = c.post("/orquestrar", json={"circuito": {"n_qubits": 4, "n_camadas": 1}, "X": X, "y": y,
                                            "repeticoes": 1})
        elif i % 4 == 1:
            r = c.post(f"/modelos/{base}/prever", json={"X": X})
        else:
            r = c.post("/modelos", json={"circuito": {"n_qubits": 4, "n_camadas": 1}, "X": X, "y": y, "epocas": 1})
        return r.status_code, time.perf_counter() - t0

    t0 = time.perf_counter()
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(chamada, range(64)))
    total = time.perf_counter() - t0
    codigos = [c for c, _ in res]
    _relatorio("64 requisições concorrentes (16 clientes)", vazao_req_s=round(64 / total, 1),
               erros_5xx=sum(c >= 500 for c in codigos), **_latencias([t for _, t in res]))
    assert all(c in (200, 201) for c in codigos), codigos


def test_cem_tarefas_assincronas_concluem(servidor):
    cli = servidor.cliente()
    X, y = dados(n=16, seed=11)
    t0 = time.perf_counter()
    ids = [cli.post("/modelos", json={"circuito": {"n_qubits": 3, "n_camadas": 1}, "X": [x[:3] for x in X], "y": y,
                                      "epocas": 1, "assincrono": True}).json()["tarefa"]["id"] for _ in range(100)]
    status = [esperar_tarefa(cli, i)["status"] for i in ids]
    _relatorio("100 treinos assíncronos", tempo_total_s=round(time.perf_counter() - t0, 1),
               concluidas=status.count("concluida"))
    assert status.count("concluida") == 100


def test_evolucoes_concorrentes_nao_perdem_registros(servidor):
    cli = servidor.cliente()
    X, y = dados(n=40, seed=12, xor=True)
    antes = cli.get("/campeoes").json().get("estresse", {}).get("tentativas", 0)
    t0 = time.perf_counter()
    ids = [cli.post("/evoluir", json={"circuito": {"n_qubits": 4}, "X": X, "y": y, "chave": "estresse",
                                      "geracoes": 1, "populacao": 4, "seed": s}).json()["tarefa"]["id"]
           for s in range(6)]
    status = [esperar_tarefa(cli, i)["status"] for i in ids]
    assert status == ["concluida"] * 6
    hist = json.loads((servidor.pasta / "campeoes.json").read_text(encoding="utf-8"))["problemas"]["estresse"]
    _relatorio("6 evoluções simultâneas no mesmo registro", tempo_s=round(time.perf_counter() - t0, 1),
               tentativas_registradas=len(hist["historico"]) - antes)
    assert len(hist["historico"]) - antes == 6                   # nenhuma escrita perdida


def test_cargas_grandes(servidor):
    cli = servidor.cliente()
    r = np.random.default_rng(13)
    X = r.uniform(0, 1, (1500, 6)); y = (X[:, 0] > 0.5).astype(int)
    t0 = time.perf_counter()
    m = cli.post("/modelos", json={"circuito": {"n_qubits": 6, "n_camadas": 1}, "X": X.tolist(), "y": y.tolist(),
                                   "epocas": 1})
    t_treino = time.perf_counter() - t0
    assert m.status_code == 201, m.text[:300]
    Xp = r.uniform(0, 1, (5000, 6)).tolist()
    t0 = time.perf_counter()
    p = cli.post(f"/modelos/{m.json()['id']}/prever", json={"X": Xp})
    t_pred = time.perf_counter() - t0
    assert p.status_code == 200 and len(p.json()["predicoes"]) == 5000
    _relatorio("cargas grandes", treino_1500_amostras_s=round(t_treino, 1), previsao_5000_amostras_s=round(t_pred, 1),
               previsoes_por_s=round(5000 / t_pred))


def test_painel_limita_sessoes_simultaneas(servidor):
    cli = servidor.cliente()
    ids = [cli.post("/painel/sessoes", json={"n_qubits": 4, "geracoes": 3, "populacao": 6}) for _ in range(4)]
    codigos = [r.status_code for r in ids]
    assert codigos.count(201) == 3 and 429 in codigos
    for r in ids:
        if r.status_code == 201:
            sid = r.json()["id"]
            limite = time.time() + 300
            while not cli.get(f"/painel/sessoes/{sid}").json()["terminou"] and time.time() < limite:
                time.sleep(0.5)
    _relatorio("painel", aceitas=codigos.count(201), recusadas_429=codigos.count(429))


def test_entradas_malformadas_nunca_derrubam_o_servidor(servidor):
    cli = servidor.cliente()
    rng = random.Random(14)
    rotas = ["/modelos", "/orquestrar", "/evoluir", "/projetar", "/dados-quanticos", "/painel/sessoes",
             "/modelos/abc/prever", "/modelos/abc/entropia"]
    lixo = [None, "", "x", -1, 0, 1e308, float("nan"), [], {}, [[]], [[1, 2], [3]], {"a": [1]}, "∞", True,
            [[0.5] * 300] * 3, list(range(10_000))]
    codigos = []
    for _ in range(240):
        corpo = {k: rng.choice(lixo) for k in rng.sample(["X", "y", "circuito", "tipo", "backend", "n_qubits",
                                                           "geracoes", "shots", "chave", "x", "subsistema"],
                                                          rng.randint(0, 6))}
        try:
            texto = json.dumps(corpo, allow_nan=True)
        except (TypeError, ValueError):
            texto = "{"
        if rng.random() < 0.15:
            texto = texto[: rng.randint(0, len(texto))]               # JSON truncado
        r = cli.post(rng.choice(rotas), content=texto, headers={"Content-Type": "application/json"})
        codigos.append(r.status_code)
    _relatorio("240 corpos malformados", **{f"http_{c}": codigos.count(c) for c in sorted(set(codigos))})
    assert max(codigos) < 500
    r = cli.get("/saude")
    if r.status_code == 429:                # vivo e se protegendo da rajada: espera a ficha que ele anuncia
        time.sleep(float(r.headers["Retry-After"]))
        r = cli.get("/saude")
    assert r.status_code == 200


def test_mps_100_qubits_em_volume():
    pytest.importorskip("qiskit_aer")
    from quirios_engine import Ansatz, BackendMPS, QuantumConfig
    a = Ansatz(QuantumConfig(100, "linear", 1), leitura="pauli")
    X = np.random.default_rng(15).uniform(0, 1, (60, 100))
    t0 = time.perf_counter()
    F = BackendMPS().features(X, a, a.params_iniciais(0))
    dt = time.perf_counter() - t0
    _relatorio("MPS 100 qubits", amostras=60, features=F.shape[1], tempo_s=round(dt, 1),
               amostras_por_s=round(60 / dt, 1))
    assert F.shape == (60, 400) and np.all(np.abs(F) <= 1 + 1e-9)


def test_registro_concorrente_entre_processos(tmp_path):
    caminho = tmp_path / "campeoes.json"
    codigo = ("import sys; from quirios_engine.evolucao import RegistroCampeoes\n"
              "for i in range(10): RegistroCampeoes(sys.argv[1]).registrar('p', {'i': i, 'proc': sys.argv[2]}, i == 9)\n")
    procs = [subprocess.Popen([sys.executable, "-c", codigo, str(caminho), str(k)]) for k in range(4)]
    assert all(p.wait(timeout=300) == 0 for p in procs)
    hist = json.loads(caminho.read_text(encoding="utf-8"))["problemas"]["p"]["historico"]
    _relatorio("4 processos x 10 escritas no mesmo registro", registradas=len(hist))
    assert len(hist) == 40 and not (tmp_path / "campeoes.json.lock").exists()
