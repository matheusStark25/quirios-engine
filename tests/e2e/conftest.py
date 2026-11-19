"""Infra dos testes end-to-end: servidor HTTP real (uvicorn em outro processo)."""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

httpx = pytest.importorskip("httpx")
RAIZ = Path(__file__).resolve().parents[2]          # pasta do projeto (contém quirios_engine/)
CHAVE_API = "e2e-chave-de-teste"


def porta_livre():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Servidor:
    def __init__(self, pasta):
        self.pasta = Path(pasta)
        self.porta = porta_livre()
        self.url = f"http://127.0.0.1:{self.porta}"
        self.proc = None
        from quirios_engine.seguranca import gerar_chaves
        self.priv, self.pub = gerar_chaves(self.pasta / "chaves")
        self.env = dict(os.environ, QURIOS_API_KEY=CHAVE_API, QURIOS_DIR_MODELOS=str(self.pasta / "modelos"),
                        QURIOS_REGISTRO=str(self.pasta / "campeoes.json"), QURIOS_CHAVE_PQC=str(self.priv),
                        QURIOS_CHAVE_PQC_PUBLICA=str(self.pub), PYTHONIOENCODING="utf-8",
                        # aqui se mede a capacidade do servidor; o limitador de taxa tem ataques
                        # próprios em test_protecao.py e test_api_protecao.py (com os limites padrão,
                        # 100 tarefas + 100 consultas em segundos esgotam a rajada de 120 de uma chave)
                        QURIOS_REQ_MINUTO="100000", QURIOS_RAJADA="10000")

    def iniciar(self):
        self.log = open(self.pasta / f"servidor_{time.time_ns()}.log", "w", encoding="utf-8")
        self.proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "--app-dir", str(RAIZ),
                                      "quirios_engine.api:app", "--port", str(self.porta)],
                                     cwd=self.pasta, env=self.env, stdout=self.log, stderr=subprocess.STDOUT)
        limite = time.time() + 60
        while time.time() < limite:
            try:
                if httpx.get(self.url + "/saude", timeout=1).status_code == 200:
                    return self
            except httpx.HTTPError:
                time.sleep(0.2)
        raise RuntimeError("servidor não subiu")

    def parar(self):
        if self.proc:
            self.proc.terminate()
            self.proc.wait(timeout=30)
            self.proc = None
            self.log.close()

    def cliente(self, chave=CHAVE_API, timeout=300):
        return httpx.Client(base_url=self.url, headers={"X-API-Key": chave} if chave else {}, timeout=timeout)


@pytest.fixture(scope="module")
def servidor(tmp_path_factory):
    s = Servidor(tmp_path_factory.mktemp("e2e")).iniciar()
    yield s
    s.parar()


def esperar_tarefa(cli, tid, limite=600):
    t0 = time.time()
    while time.time() - t0 < limite:
        r = cli.get(f"/tarefas/{tid}").json()
        if r["status"] in ("concluida", "erro", "cancelada"):
            return r
        time.sleep(0.3)
    raise TimeoutError(tid)


def dados(n=40, q=4, seed=0, xor=False):
    import numpy as np
    r = np.random.default_rng(seed)
    X = r.uniform(0, 1, (n, q))
    y = ((X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)) if xor else (X[:, 0] + X[:, 1] > 1)
    return X.tolist(), y.astype(int).tolist()
