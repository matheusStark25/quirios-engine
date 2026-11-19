"""Conta da IQM e o caminho por backend.run (sem rede: a IQM é simulada aqui)."""
import os

import pytest

from quirios_engine import conta_iqm
from quirios_engine.backends import BackendQiskit, JobPorRun


@pytest.fixture
def sem_token(monkeypatch, tmp_path):
    monkeypatch.setenv("QURIOS_IQM_TOKEN_ARQUIVO", str(tmp_path / "iqm_token"))
    monkeypatch.setenv("IQM_TOKEN", "x")
    monkeypatch.delenv("IQM_TOKEN")                   # registra a volta ao estado sem a variável
    return tmp_path / "iqm_token"


def test_salva_o_token_sem_mostrar_e_entrega_so_ao_processo(monkeypatch, capsys, sem_token):
    from quirios_engine import conta_ibm
    token = "token-secreto-da-iqm-0123456789abcdef"
    monkeypatch.setattr(conta_ibm, "_area_de_transferencia", lambda: "")
    monkeypatch.setattr(conta_ibm.getpass, "getpass", lambda *_: token)
    monkeypatch.setattr("builtins.input", lambda *_: "")
    monkeypatch.setattr(conta_iqm, "computador", lambda d="garnet": {"nome": d, "qubits": 20})
    conta_iqm.main(["--terminal"])
    out = capsys.readouterr().out
    assert token not in out and "20 qubits" in out and "garnet:mock" in out
    assert sem_token.read_text(encoding="utf-8") == token
    assert conta_iqm.carregar_token() and os.environ["IQM_TOKEN"] == token


def test_sem_token_e_token_incompleto(sem_token):
    assert not conta_iqm.carregar_token()
    with pytest.raises(SystemExit, match="Nenhum token"):
        conta_iqm.main(["--verificar"])
    with pytest.raises(ValueError, match="incompleto"):
        conta_iqm.salvar("curto")
    assert not sem_token.exists()


class _JobIQMFalso:
    """Imita um job da IQM: contagens com o shots pedido e a timeline de execução."""
    n = 0

    def __init__(self, circuitos, shots):
        self.circuitos, self.shots = circuitos, shots
        self.id = f"iqm-{_JobIQMFalso.n}"
        _JobIQMFalso.n += 1

    def job_id(self):
        return self.id

    def status(self):
        return type("S", (), {"name": "DONE"})()

    def result(self):
        job = self

        class R:
            timeline = [{"status": "execution_started", "timestamp": "2026-11-19T10:00:00.000+00:00"},
                        {"status": "execution_ended", "timestamp": "2026-11-19T10:00:02.500+00:00"}]

            def get_counts(self, i):
                return {f"{job.circuitos[i]:03b}": job.shots}     # o "circuito" aqui é só o seu número
        return R()


class _BackendIQMFalso:
    name = "IQMGarnetBackend"

    def run(self, circuitos, shots):
        return _JobIQMFalso(circuitos, shots)


def test_job_por_run_mantem_a_ordem_separa_jobs_e_mede_o_tempo():
    circuitos = list(range(7))
    shots = [10, 10, 99, 10, 99, 10, 10]                  # dois grupos de shots
    j = JobPorRun(_BackendIQMFalso(), circuitos, shots, max_por_job=2)
    r = j.result()
    assert [list(x.data.meas.get_counts()) for x in r] == [[f"{i:03b}"] for i in circuitos]
    assert [x.data.meas.get_counts()[f"{i:03b}"] for i, x in enumerate(r)] == shots
    assert len(j.job_id().split(",")) == 4                 # 5 circuitos com 10 shots em 3 jobs + 2 com 99 em 1
    m = j.metrics()
    assert m["usage"]["quantum_seconds"] == 10.0 and len(m["timestamps"]) == 8
    assert j.status().name == "DONE"


def test_backend_iqm_registra_o_comprovante_igual_a_ibm(monkeypatch):
    b = BackendQiskit("iqm", shots=5, dispositivo="garnet")
    b._backend = _BackendIQMFalso()
    monkeypatch.setattr(b, "_transpilar", lambda c, backend: list(c))
    cont = b.amostrar_lote(list(range(450)), tipo="memoria-bell")
    assert len(cont) == 450 and cont[449] == {f"{449:03b}": 5}
    item = b.registro_jobs[-1]
    assert item["circuitos"] == 450 and len(item["job_id"].split(",")) == 3      # 200 + 200 + 50
    assert item["uso_qpu_s"] == 7.5 and item["contagens_sha256"] and item["backend"] == "IQMGarnetBackend"


def test_backend_iqm_sem_token_explica_o_que_fazer(sem_token):
    pytest.importorskip("iqm.qiskit_iqm")
    with pytest.raises(RuntimeError, match="conta_iqm"):
        BackendQiskit("iqm", dispositivo="garnet")._get_backend()
