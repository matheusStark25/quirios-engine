"""Comprovante de execução na QPU e sessões do painel ao vivo (unidade)."""
import json

import pytest

from quirios_engine import qpu
from quirios_engine.painel import GerenciadorSessoes, Sessao, carregar_dados


def test_comprovante_selado_e_verificavel(tmp_path):
    comp = {"formato": "quirios.comprovante_qpu", "tipo": "evolucao", "backend": "ibm_kingston",
            "jobs": [{"job_id": "abc123", "uso_qpu_s": 2.5}], "melhor": {"acuracia_cv": 0.9}}
    p = qpu._selar(comp, tmp_path / "c.json")
    v = qpu.verificar(p)
    assert v["hash_ok"] is True and v["jobs"] == ["abc123"] and v["assinatura_ok"] is None
    d = json.loads(p.read_text(encoding="utf-8")); d["melhor"]["acuracia_cv"] = 1.0
    p.write_text(json.dumps(d), encoding="utf-8")
    assert qpu.verificar(p)["hash_ok"] is False


def test_comprovante_assinado(tmp_path):
    pytest.importorskip("cryptography.hazmat.primitives.asymmetric.mldsa")
    from quirios_engine.seguranca import Assinador, Verificador, gerar_chaves
    priv, pub = gerar_chaves(tmp_path)
    p = qpu._selar({"jobs": [], "tipo": "dados"}, tmp_path / "c.json", Assinador(priv))
    assert qpu.verificar(p, Verificador(pub))["assinatura_ok"] is True


def test_features_classicas_das_contagens():
    import numpy as np
    # 2 qubits: '00' e '11' meio a meio na base Z -> <ZZ> = 1, |M| = 1; base X toda em '00' -> <X> = 1
    f = qpu._features_classicas({"00": 50, "11": 50}, {"00": 100}, 2)
    assert np.allclose(f, [1, 1, 1, 1, 1])


def test_backend_qiskit_registra_jobs():
    pytest.importorskip("qiskit_aer")
    from qiskit import QuantumCircuit
    from quirios_engine import BackendQiskit
    b = BackendQiskit("ideal", shots=100, seed=1)
    qc = QuantumCircuit(2); qc.h(0); qc.cx(0, 1); qc.measure_all()
    cont = b.amostrar_lote([qc, qc], tipo="teste")
    assert len(cont) == 2 and set(cont[0]) <= {"00", "11"}
    j = b.registro_jobs[-1]
    assert j["tipo"] == "teste" and j["circuitos"] == 2 and j["shots"] == 100 and len(j["contagens_sha256"]) == 64


def test_job_de_features_registra_circuitos_reais_e_hash():
    # Bug do comprovante: registrava o número de amostras como número de circuitos, sem hash.
    pytest.importorskip("qiskit_aer")
    import numpy as np
    from quirios_engine import Ansatz, BackendQiskit, QuantumConfig
    a = Ansatz(QuantumConfig(3, "anel", 1), leitura="pauli")            # bases X, Y e Z
    b = BackendQiskit("ideal", shots=50, seed=2)
    b.features(np.random.default_rng(0).uniform(0, 1, (4, 3)), a, a.params_iniciais(0))
    j = b.registro_jobs[-1]
    assert j["tipo"] == "features" and j["circuitos"] == 4 * 3 and j["shots"] == 50
    assert len(j["contagens_sha256"]) == 64


def test_sessao_do_painel_publica_e_termina():
    s = Sessao("sintetico", n_qubits=3, geracoes=1, populacao=4)
    s.rodar()
    tipos = [e["tipo"] for e in s.eventos]
    assert tipos == ["inicio", "geracao", "geracao", "fim"] and s.terminou
    assert list(s.fluxo())[-1].startswith("data: ")
    assert carregar_dados("sintetico", 3)[0].shape == (96, 3)
    with pytest.raises(ValueError):
        Sessao("inexistente", 3, 1, 4)


def test_gerenciador_limita_sessoes(monkeypatch):
    import quirios_engine.painel as pn
    monkeypatch.setattr(pn.Sessao, "rodar", lambda self: None)          # nunca termina
    g = GerenciadorSessoes()
    for _ in range(pn.MAX_SESSOES_ATIVAS):
        g.criar(dataset="sintetico", n_qubits=3, geracoes=1, populacao=4)
    with pytest.raises(RuntimeError):
        g.criar(dataset="sintetico", n_qubits=3, geracoes=1, populacao=4)


# ── prova completa (qpu prova) ──────────────────────────────────────────────

def test_prova_sem_conta_ibm_explica_o_que_fazer(monkeypatch, tmp_path):
    from quirios_engine import conta_ibm
    monkeypatch.setattr(conta_ibm, "contas_salvas", lambda: {})
    with pytest.raises(SystemExit, match="conta_ibm"):
        qpu.prova("ibm", pasta=tmp_path, perguntar=lambda _: "s")
    assert not list(tmp_path.iterdir())                                       # nada rodou


def test_prova_na_ibm_pede_confirmacao_uma_vez(monkeypatch, tmp_path):
    from quirios_engine import conta_ibm
    monkeypatch.setattr(conta_ibm, "contas_salvas", lambda: {"padrao": {}})
    monkeypatch.setattr(conta_ibm, "processadores", lambda: [
        {"nome": "ibm_pequena", "qubits": 5, "fila": 0},                      # pequena demais para 8 qubits
        {"nome": "ibm_fila_curta", "qubits": 156, "fila": 3},
        {"nome": "ibm_fila_longa", "qubits": 156, "fila": 90}])
    perguntas = []
    with pytest.raises(SystemExit, match="Cancelado"):
        qpu.prova("ibm", pasta=tmp_path, perguntar=lambda t: perguntas.append(t) or "n")
    assert len(perguntas) == 1 and not list(tmp_path.iterdir())


def test_destaque_le_os_tres_comprovantes_como_eles_sao_gravados():
    assert qpu._destaque({"tipo": "memoria", "acuracia": {"com_memoria_quantica": 1.0, "sem_memoria": 0.55}}) == \
        "com memória quântica 100% · sem memória 55%"
    assert qpu._destaque({"tipo": "dados", "acuracia_cv": {"kernel_quantico_na_qpu": [0.75, 0.1],
                                                           "classico_medidas_na_qpu": [0.5, 0.2]}}) == \
        "kernel quântico 75% · clássico 50%"                                  # [média, desvio]
    assert qpu._destaque({"tipo": "evolucao", "melhor": {"acuracia_cv": 0.958},
                          "padrao": {"acuracia_cv": 0.917}}) == "campeão 96% · circuito padrão 92%"


def test_prova_em_ensaio_assina_confere_e_sobrevive_a_uma_falha(monkeypatch, tmp_path):
    pytest.importorskip("qiskit_aer")
    pytest.importorskip("qiskit_ibm_runtime")                                  # ruído do ibm_kingston
    pytest.importorskip("cryptography.hazmat.primitives.asymmetric.mldsa")
    from quirios_engine.seguranca import Verificador
    monkeypatch.chdir(tmp_path)

    def quebra(*a, **k):
        raise RuntimeError("fila da IBM caiu")
    monkeypatch.setattr(qpu, "dados_na_qpu", quebra)                           # um experimento falha
    r = qpu.prova("ruidoso", pasta=tmp_path / "prova", pequena=True)
    por = {l["experimento"]: l for l in r["experimentos"]}
    assert "fila da IBM caiu" in por["dados"]["erro"]
    for nome in ("memoria", "evolucao"):                                        # os outros seguem
        assert por[nome]["verificado"] is True and por[nome]["resultado"]
    assert "com memória quântica" in por["memoria"]["resultado"]
    assert (tmp_path / "segredos" / "quirios_mldsa65.pem").exists()            # privada fora da pasta pública
    pasta = tmp_path / "prova"
    assert not list(pasta.glob("*mldsa65.pem")) or all(p.name.endswith(".pub.pem") for p in pasta.glob("*.pem"))
    resumo = next(pasta.glob("resumo_prova_*.json"))
    ver = Verificador(pasta / "chave_publica_mldsa65.pub.pem")
    assert qpu.verificar(resumo, ver) == dict(qpu.verificar(resumo, ver), hash_ok=True, assinatura_ok=True)


def test_prova_na_iqm_sem_token_ou_sem_o_ambiente_nao_roda(monkeypatch, tmp_path):
    monkeypatch.setenv("QURIOS_IQM_TOKEN_ARQUIVO", str(tmp_path / "nada"))
    monkeypatch.setenv("IQM_TOKEN", "x")
    monkeypatch.delenv("IQM_TOKEN")
    with pytest.raises(SystemExit, match="conta_iqm|ambiente"):
        qpu.prova("iqm", pasta=tmp_path / "prova", perguntar=lambda _: "s")
    assert not (tmp_path / "prova").exists()


def test_prova_em_ensaio_com_o_ruido_de_um_chip_iqm(monkeypatch, tmp_path):
    pytest.importorskip("iqm.qiskit_iqm")                                     # só no ambiente da IQM
    pytest.importorskip("cryptography.hazmat.primitives.asymmetric.mldsa")
    monkeypatch.chdir(tmp_path)
    r = qpu.prova("ruidoso", dispositivo="iqm_garnet", pasta=tmp_path / "prova", pequena=True)
    assert all(l.get("verificado") for l in r["experimentos"]), r["experimentos"]
    assert "com memória quântica" in r["experimentos"][0]["resultado"]


def test_tempo_de_qpu_vem_de_usage_na_runtime_nova():
    # qiskit-ibm-runtime 0.50: metrics() não traz mais usage.quantum_seconds; job.usage() traz
    from quirios_engine.backends import BackendQiskit

    class JobIBM:
        def job_id(self):
            return "dbvlr3lhdvas73bia8cg"

        def metrics(self):
            return {"timestamps": {"created": "2026-11-19T20:35:07Z"}}

        def usage(self):
            return 4

    b = BackendQiskit("ibm", dispositivo="ibm_marrakesh")
    item = b._registrar_job(JobIBM(), "memoria-bell", 640, 1, [{"0": 1}])
    assert item["uso_qpu_s"] == 4.0 and item["horarios"]
