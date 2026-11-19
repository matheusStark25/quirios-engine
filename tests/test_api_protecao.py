"""Ataques contra a API com a proteção ligada (cofre de chaves, auditoria, cofre pós-quântico)."""
import json

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("cryptography.hazmat.primitives.asymmetric.mlkem")
from fastapi.testclient import TestClient  # noqa: E402

from quirios_engine import api, protecao  # noqa: E402


def _dados(n=16, q=3, seed=0):
    r = np.random.default_rng(seed)
    X = r.uniform(0, 1, (n, q))
    return X.tolist(), (X[:, 0] > 0.5).astype(int).tolist()


@pytest.fixture
def servidor_blindado(tmp_path, monkeypatch):
    from quirios_engine.seguranca import gerar_chaves
    cofre = protecao.CofreDeChaves(tmp_path / "chaves.json")
    chaves = {e: cofre.criar([e])[0] for e in ("leitura", "treino", "admin")}
    priv, pub = gerar_chaves(tmp_path / "pqc")
    kpriv, kpub = protecao.Cofre.gerar_chaves(tmp_path / "cofre")
    monkeypatch.delenv("QURIOS_API_KEY", raising=False)
    monkeypatch.setenv("QURIOS_CHAVES_API", str(tmp_path / "chaves.json"))
    monkeypatch.setenv("QURIOS_AUDITORIA", str(tmp_path / "trilha.jsonl"))
    monkeypatch.setenv("QURIOS_CHAVE_PQC", str(priv))
    monkeypatch.setenv("QURIOS_CHAVE_PQC_PUBLICA", str(pub))
    monkeypatch.setenv("QURIOS_COFRE_PUBLICA", str(kpub))
    monkeypatch.setenv("QURIOS_COFRE_PRIVADA", str(kpriv))
    monkeypatch.setenv("QURIOS_DIR_MODELOS", str(tmp_path / "modelos"))
    monkeypatch.setattr(api, "_bloqueio", protecao.BloqueioForcaBruta(max_falhas=5, janela_s=60, bloqueio_s=60))
    monkeypatch.setattr(api, "_limite", protecao.LimitadorTaxa(por_minuto=6000, rajada=500))
    api._trilhas.clear()
    yield TestClient(api.app), chaves, tmp_path, pub
    api._trilhas.clear()


def test_sem_chave_chave_falsa_e_forca_bruta(servidor_blindado):
    cli, chaves, _, _ = servidor_blindado
    assert cli.get("/saude").json()["protecao"] == {"chaves": "cofre de chaves", "auditoria": True,
                                                    "cofre_pos_quantico": True, "assinatura_pos_quantica": True}
    assert cli.get("/modelos").status_code == 401
    for _ in range(4):
        assert cli.get("/modelos", headers={"X-API-Key": "qur_chute"}).status_code == 401
    r = cli.get("/modelos", headers={"X-API-Key": chaves["admin"]})           # 5ª falha já bloqueou o IP
    assert r.status_code == 429 and int(r.headers["Retry-After"]) > 0


def test_cada_chave_so_faz_o_que_o_escopo_permite(servidor_blindado):
    cli, chaves, _, _ = servidor_blindado
    X, y = _dados()
    corpo = {"circuito": {"n_qubits": 3, "n_camadas": 1}, "X": X, "y": y, "epocas": 2}
    H = lambda e: {"X-API-Key": chaves[e]}
    assert cli.get("/modelos", headers=H("leitura")).status_code == 200
    assert cli.post("/modelos", json=corpo, headers=H("leitura")).status_code == 403
    mid = cli.post("/modelos", json=corpo, headers=H("treino")).json()["id"]
    assert cli.post(f"/modelos/{mid}/prever", json={"X": X[:2]}, headers=H("leitura")).status_code == 200
    assert cli.delete(f"/modelos/{mid}", headers=H("treino")).status_code == 403
    assert cli.delete(f"/modelos/{mid}", headers=H("admin")).status_code == 204


def test_apagar_modelo_apaga_o_cifrado_e_as_assinaturas(servidor_blindado):
    cli, chaves, tmp, _ = servidor_blindado
    X, y = _dados(seed=2)
    corpo = {"circuito": {"n_qubits": 3, "n_camadas": 1}, "X": X, "y": y, "epocas": 2}
    mid = cli.post("/modelos", json=corpo, headers={"X-API-Key": chaves["treino"]}).json()["id"]
    assert list((tmp / "modelos").glob(f"{mid}*"))
    assert cli.delete(f"/modelos/{mid}", headers={"X-API-Key": chaves["admin"]}).status_code == 204
    assert list((tmp / "modelos").glob(f"{mid}*")) == []                     # nada do modelo fica no disco


def test_escopo_desconhecido_no_arquivo_vira_403_e_nao_500(servidor_blindado):
    cli, _, tmp, _ = servidor_blindado
    arq = tmp / "chaves.json"
    chave, ident = protecao.CofreDeChaves(arq).criar(["leitura"])
    d = json.loads(arq.read_text(encoding="utf-8"))
    for e in d["chaves"]:
        if e["id"] == ident:
            e["escopos"] = ["superusuario"]                                    # editado à mão
    arq.write_text(json.dumps(d), encoding="utf-8")
    assert cli.get("/modelos", headers={"X-API-Key": chave}).status_code == 403


def test_modelo_gravado_cifrado_assinado_e_recarregado(servidor_blindado):
    cli, chaves, tmp, _ = servidor_blindado
    X, y = _dados(seed=1)
    corpo = {"circuito": {"n_qubits": 3, "n_camadas": 1}, "X": X, "y": y, "epocas": 2}
    mid = cli.post("/modelos", json=corpo, headers={"X-API-Key": chaves["treino"]}).json()["id"]
    pasta = tmp / "modelos"
    assert not (pasta / f"{mid}.json").exists()                                 # nada em texto aberto
    cifrado = (pasta / f"{mid}.json.cofre").read_bytes()
    assert b"params" not in cifrado and (pasta / f"{mid}.json.cofre.sig").exists()
    antes = cli.post(f"/modelos/{mid}/prever", json={"X": X[:4]}, headers={"X-API-Key": chaves["leitura"]}).json()
    with api._trava:
        api._modelos.pop(mid)
    assert api.carregar_persistidos() >= 1                                     # reinício: decifra e recarrega
    depois = cli.post(f"/modelos/{mid}/prever", json={"X": X[:4]}, headers={"X-API-Key": chaves["leitura"]}).json()
    assert antes == depois
    e = json.loads(cifrado)
    e["nonce"] = e["nonce"][::-1]
    (pasta / f"{mid}.json.cofre").write_bytes(json.dumps(e).encode())       # adulteração no disco
    with api._trava:
        api._modelos.pop(mid)
    api.carregar_persistidos()
    assert mid in api._rejeitados


def test_trilha_registra_tudo_e_denuncia_adulteracao(servidor_blindado):
    cli, chaves, tmp, pub = servidor_blindado
    from quirios_engine.seguranca import Verificador
    cli.get("/saude")
    cli.get("/modelos")
    cli.get("/modelos", headers={"X-API-Key": chaves["leitura"]})
    trilha = tmp / "trilha.jsonl"
    regs = [json.loads(l) for l in trilha.read_text(encoding="utf-8").splitlines()]
    assert [(r["rota"], r["status"], r["quem"]) for r in regs][-3:] == [
        ("/saude", 200, None), ("/modelos", 401, None), ("/modelos", 200, regs[-1]["quem"])]
    assert regs[-1]["quem"] and "X-API-Key" not in trilha.read_text(encoding="utf-8")   # a chave nunca vai à trilha
    assert protecao.verificar_trilha(trilha, Verificador(pub))["ok"]
    linhas = trilha.read_text(encoding="utf-8").splitlines()
    trilha.write_text("\n".join(l for l in linhas if '"status": 401' not in l) + "\n", encoding="utf-8")
    assert not protecao.verificar_trilha(trilha, Verificador(pub))["ok"]      # sumir com a tentativa falha


def test_cabecalhos_de_seguranca_e_corpo_grande(servidor_blindado, monkeypatch):
    cli, chaves, _, _ = servidor_blindado
    r = cli.get("/modelos", headers={"X-API-Key": chaves["leitura"]})
    assert r.headers["X-Content-Type-Options"] == "nosniff" and r.headers["X-Frame-Options"] == "DENY"
    assert "default-src 'none'" in r.headers["Content-Security-Policy"] and r.headers["Cache-Control"] == "no-store"
    assert "frame-ancestors 'none'" in cli.get("/painel").headers["Content-Security-Policy"]
    monkeypatch.setattr(api, "MAX_CORPO", 1000)
    grande = json.dumps({"X": [[0.5] * 50] * 50})
    r = cli.post("/modelos", content=grande, headers={"X-API-Key": chaves["treino"], "Content-Type": "application/json"})
    assert r.status_code == 413
    pedacos = (grande[i:i + 100].encode() for i in range(0, len(grande), 100))   # sem Content-Length
    r = cli.post("/modelos", content=pedacos, headers={"X-API-Key": chaves["treino"], "Content-Type": "application/json"})
    assert r.status_code == 411                                                # corpo em pedaços não fura o limite
    r = cli.post("/modelos", content=b"{}", headers={"X-API-Key": chaves["treino"], "Content-Length": "abc"})
    assert r.status_code in (400, 411)


def test_rota_com_separador_unicode_nao_estraga_a_trilha(servidor_blindado):
    cli, _, tmp, pub = servidor_blindado
    from quirios_engine.seguranca import Verificador
    cli.get("/saude")
    cli.get("/%E2%80%A8")                                                      # ataque sem chave nenhuma
    cli.get("/saude")
    assert protecao.verificar_trilha(tmp / "trilha.jsonl", Verificador(pub))["ok"]
