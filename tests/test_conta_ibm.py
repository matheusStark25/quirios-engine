import pytest

pytest.importorskip("qiskit_ibm_runtime")
from quirios_engine import conta_ibm  # noqa: E402


class _ServicoFalso:
    salvo = {}

    def __init__(self, *a, **k):
        pass

    @classmethod
    def save_account(cls, **kw):
        cls.salvo = kw

    @classmethod
    def saved_accounts(cls):
        return {"default": {}} if cls.salvo else {}

    def backends(self, **kw):
        class B:
            def __init__(self, nome, q, fila):
                self.name, self.num_qubits, self._fila = nome, q, fila

            def status(self):
                return type("S", (), {"pending_jobs": self._fila})()
        return [B("ibm_torino", 133, 40), B("ibm_kingston", 156, 3)]


def test_salva_sem_mostrar_a_chave(monkeypatch, capsys):
    import qiskit_ibm_runtime
    _ServicoFalso.salvo = {}
    monkeypatch.setattr(qiskit_ibm_runtime, "QiskitRuntimeService", _ServicoFalso)
    chave = "chave-secreta-de-teste-1234567890"
    monkeypatch.setattr(conta_ibm, "_area_de_transferencia", lambda: "")      # nada copiado: pede para colar
    monkeypatch.setattr(conta_ibm.getpass, "getpass", lambda *_: chave)
    monkeypatch.setattr("builtins.input", lambda *_: "")
    conta_ibm.main(["--terminal"])
    out = capsys.readouterr().out
    assert chave not in out and "ibm_kingston" in out
    assert _ServicoFalso.salvo["token"] == chave and _ServicoFalso.salvo["channel"] == "ibm_quantum_platform"
    assert _ServicoFalso.salvo["instance"] is None
    assert [q["nome"] for q in conta_ibm.processadores()] == ["ibm_kingston", "ibm_torino"]   # fila mais curta antes


def test_no_windows_le_a_chave_copiada_sem_colar_e_limpa(monkeypatch, capsys):
    # colar (Ctrl+V) num campo escondido do Windows costuma chegar vazio: a chave copiada é lida direto
    import qiskit_ibm_runtime
    _ServicoFalso.salvo = {}
    monkeypatch.setattr(qiskit_ibm_runtime, "QiskitRuntimeService", _ServicoFalso)
    monkeypatch.setattr(conta_ibm.sys, "platform", "win32")
    chave = "Abc-123_chave-copiada-da-ibm-0123456789xy"
    limpou = []
    monkeypatch.setattr(conta_ibm, "_area_de_transferencia", lambda: f"  {chave}\r\n")   # vem com espaço e Enter
    monkeypatch.setattr(conta_ibm, "_limpar_area_de_transferencia", lambda: limpou.append(True))
    monkeypatch.setattr(conta_ibm.getpass, "getpass", lambda *_: pytest.fail("não devia pedir para colar"))
    monkeypatch.setattr("builtins.input", lambda *_: "")
    conta_ibm.main(["--terminal"])
    out = capsys.readouterr().out
    assert _ServicoFalso.salvo["token"] == chave and chave not in out and limpou == [True]
    assert f"{len(chave)} caracteres" in out


def test_verificar_sem_conta_e_chave_incompleta(monkeypatch):
    import qiskit_ibm_runtime
    _ServicoFalso.salvo = {}
    monkeypatch.setattr(qiskit_ibm_runtime, "QiskitRuntimeService", _ServicoFalso)
    with pytest.raises(SystemExit, match="Nenhuma conta"):
        conta_ibm.main(["--verificar"])
    with pytest.raises(ValueError, match="não parece"):
        conta_ibm.salvar("curta")
    with pytest.raises(ValueError, match="não parece"):                    # o comando colado no lugar da chave
        conta_ibm.salvar('&"C:\\Users\\x\\.venvs\\quirios\\Scripts\\python.exe"-mquirios_engine.conta_ibm')


def test_se_o_copiado_for_o_comando_nao_salva_e_pede_a_chave_de_novo(monkeypatch, capsys):
    import qiskit_ibm_runtime
    _ServicoFalso.salvo = {}
    monkeypatch.setattr(qiskit_ibm_runtime, "QiskitRuntimeService", _ServicoFalso)
    monkeypatch.setattr(conta_ibm.sys, "platform", "win32")
    chave = "Chave_IBM-de-verdade-0123456789abcdefghijk"
    copiados = iter(['& "C:\\Users\\mathe\\.venvs\\quirios\\Scripts\\python.exe" -m quirios_engine.conta_ibm',
                     chave])
    monkeypatch.setattr(conta_ibm, "_area_de_transferencia", lambda: next(copiados))
    monkeypatch.setattr(conta_ibm, "_limpar_area_de_transferencia", lambda: None)
    monkeypatch.setattr("builtins.input", lambda *_: "")
    conta_ibm.main(["--terminal"])
    out = capsys.readouterr().out
    assert "talvez seja o comando" in out and _ServicoFalso.salvo["token"] == chave and chave not in out


def test_a_janela_salva_a_chave_e_o_crn_sem_passar_pelo_terminal(monkeypatch, capsys):
    import qiskit_ibm_runtime
    _ServicoFalso.salvo = {}
    monkeypatch.setattr(qiskit_ibm_runtime, "QiskitRuntimeService", _ServicoFalso)
    chave, crn = "Chave_IBM-pela-janela-0123456789abcdefghij", "crn:v1:bluemix:public:quantum-computing:us-east:a/x::"
    monkeypatch.setattr(conta_ibm, "_janela_segredo", lambda *a, **k: (chave, crn))
    monkeypatch.setattr(conta_ibm.getpass, "getpass", lambda *_: pytest.fail("não devia pedir no terminal"))
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("não devia pedir no terminal"))
    conta_ibm.main([])
    assert _ServicoFalso.salvo["token"] == chave and _ServicoFalso.salvo["instance"] == crn
    assert chave not in capsys.readouterr().out
    monkeypatch.setattr(conta_ibm, "_janela_segredo", lambda *a, **k: (None, None))
    with pytest.raises(SystemExit, match="Nada mudou"):                      # fechou a janela: não salva nada
        conta_ibm.main([])
