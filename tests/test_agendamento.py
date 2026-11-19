import platform
import sys
from types import SimpleNamespace

import pytest

from quirios_engine import agendamento as ag


def test_comando_da_evolucao_noturna(tmp_path):
    cmd = ag.comando_evolucao(dataset="iris", n_qubits=4, tempo_max=600, registro=tmp_path / "c.json",
                              min_portas_2q=4)
    assert cmd[0] == sys.executable and cmd[1:3] == ["-m", "quirios_engine.evolucao"]
    assert cmd[cmd.index("--rodadas") + 1] == "0" and cmd[cmd.index("--tempo-max") + 1] == "600"
    assert cmd[cmd.index("--motor") + 1] == "quantico" and "--min-portas-2q" in cmd


def test_script_gerado(tmp_path):
    script = ag.gerar_script(tmp_path / "ag", ag.comando_evolucao(registro=tmp_path / "c.json"), diretorio=tmp_path)
    txt = script.read_text(encoding="utf-8")
    assert "quirios_engine.evolucao" in txt and "evolucao_noturna.log" in txt
    assert script.suffix == (".cmd" if platform.system() == "Windows" else ".sh")


def test_agendadores():
    assert ag.comando_schtasks("Q", "03:30", "C:/x.cmd")[-3:] == ["/ST", "03:30", "/F"]
    assert ag.linha_cron("03:30", "/x.sh").startswith("30 3 * * *")
    with pytest.raises(ValueError):
        ag.linha_cron("25:00", "/x.sh")


def test_instalar_e_remover_sem_tocar_no_sistema(tmp_path):
    chamadas = []

    def falso(cmd, **kw):
        chamadas.append((cmd, kw))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    r = ag.instalar("04:00", "QTESTE", tmp_path / "ag", executar=falso, registro=tmp_path / "c.json")
    assert r["codigo"] == 0 and (tmp_path / "ag").exists()
    if platform.system() == "Windows":
        assert chamadas[0][0][:2] == ["schtasks", "/Create"] and "QTESTE" in chamadas[0][0]
    else:
        assert "QTESTE" in chamadas[-1][1]["input"]
    ag.remover("QTESTE", executar=falso)
    assert any("QTESTE" in str(c[0]) or "QTESTE" not in str(c[1].get("input", "")) for c in chamadas[-1:])


def test_cli_mostrar_nao_instala(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ag.main(["mostrar", "--hora", "02:15", "--pasta", str(tmp_path / "ag")])
    saida = capsys.readouterr().out
    assert "quirios_engine.evolucao" in saida and ("02:15" in saida or "15 2" in saida)
