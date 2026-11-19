"""
Evolução contínua agendada: o QURIOS roda a auto-evolução sozinho, toda
noite, e amanhece com circuitos melhores -- ou com o mesmo campeão, se
nada melhor foi comprovado. Ninguém precisa atualizar nada.

    python -m quirios_engine.agendamento mostrar  --hora 03:00 --dataset breast_cancer
    python -m quirios_engine.agendamento instalar --hora 03:00 --dataset breast_cancer
    python -m quirios_engine.agendamento remover

    mostrar   gera o script e mostra o que seria agendado (não muda o sistema)
    instalar  registra no sistema: Windows -> Agendador de Tarefas (schtasks);
              Linux/macOS -> acrescenta a linha ao crontab do usuário
    remover   tira o agendamento

O script gerado (pasta --pasta, padrão ./agendamento_quirios) roda
`python -m quirios_engine.evolucao --rodadas 0 --tempo-max <s>` com o motor
quântico e anexa a saída a evolucao_noturna.log. O registro de campeões
guarda o histórico de cada noite (promovido ou mantido, e por quê).
"""
import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

NOME_PADRAO = "QURIOS_evolucao_noturna"


def comando_evolucao(dataset="breast_cancer", n_qubits=8, n_treino=120, tempo_max=3600, registro=None,
                     motor="quantico", min_portas_2q=0, populacao=12, geracoes=6):
    """Linha de comando (lista) que a tarefa agendada executa."""
    registro = str(Path(registro or "campeoes_quirios.json").resolve())
    cmd = [sys.executable, "-m", "quirios_engine.evolucao", "--dataset", dataset, "--n-qubits", str(n_qubits),
           "--n-treino", str(n_treino), "--rodadas", "0", "--tempo-max", str(int(tempo_max)),
           "--registro", registro, "--motor", motor, "--populacao", str(populacao),
           "--geracoes", str(geracoes)]
    if min_portas_2q:
        cmd += ["--min-portas-2q", str(min_portas_2q)]
    return cmd


def _citar(arg):
    return f'"{arg}"' if (" " in arg or "\\" in arg or "/" in arg) else arg


def gerar_script(pasta, cmd, diretorio=None):
    """Escreve o script que a tarefa roda (.cmd no Windows, .sh nos demais)."""
    pasta = Path(pasta).resolve()
    pasta.mkdir(parents=True, exist_ok=True)
    diretorio = Path(diretorio or os.getcwd()).resolve()
    log = pasta / "evolucao_noturna.log"
    linha = " ".join(_citar(a) for a in cmd)
    if platform.system() == "Windows":
        script = pasta / "quirios_noturno.cmd"
        # chcp 65001: o cmd.exe passa a ler o resto do script em UTF-8 (caminhos com acento)
        script.write_text(f'@echo off\r\nchcp 65001 >nul\r\ncd /d "{diretorio}"\r\nset PYTHONIOENCODING=utf-8\r\n'
                          f'echo ==== %DATE% %TIME% ==== >> "{log}"\r\n{linha} >> "{log}" 2>&1\r\n',
                          encoding="utf-8")
    else:
        script = pasta / "quirios_noturno.sh"
        script.write_text(f'#!/bin/sh\ncd "{diretorio}"\nexport PYTHONIOENCODING=utf-8\n'
                          f'echo "==== $(date) ====" >> "{log}"\n{linha} >> "{log}" 2>&1\n', encoding="utf-8")
        script.chmod(0o755)
    return script


def _hora(hora):
    h, m = hora.split(":")
    if not (0 <= int(h) < 24 and 0 <= int(m) < 60):
        raise ValueError("hora deve estar no formato HH:MM")
    return int(h), int(m)


def comando_schtasks(nome, hora, script):
    _hora(hora)
    return ["schtasks", "/Create", "/TN", nome, "/TR", f'"{script}"', "/SC", "DAILY", "/ST", hora, "/F"]


def linha_cron(hora, script, nome=NOME_PADRAO):
    h, m = _hora(hora)
    return f'{m} {h} * * * "{script}"  # {nome}'


def instalar(hora="03:00", nome=NOME_PADRAO, pasta="agendamento_quirios", executar=subprocess.run, **kw):
    """Gera o script e registra a tarefa diária no sistema."""
    script = gerar_script(pasta, comando_evolucao(**kw))
    if platform.system() == "Windows":
        cmd = comando_schtasks(nome, hora, script)
        r = executar(cmd, capture_output=True, text=True)
        return {"sistema": "windows", "script": str(script), "comando": cmd, "codigo": r.returncode,
                "saida": (r.stdout or "") + (r.stderr or "")}
    atual = executar(["crontab", "-l"], capture_output=True, text=True)
    linhas = [l for l in (atual.stdout or "").splitlines() if nome not in l]
    novo = "\n".join(linhas + [linha_cron(hora, script, nome)]) + "\n"
    r = executar(["crontab", "-"], input=novo, capture_output=True, text=True)
    return {"sistema": "cron", "script": str(script), "linha": linha_cron(hora, script, nome), "codigo": r.returncode}


def remover(nome=NOME_PADRAO, executar=subprocess.run):
    if platform.system() == "Windows":
        r = executar(["schtasks", "/Delete", "/TN", nome, "/F"], capture_output=True, text=True)
        return {"codigo": r.returncode, "saida": (r.stdout or "") + (r.stderr or "")}
    atual = executar(["crontab", "-l"], capture_output=True, text=True)
    novo = "\n".join(l for l in (atual.stdout or "").splitlines() if nome not in l) + "\n"
    r = executar(["crontab", "-"], input=novo, capture_output=True, text=True)
    return {"codigo": r.returncode}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m quirios_engine.agendamento",
                                 description="Agenda a auto-evolução noturna do QURIOS.")
    ap.add_argument("acao", choices=["mostrar", "instalar", "remover"])
    ap.add_argument("--hora", default="03:00")
    ap.add_argument("--nome", default=NOME_PADRAO)
    ap.add_argument("--pasta", default="agendamento_quirios")
    ap.add_argument("--dataset", default="breast_cancer")
    ap.add_argument("--n-qubits", type=int, default=8)
    ap.add_argument("--n-treino", type=int, default=120)
    ap.add_argument("--tempo-max", type=int, default=3600, help="segundos de evolução por noite")
    ap.add_argument("--registro", default="campeoes_quirios.json")
    ap.add_argument("--motor", choices=["quantico", "classico"], default="quantico")
    ap.add_argument("--min-portas-2q", type=int, default=0)
    a = ap.parse_args(argv)
    kw = dict(dataset=a.dataset, n_qubits=a.n_qubits, n_treino=a.n_treino, tempo_max=a.tempo_max,
              registro=a.registro, motor=a.motor, min_portas_2q=a.min_portas_2q)
    if a.acao == "remover":
        print(remover(a.nome))
    elif a.acao == "instalar":
        r = instalar(a.hora, a.nome, a.pasta, **kw)
        print(f"agendado ({r['sistema']}), código {r['codigo']}: script {r['script']}")
        if r.get("saida"):
            print(r["saida"])
    else:
        script = gerar_script(a.pasta, comando_evolucao(**kw))
        print(f"script: {script}\n{script.read_text(encoding='utf-8')}")
        if platform.system() == "Windows":
            print("seria registrado com:", " ".join(comando_schtasks(a.nome, a.hora, script)))
        else:
            print("linha de crontab:", linha_cron(a.hora, script, a.nome))


if __name__ == "__main__":
    main()
