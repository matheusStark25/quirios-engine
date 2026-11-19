"""
Liga o QURIOS à sua conta da IQM Resonance (Finlândia), uma vez, nesta máquina.

    python -m quirios_engine.conta_iqm              # pede o token (não aparece na tela) e salva
    python -m quirios_engine.conta_iqm --verificar  # confere o token salvo e o computador quântico

A IQM usa uma versão do Qiskit diferente da IBM, então estes comandos rodam no
ambiente próprio da IQM (com `pip install "iqm-client[qiskit]"`; ver README.pt-BR.md,
"Hardware quântico real"). Misturar os dois no mesmo ambiente rebaixa o Qiskit.

O token é digitado no SEU terminal e vai para ~/.quirios/iqm_token, na sua
pasta de usuário (fora do projeto e do git). O QURIOS não o imprime e só o
entrega à própria IQM, pela variável IQM_TOKEN dentro do processo. Conta
grátis em https://resonance.iqm.tech (plano Starter: até 30 créditos por mês).
"""
import argparse
import os
import re
from pathlib import Path

URL_RESONANCE = "https://resonance.iqm.tech"
# tokens da IQM: longos, sem espaço, aspas, barra invertida ou "&" (o que um comando colado por engano tem)
FORMATO_TOKEN = re.compile(r"[A-Za-z0-9._~+/=-]{20,4096}")
DISPOSITIVO_PADRAO = "garnet"            # 20 qubits; "garnet:mock" testa tudo sem gastar créditos


def arquivo_token():
    return Path(os.environ.get("QURIOS_IQM_TOKEN_ARQUIVO") or Path.home() / ".quirios" / "iqm_token")


def carregar_token():
    """Põe o token salvo em IQM_TOKEN (só neste processo). True se há token."""
    if os.environ.get("IQM_TOKEN"):
        return True
    p = arquivo_token()
    if p.exists():
        t = p.read_text(encoding="utf-8").strip()
        if t:
            os.environ["IQM_TOKEN"] = t
            return True
    return False


def salvar(token):
    if not token or not FORMATO_TOKEN.fullmatch(token):
        raise ValueError(f"o token parece incompleto ou não é um token ({len(token or '')} caracteres): copie o "
                         f"token inteiro em resonance.iqm.tech (Dashboard)")
    p = arquivo_token()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(token, encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)                          # só o dono lê (onde o sistema permite)
    except OSError:                                   # pragma: no cover
        pass
    os.replace(tmp, p)


def computador(dispositivo=DISPOSITIVO_PADRAO):
    """Conecta à IQM e devolve {'nome', 'qubits'} do computador quântico (não gasta créditos)."""
    if not carregar_token():
        raise RuntimeError("nenhum token da IQM salvo")
    from iqm.qiskit_iqm import IQMProvider
    b = IQMProvider(URL_RESONANCE, quantum_computer=dispositivo).get_backend()
    return {"nome": dispositivo, "qubits": int(b.num_qubits)}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m quirios_engine.conta_iqm",
                                 description="Salva o token da IQM Resonance nesta máquina (não aparece na tela).")
    ap.add_argument("--verificar", action="store_true", help="só confere o token salvo")
    ap.add_argument("--dispositivo", default=DISPOSITIVO_PADRAO)
    ap.add_argument("--terminal", action="store_true", help="pede o token no terminal em vez de abrir a janela")
    a = ap.parse_args(argv)
    if a.verificar:
        if not carregar_token():
            raise SystemExit("Nenhum token da IQM salvo. Rode: python -m quirios_engine.conta_iqm")
        c = computador(a.dispositivo)
        print(f"Token da IQM encontrado. {c['nome']}: {c['qubits']} qubits.")
        return
    print("Pegue o token em https://resonance.iqm.tech: entre na conta e, no Dashboard, clique em "
          "'Generate token'. Copie na hora: a IQM não mostra o mesmo token de novo.")
    from .conta_ibm import _janela_segredo, ler_segredo
    token = None
    if not a.terminal:
        try:
            print("Abrindo uma janela para você colar o token...")
            token, _ = _janela_segredo("chave (token) da IQM", FORMATO_TOKEN)
            if token is None:
                raise SystemExit("Janela fechada sem salvar. Nada mudou.")
        except Exception as e:                              # sem tkinter ou sem tela: segue pelo terminal
            print(f"(sem janela: {type(e).__name__}; seguindo pelo terminal)")
    if token is None:
        token = ler_segredo("chave (token) da IQM", FORMATO_TOKEN)
    salvar(token)
    del token
    print(f"Token salvo em {arquivo_token()}. Conferindo o acesso...")
    try:
        c = computador(a.dispositivo)
    except Exception as e:
        raise SystemExit(f"O token foi salvo, mas a IQM recusou o acesso: {type(e).__name__}. Confira o token "
                         f"e rode de novo.")
    print(f"Pronto: {c['nome']}, {c['qubits']} qubits. Teste sem gastar créditos e depois rode de verdade:\n"
          "  python -m quirios_engine.qpu prova --modo iqm --dispositivo garnet:mock\n"
          "  python -m quirios_engine.qpu prova --modo iqm")


if __name__ == "__main__":
    main()
