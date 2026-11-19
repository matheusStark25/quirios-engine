"""
Liga o QURIOS à sua conta da IBM Quantum, uma vez, nesta máquina.

    python -m quirios_engine.conta_ibm              # pede a API key (não aparece na tela) e salva
    python -m quirios_engine.conta_ibm --verificar  # só confere se há conta salva e lista as QPUs

A chave é digitada no SEU terminal e vai direto para o arquivo de contas do
Qiskit (~/.qiskit/qiskit-ibm.json). O QURIOS não a imprime, não a grava em
outro lugar e não a envia a nenhum serviço além da própria IBM. Conta grátis
em https://quantum.cloud.ibm.com (Open Plan: 10 minutos de QPU a cada 28 dias).
"""
import argparse
import getpass
import re
import subprocess
import sys


def _area_de_transferencia():
    """O texto copiado (Ctrl+C), lido sem aparecer na tela. Só no Windows; senão ''."""
    if sys.platform != "win32":
        return ""
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", "Get-Clipboard -Raw"],
                           capture_output=True, text=True, timeout=30)
        return r.stdout or ""
    except Exception:                          # pragma: no cover
        return ""


def _limpar_area_de_transferencia():
    if sys.platform == "win32":
        try:
            subprocess.run(["powershell", "-NoProfile", "-Command", "Set-Clipboard -Value ' '"],
                           capture_output=True, timeout=30)
        except Exception:                      # pragma: no cover
            pass


def _limpa(texto):
    """Tira espaços, quebras de linha e caracteres de controle que a colagem traz junto."""
    return "".join(c for c in (texto or "") if c.isprintable() and not c.isspace())


FORMATO_CHAVE_IBM = re.compile(r"[A-Za-z0-9_-]{30,80}")         # chaves da IBM Cloud: ~44 caracteres assim


def _janela_segredo(nome, formato, campo_extra=None):
    """Uma janelinha com campo escondido: nela o Ctrl+V sempre funciona (no terminal do
    Windows, nem sempre). Recusa o que não tem formato de chave e avisa quando o colado
    é o CRN. Devolve (segredo, extra) ou (None, None) se a janela for fechada."""
    import tkinter as tk
    res = {"valor": None, "extra": None}
    raiz = tk.Tk()
    raiz.title(f"QURIOS: {nome}")
    raiz.attributes("-topmost", True)
    tk.Label(raiz, text=f"Cole aqui a {nome} (Ctrl+V). Ela não aparece, é normal:").pack(padx=14, pady=(14, 4),
                                                                                       anchor="w")
    campo = tk.Entry(raiz, show="•", width=56)
    campo.pack(padx=14)
    extra = None
    if campo_extra:
        tk.Label(raiz, text=campo_extra).pack(padx=14, pady=(10, 4), anchor="w")
        extra = tk.Entry(raiz, width=56)
        extra.pack(padx=14)
    aviso = tk.Label(raiz, text="", fg="#b00020", wraplength=420, justify="left")
    aviso.pack(padx=14, pady=(8, 0), anchor="w")

    def salvar(*_):
        v = _limpa(campo.get())
        if not formato.fullmatch(v):
            dica = " Isso é o CRN da instância, não a chave." if v.lower().startswith("crn:") else ""
            aviso.config(text=f"Isso não parece uma {nome} ({len(v)} caracteres).{dica} Cole a {nome} de novo.")
            campo.delete(0, "end")
            return
        res["valor"], res["extra"] = v, (extra.get().strip() if extra is not None else None)
        raiz.destroy()

    tk.Button(raiz, text="Salvar", width=14, command=salvar).pack(pady=14)
    raiz.bind("<Return>", salvar)
    raiz.after(200, lambda: (raiz.lift(), raiz.focus_force(), campo.focus_set()))
    raiz.mainloop()
    return res["valor"], res["extra"]


def ler_segredo(nome, formato, tentativas=3):
    """No Windows, colar num campo escondido (Ctrl+V) costuma não chegar. Então o
    segredo é lido da área de transferência, sem aparecer, e ela é limpa depois.
    Só aceita texto no FORMATO de uma chave: se o copiado for outra coisa (o próprio
    comando, por exemplo), não salva e pede para copiar a chave de novo."""
    if sys.platform == "win32":
        print(f"Agora copie a {nome}: selecione e tecle Ctrl+C no navegador ou no chat (NÃO aqui no "
              f"terminal: aqui o Ctrl+C fecha o programa).")
        for _ in range(tentativas):
            input("Depois de copiar, tecle Enter aqui: ")
            valor = _limpa(_area_de_transferencia())
            if formato.fullmatch(valor):
                _limpar_area_de_transferencia()
                print(f"{nome} lida da área de transferência ({len(valor)} caracteres); área de transferência limpa.")
                return valor
            print(f"O que está copiado não parece uma {nome} ({len(valor)} caracteres; talvez seja o comando). "
                  f"Copie a {nome} e tente de novo.")
    valor = _limpa(getpass.getpass(f"Cole a {nome} aqui (ela não aparece na tela) e tecle Enter: "))
    if not formato.fullmatch(valor):
        raise ValueError(f"isso não parece uma {nome} ({len(valor)} caracteres); nada foi salvo")
    return valor


def contas_salvas():
    from qiskit_ibm_runtime import QiskitRuntimeService
    return QiskitRuntimeService.saved_accounts()


def processadores():
    """QPUs reais em operação que a conta salva enxerga, da fila mais curta para a mais longa."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    s = QiskitRuntimeService()
    out = []
    for b in s.backends(operational=True, simulator=False):
        try:
            fila = b.status().pending_jobs
        except Exception:                      # pragma: no cover
            fila = None
        out.append({"nome": b.name, "qubits": b.num_qubits, "fila": fila})
    return sorted(out, key=lambda d: (d["fila"] is None, d["fila"] or 0))


def salvar(chave, instancia=None):
    from qiskit_ibm_runtime import QiskitRuntimeService
    if not chave or not FORMATO_CHAVE_IBM.fullmatch(chave):
        raise ValueError(f"isso não parece uma API key da IBM ({len(chave or '')} caracteres; elas têm uns 44, só "
                         f"letras, números, - e _): copie a chave inteira e rode de novo")
    QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token=chave, instance=instancia or None,
                                      set_as_default=True, overwrite=True)


def _mostrar(qpus):
    for q in qpus:
        print(f"  {q['nome']:16s} {q['qubits']:4d} qubits   fila: {q['fila'] if q['fila'] is not None else '?'} jobs")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m quirios_engine.conta_ibm",
                                 description="Salva a conta da IBM Quantum nesta máquina (a chave não aparece na tela).")
    ap.add_argument("--verificar", action="store_true", help="só confere a conta salva e lista as QPUs")
    ap.add_argument("--terminal", action="store_true", help="pede a chave no terminal em vez de abrir a janela")
    a = ap.parse_args(argv)
    if a.verificar:
        if not contas_salvas():
            raise SystemExit("Nenhuma conta da IBM salva. Rode: python -m quirios_engine.conta_ibm")
        print("Conta da IBM encontrada. QPUs disponíveis:")
        _mostrar(processadores())
        return
    print("A API key é criada em https://quantum.cloud.ibm.com, no quadro 'API key' da página inicial.")
    chave = instancia = None
    if not a.terminal:
        try:
            print("Abrindo uma janela para você colar a API key...")
            chave, instancia = _janela_segredo("API key", FORMATO_CHAVE_IBM,
                                               "CRN da instância (opcional: pode deixar em branco)")
            if chave is None:
                raise SystemExit("Janela fechada sem salvar. Nada mudou.")
        except Exception as e:                              # sem tkinter ou sem tela: segue pelo terminal
            print(f"(sem janela: {type(e).__name__}; seguindo pelo terminal)")
    if chave is None:
        chave = ler_segredo("API key", FORMATO_CHAVE_IBM)
        instancia = input("CRN da instância (Enter para a padrão da conta): ").strip()
    instancia = (instancia or "").strip()
    salvar(chave, instancia)
    del chave
    print("Conta salva. Conferindo o acesso...")
    try:
        _mostrar(processadores())
    except Exception as e:
        raise SystemExit(f"A conta foi salva, mas a IBM recusou o acesso: {type(e).__name__}. Confira a chave e a "
                         f"instância e rode de novo.")
    print("Pronto. Agora o QURIOS roda na QPU real, por exemplo:\n"
          "  python -m quirios_engine.qpu memoria --modo ibm")


if __name__ == "__main__":
    main()
