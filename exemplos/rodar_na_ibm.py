"""
QURIOS em hardware quântico REAL (IBM Quantum).

1) Teste local, sem conta (usa o modelo de ruído do ibm_kingston):
       python exemplos/rodar_na_ibm.py --modo seco

2) Hardware real (precisa de conta gratuita em https://quantum.cloud.ibm.com):
       python exemplos/rodar_na_ibm.py --modo ibm --token SEU_TOKEN
   ou, depois de salvar a conta uma vez (ver README.pt-BR.md), só:
       python exemplos/rodar_na_ibm.py --modo ibm

3) Fila longa? Envie, desligue o computador e busque depois:
       python exemplos/rodar_na_ibm.py --modo ibm --so-enviar
       python exemplos/rodar_na_ibm.py --modo ibm --recuperar JOB_ID
   (use os mesmos --n-treino/--n-teste/--seed nas duas chamadas)

Envia TODOS os circuitos (treino + teste) num único job para economizar
tempo de QPU. Compara, nos MESMOS dados:
    clássico puro | + features quânticas exatas | + features do backend
e mede o quanto o hardware se afasta da simulação ideal.
Salva tudo em resultado_<modo>_<data>.json.
"""
import argparse, json, os, time, datetime
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from quirios_engine import QuantumConfig, Ansatz, BackendExato, BackendQiskit
from quirios_engine.vqc import logistica_l2, _sig

ap = argparse.ArgumentParser()
ap.add_argument("--modo", choices=["seco", "ibm"], default="seco")
ap.add_argument("--token", default=os.environ.get("QISKIT_IBM_TOKEN"))
ap.add_argument("--instancia", default=None)
ap.add_argument("--dispositivo", default="ibm_kingston")
ap.add_argument("--shots", type=int, default=2000)
ap.add_argument("--n-treino", type=int, default=40)
ap.add_argument("--n-teste", type=int, default=60)
ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--sim", action="store_true", help="não pedir confirmação antes de usar a QPU")
ap.add_argument("--so-enviar", action="store_true", help="(ibm) envia o job, mostra o id e sai")
ap.add_argument("--recuperar", metavar="JOB_ID", help="(ibm) busca o resultado de um job já enviado")
a = ap.parse_args()

# ── dados (público, real) ─────────────────────────────────────────────────────
d = load_breast_cancer()
X = MinMaxScaler().fit_transform(PCA(8, random_state=0).fit_transform(d.data))
Xr, Xte_all, yr, yte_all = train_test_split(X, d.target, test_size=120, stratify=d.target, random_state=42)
r = np.random.default_rng(a.seed)
i0, i1 = np.where(yr == 0)[0], np.where(yr == 1)[0]
n0 = max(1, round(a.n_treino * len(i0) / len(yr)))
sel = np.concatenate([r.choice(i0, n0, replace=False), r.choice(i1, a.n_treino - n0, replace=False)])
Xtr, ytr = Xr[sel], yr[sel].astype(float)
Xte, yte = Xte_all[:a.n_teste], yte_all[:a.n_teste]
X_job = np.vstack([Xtr, Xte])

# ── circuito fixo (mesmos parâmetros em todos os backends) ────────────────────
cfg = QuantumConfig(8, "anel", 2)
ans = Ansatz(cfg)
rp = np.random.default_rng(0)
params = ans.empacotar(rp.uniform(-0.1, 0.1, 8), rp.uniform(-np.pi, np.pi, 8), rp.uniform(-np.pi / 2, np.pi / 2, 8))

if a.modo == "ibm":
    if not a.sim and not a.recuperar:
        print(f"Vai enviar {len(X_job)} circuitos x {a.shots} shots para {a.dispositivo} (consome tempo de QPU do seu plano).")
        if input("Continuar? [s/N] ").strip().lower() != "s":
            raise SystemExit("Cancelado.")
    backend = BackendQiskit("ibm", shots=a.shots, dispositivo=a.dispositivo, token=a.token, instancia=a.instancia)
else:
    backend = BackendQiskit("ruidoso", shots=a.shots, dispositivo=a.dispositivo, seed=1)

t0 = time.time()
try:
    if a.recuperar:
        if a.modo != "ibm":
            raise SystemExit("--recuperar só vale com --modo ibm")
        tarefa = backend.recuperar(a.recuperar, cfg.n_qubits)
        print(f"Job {tarefa.id}: {tarefa.status()}")
    else:
        print(f"Enviando {len(X_job)} circuitos para {backend.nome}...")
        tarefa = backend.submeter(X_job, ans, params)
        print(f"Job enviado: {tarefa.id}")
        if a.so_enviar:
            raise SystemExit(f"\nStatus agora: {tarefa.status()}. Para buscar o resultado depois:\n"
                             f"  python exemplos/rodar_na_ibm.py --modo ibm --recuperar {tarefa.id} "
                             f"--n-treino {a.n_treino} --n-teste {a.n_teste} --seed {a.seed}")
    if a.modo == "ibm" and not tarefa.pronta():
        print("Aguardando a fila da IBM (Ctrl+C para sair; o job continua e pode ser recuperado pelo id)...")
    F_back = tarefa.resultado()
except Exception as e:
    if a.modo == "ibm" and "account" in str(e).lower():
        raise SystemExit(
            "\nConta IBM Quantum não encontrada. Faça uma vez:\n"
            "  1. Crie conta grátis em https://quantum.cloud.ibm.com\n"
            "  2. Copie sua API key (painel inicial)\n"
            "  3. Rode: python exemplos/rodar_na_ibm.py --modo ibm --token SUA_API_KEY\n"
            "     (ou salve de vez: python -c \"from qiskit_ibm_runtime import QiskitRuntimeService as S; "
            "S.save_account(channel='ibm_quantum_platform', token='SUA_API_KEY', set_as_default=True)\")")
    raise
t_exec = time.time() - t0
F_exato = BackendExato().features(X_job, ans, params)

# ── avaliação ─────────────────────────────────────────────────────────────────
def avaliar(Ftr, Fte):
    Ztr, Zte = np.hstack([Xtr, Ftr]), np.hstack([Xte, Fte])
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-9
    w, b = logistica_l2((Ztr - mu) / sd, ytr)
    return float(np.mean((_sig(((Zte - mu) / sd) @ w + b) > 0.5) == yte))

nt = len(Xtr)
vazio = np.zeros((len(X_job), 0))
res = {
    "data": datetime.datetime.now().isoformat(timespec="seconds"),
    "modo": a.modo, "backend": backend.nome, "job_id": tarefa.id,
    "shots": a.shots, "n_treino": nt, "n_teste": len(Xte), "tempo_execucao_s": round(t_exec, 1),
    "acuracia": {
        "classico_puro": avaliar(vazio[:nt], vazio[nt:]),
        "classico_+_quantico_exato": avaliar(F_exato[:nt], F_exato[nt:]),
        f"classico_+_quantico_{backend.nome}": avaliar(F_back[:nt], F_back[nt:]),
    },
    "fidelidade_features": {
        "erro_medio_vs_exato": float(np.abs(F_back - F_exato).mean()),
        "correlacao_vs_exato": float(np.corrcoef(F_back.ravel(), F_exato.ravel())[0, 1]),
    },
}
print(json.dumps(res, indent=2, ensure_ascii=False))
nome = f"resultado_{a.modo}_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
with open(nome, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
print(f"\nSalvo em {nome}")
