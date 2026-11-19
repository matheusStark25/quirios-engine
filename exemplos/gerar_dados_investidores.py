"""
Gera dados_investidores.json com execuções REAIS do QURIOS, para a página de
investidores. Cada número leva a sua origem (o comando que o reproduz).

    python exemplos/gerar_dados_investidores.py
    python exemplos/gerar_dados_investidores.py --comprovante comprovante_qpu_evolucao_ibm_....json
    python exemplos/gerar_dados_investidores.py --secoes memoria      # refaz só uma seção

Com --comprovante (gerado por `python -m quirios_engine.qpu ... --modo ibm`),
a página mostra a execução na QPU real; sem ele, mostra o ensaio no simulador
com o ruído do ibm_kingston. Com --secoes, as outras seções vêm do arquivo
de saída que já existe.
"""
import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from quirios_engine import __version__
from quirios_engine.benchmark import _preparar, carregar_dataset
from quirios_engine.dados_quanticos import comparar, fases_ising
from quirios_engine.evolucao_quantica import EvolucaoQuantica
from quirios_engine.qpu import _backend, _selar, dados_na_qpu, evolucao_na_qpu, memoria_na_qpu, verificar

SECOES = ("evolucao", "dados", "memoria", "qpu", "medidos")
ap = argparse.ArgumentParser()
ap.add_argument("--comprovante", nargs="*", default=[], help="comprovantes reais de QPU (JSON)")
ap.add_argument("--saida", default="dados_investidores.json")
ap.add_argument("--secoes", nargs="+", choices=SECOES, default=list(SECOES))
ap.add_argument("--rapido", action="store_true", help="versões menores (teste da página)")
a = ap.parse_args()
Path(a.saida).parent.mkdir(parents=True, exist_ok=True)
pasta = Path(a.saida).parent
t_ini = time.perf_counter()
saida = json.loads(Path(a.saida).read_text(encoding="utf-8")) if Path(a.saida).exists() else {}
saida.update(gerado_em=datetime.datetime.now().isoformat(timespec="seconds"), versao_quirios=__version__)
reais = {}
for p in map(Path, a.comprovante):
    reais.setdefault(json.loads(p.read_text(encoding="utf-8"))["tipo"], []).append(p)

X, y = carregar_dataset("breast_cancer")
Xp, Xt, yp, yt = train_test_split(X, y, test_size=120, stratify=y, random_state=42)
Xp, Xt = _preparar(Xp, Xt, 8)
sel = np.random.default_rng(0).choice(len(yp), 120, replace=False)

# 1. evolução quântica ao vivo (replay geração a geração)
if "evolucao" in a.secoes:
    print("evolução quântica (Breast Cancer)...", flush=True)
    eventos = []
    eq = EvolucaoQuantica(8, populacao=12, geracoes=3 if a.rapido else 6, seed=0, ao_progresso=eventos.append)
    res = eq.executar(Xp[sel], yp[sel].astype(float))
    from quirios_engine.orquestrador import DeGerador
    from quirios_engine.sistema import Quantico
    from quirios_engine.evolucao import GENOMA_PADRAO
    teste = {}
    for nome, g in (("padrao", GENOMA_PADRAO), ("campeao", res["melhor"]["genoma"])):
        q = Quantico.de_genoma(g, 8, seed=0)
        c = DeGerador(q).ajustar(Xp[sel], yp[sel].astype(float))
        teste[nome] = {"acuracia_teste": float(np.mean((c.predict_proba(Xt) > 0.5) == yt)),
                       "portas_2q": q.ansatz.portas_2q()}
    saida["evolucao"] = {
        "origem": "python -m quirios_engine.evolucao --dataset breast_cancer --n-qubits 8 --n-treino 120 "
                  "--populacao 12 --geracoes 6",
        "geracoes": [{k: e[k] for k in ("geracao", "melhor_acuracia", "entropia_bits", "genomas_distintos",
                                         "ligacao_max_bits", "ligacoes", "padrao_acuracia")} for e in eventos],
        "n_qubits_gerador": res["n_qubits_gerador"], "genomas_possiveis": res["genomas_possiveis"],
        "dataset": "breast_cancer", "n_qubits": 8, "n_treino": len(sel), "n_teste": len(yt),
        "populacao": eq.populacao, "geracoes_total": eq.geracoes,
        "melhor": {"descricao": res["melhor"]["descricao"], "acuracia_cv": res["melhor"]["acuracia"],
                   "portas_2q": res["melhor"]["custo"]},
        "teste_120_nunca_vistas": teste, "tempo_s": res["tempo_s"]}

# 2. dados quânticos de Ising: comparação JUSTA e teto teórico
if "dados" in a.secoes:
    print("dados quânticos (fases de Ising)...", flush=True)
    dq = []
    for shots in ((10,) if a.rapido else (10, 100)):
        d = fases_ising(8, 120 if a.rapido else 200, h_min=0.85, h_max=1.2, shots=shots, seed=0, preparo="circuito")
        justo = comparar(d, 40, 2 if a.rapido else 5, 0)
        teto = comparar(d, 40, 2 if a.rapido else 5, 0, justo=False)
        dq.append({"medicoes_por_sistema": 2 * shots, "fidelidade_preparo": justo["fidelidade_media"],
                   "classico": justo["modelos"][justo["comparacao"]["melhor_classico"]]["media"],
                   "melhor_classico": justo["comparacao"]["melhor_classico"],
                   "quantico_justo": justo["modelos"][justo["comparacao"]["melhor_quantico"]]["media"],
                   "quantico_teto_injusto": teto["modelos"][teto["comparacao"]["melhor_quantico"]]["media"],
                   "ganho_justo_pp": justo["comparacao"]["ganho_pp"], "p_valor_justo": justo["comparacao"]["p_valor"]})
    saida["dados_quanticos"] = {"origem": "python -m quirios_engine.dados_quanticos --n-spins 8 --amostras 200 "
                                          "--shots 10 100 --h-min 0.85 --h-max 1.2 --preparo circuito --teto",
                                "linhas": dq}

# 3. aprender com memória quântica: simulação exata + experimento na QPU (ou ensaio)
if "memoria" in a.secoes:
    print("memória quântica...", flush=True)
    from quirios_engine.memoria_quantica import comparar as comparar_memoria
    ns, cs, inst = ((3, 4), (8, 32), 40) if a.rapido else ((3, 4, 5, 6), (4, 8, 16, 32, 64, 128, 256), 200)
    r = comparar_memoria(ns, cs, instancias=inst)
    if "memoria" in reais:
        arqs = reais["memoria"]
    else:
        comp = memoria_na_qpu(_backend("ruidoso", shots=1), n_qubits=4, instancias=6 if a.rapido else 20,
                              copias=16 if a.rapido else 64)
        arqs = [_selar(comp, pasta / "comprovante_ensaio_memoria.json")]
    saida["memoria"] = {
        "origem": f"python -m quirios_engine.memoria_quantica --n-qubits {' '.join(map(str, ns))} "
                  f"--copias {' '.join(map(str, cs))} --instancias {inst}",
        "eps": r["eps"], "instancias": r["instancias"], "linhas": r["linhas"], "copias_para_90": r["copias_para_90"],
        "qpu": {"tipo": "real" if "memoria" in reais else "ensaio",
                "origem": "python -m quirios_engine.qpu memoria --modo " + ("ibm" if "memoria" in reais else "ruidoso"),
                "comprovantes": [json.loads(p.read_text(encoding="utf-8")) for p in arqs],
                "verificacao": [dict(verificar(p), arquivo=p.name) for p in arqs]}}

# 4. QPU: comprovantes reais, ou ensaio com o ruído do ibm_kingston
if "qpu" in a.secoes:
    print("QPU (comprovantes ou ensaio)...", flush=True)
    arquivos = reais.get("evolucao", []) + reais.get("dados", [])
    if arquivos:
        saida["qpu"] = {"tipo": "real"}
    else:
        Xq, yq = Xp[sel][:60, :6], yp[sel][:60].astype(float)
        ev = evolucao_na_qpu(Xq, yq, _backend("ruidoso", shots=24), 6, populacao=24, geracoes=1 if a.rapido else 2)
        dd = dados_na_qpu(_backend("ruidoso", shots=60), n_spins=6, n_sistemas=16 if a.rapido else 24, shots=60)
        arquivos = [_selar(ev, pasta / "comprovante_ensaio_evolucao.json"),
                    _selar(dd, pasta / "comprovante_ensaio_dados.json")]
        saida["qpu"] = {"tipo": "ensaio", "origem": "python -m quirios_engine.qpu evolucao|dados --modo ruidoso"}
    saida["qpu"]["comprovantes"] = [json.loads(p.read_text(encoding="utf-8")) for p in arquivos]
    saida["qpu"]["verificacao"] = [dict(verificar(p), arquivo=p.name) for p in arquivos]

# 5. resultados medidos anteriormente (com o comando que os reproduz)
if "medidos" in a.secoes:
    saida["medidos"] = {
        "concentracao_kernels": {"origem": "exemplos/fase_5.py (seção 2)", "linhas": [
            {"qubits": 4, "fidelidade": 2.5e-2, "projetado": 2.9e-2},
            {"qubits": 8, "fidelidade": 1.8e-3, "projetado": 1.4e-2},
            {"qubits": 12, "fidelidade": 6.2e-5, "projetado": 1.85e-2}]},
        "prontidao_hardware": {"origem": "quirios_engine.futuro.projetar_hardware (Breast Cancer PCA-6, 60 amostras)",
                               "linhas": [{"geracao": "ideal", "acuracia": 0.898, "erro": 0.0},
                                          {"geracao": "hoje (Heron r2)", "acuracia": 0.849, "erro": 0.0125},
                                          {"geracao": "3x melhor", "acuracia": 0.848, "erro": 0.0099},
                                          {"geracao": "10x melhor", "acuracia": 0.864, "erro": 0.0095},
                                          {"geracao": "tolerante a falhas", "acuracia": 0.864, "erro": 0.0095}]},
        "motores": {"origem": "comparação com mesmo orçamento: população 8, 5 gerações, 4 sorteios (Breast Cancer PCA-6)",
                    "linhas": [{"motor": "quântico emaranhado", "acuracia": 0.957, "desvio": 0.005, "avaliados": 33.0},
                               {"motor": "quântico sem emaranhamento", "acuracia": 0.956, "desvio": 0.006,
                                "avaliados": 38.8},
                               {"motor": "genético clássico", "acuracia": 0.955, "desvio": 0.004, "avaliados": 37.8}]},
        "mitigacao": {"origem": "BackendQiskit('parametrico', erros='hoje'), 60 circuitos de 6 qubits",
                      "linhas": [{"tecnica": "nenhuma", "erro": 0.0104}, {"tecnica": "leitura", "erro": 0.0076},
                                 {"tecnica": "ZNE", "erro": 0.0094}, {"tecnica": "leitura + ZNE", "erro": 0.0088}]},
    }
saida["tempo_ultima_execucao_s"] = round(time.perf_counter() - t_ini, 1)
Path(a.saida).write_text(json.dumps(saida, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
print(f"salvo em {a.saida} ({saida['tempo_ultima_execucao_s']}s)")
