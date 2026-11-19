"""
Benchmark contínuo: mede, rodada após rodada, onde o quântico ganha do
clássico -- e registra tudo para acompanhar a evolução do QURIOS.

    python -m quirios_engine.benchmark rodar --dataset breast_cancer --n-treino 10 40 --sorteios 5
    python -m quirios_engine.benchmark resumo

Protocolo: conjunto de teste FIXO (estratificado); a cada sorteio, um
conjunto de treino novo do tamanho pedido sai do restante. PCA (se houver
mais features que qubits) e escala para [0, 1] são ajustados no restante
SEM rótulos -- o teste nunca participa. Todos os candidatos usam os mesmos
sorteios, então a comparação é pareada.

Cada resultado vira uma linha no histórico JSONL (versão do QURIOS, data,
dataset, n_treino, sorteio, candidato, acurácia, tempo, circuitos). O
resumo agrega por (dataset, qubits, n_treino) e compara o melhor quântico
com o melhor clássico (teste t pareado entre sorteios). Exige o extra
[classico] (scikit-learn).
"""
import argparse
import copy
import datetime
import json
import time
from pathlib import Path

import numpy as np

from .core import QuantumConfig
from .estatistica import comparar_pareado
from .orquestrador import DeGerador, DeModelo, Orquestrador, candidatos_padrao, escolher_backend
from .sistema import RFF, Brutos, Quantico
from .vqc import VQCClassifier

DATASETS = ("breast_cancer", "wine", "iris", "digitos")
CANDIDATOS = ("logistica", "rff", "svm", "quantico", "vqc", "vqc-spsa", "orquestrador")
HISTORICO_PADRAO = "benchmark_historico.jsonl"


def carregar_dataset(nome):
    from sklearn import datasets as ds
    if nome == "breast_cancer":
        d = ds.load_breast_cancer(); return d.data, d.target.astype(int)
    if nome == "wine":                                  # classe 0 vs resto
        d = ds.load_wine(); return d.data, (d.target == 0).astype(int)
    if nome == "iris":                                  # versicolor vs virginica (a parte difícil)
        d = ds.load_iris(); m = d.target > 0
        return d.data[m], (d.target[m] == 2).astype(int)
    if nome == "digitos":                               # 3 vs 8
        d = ds.load_digits(); m = np.isin(d.target, [3, 8])
        return d.data[m], (d.target[m] == 8).astype(int)
    raise ValueError(f"dataset deve ser um de {DATASETS}")


class _SVM:
    """SVC RBF com probabilidade via sigmoide da função de decisão (sem CV interna)."""

    def fit(self, X, y):
        from sklearn.svm import SVC
        self.m = SVC(C=1.0, gamma="scale").fit(X, y)
        return self

    def predict_proba(self, X):
        return 1 / (1 + np.exp(-self.m.decision_function(X)))


def criar_candidato(nome, config, seed):
    """Candidato novo (sem estado) -> (objeto, tipo)."""
    backend, _ = escolher_backend(config)
    if nome == "logistica":
        return DeGerador(Brutos(), nome="logistica"), "classico"
    if nome == "rff":
        return DeGerador(RFF(escala=1.0, seed=seed), nome="rff"), "classico"
    if nome == "svm":
        return DeModelo("svm", _SVM), "classico"
    if nome == "quantico":
        return DeGerador(Quantico(config, backend, seed=seed), nome="quantico"), "quantico"
    if nome == "vqc":
        return DeModelo("vqc", lambda: VQCClassifier(copy.deepcopy(config), epocas=15, seed=seed,
                                                     backend=backend), quantico=True), "quantico"
    if nome == "vqc-spsa":
        return DeModelo("vqc-spsa", lambda: VQCClassifier(copy.deepcopy(config), epocas=30, seed=seed,
                                                          otimizador="spsa", backend=backend),
                        quantico=True), "quantico"
    if nome == "orquestrador":
        return DeModelo("orquestrador", lambda: Orquestrador(candidatos_padrao(config, backend, seed=seed),
                                                             repeticoes=2, seed=seed)), "hibrido"
    raise ValueError(f"candidato deve ser um de {CANDIDATOS}")


def _preparar(X_pool, X_teste, n_dim):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler
    if X_pool.shape[1] > n_dim:
        pca = PCA(n_dim, random_state=0).fit(X_pool)
        X_pool, X_teste = pca.transform(X_pool), pca.transform(X_teste)
    esc = MinMaxScaler().fit(X_pool)
    return esc.transform(X_pool), np.clip(esc.transform(X_teste), 0, 1)


def rodar(dataset="breast_cancer", n_treino=(10, 40), sorteios=5, n_qubits=8,
          topologia="anel", n_camadas_entrelacamento=2, candidatos=("logistica", "rff", "svm", "quantico"),
          n_teste=120, historico=HISTORICO_PADRAO, verbose=True):
    """Executa o protocolo e anexa os resultados ao histórico. Devolve os registros."""
    from sklearn.model_selection import train_test_split
    from . import __version__
    X, y = carregar_dataset(dataset)
    n_teste = min(n_teste, int(0.3 * len(y)))
    X_pool, X_te, y_pool, y_te = train_test_split(X, y, test_size=n_teste, stratify=y, random_state=42)
    nq = min(n_qubits, X.shape[1])
    X_pool, X_te = _preparar(X_pool, X_te, nq)
    cfg = QuantumConfig(nq, topologia, n_camadas_entrelacamento)
    data = datetime.datetime.now().isoformat(timespec="seconds")
    registros = []
    for nt in n_treino:
        for s in range(sorteios):
            r = np.random.default_rng(1000 + s)
            i0, i1 = np.where(y_pool == 0)[0], np.where(y_pool == 1)[0]
            n0 = int(np.clip(round(nt * len(i0) / len(y_pool)), 2, nt - 2))
            sel = np.concatenate([r.choice(i0, n0, replace=False), r.choice(i1, nt - n0, replace=False)])
            Xtr, ytr = X_pool[sel], y_pool[sel].astype(float)
            for nome in candidatos:
                cand, tipo = criar_candidato(nome, cfg, s)
                t0 = time.perf_counter()
                cand.ajustar(Xtr, ytr)
                acc = float(np.mean((cand.predict_proba(X_te) > 0.5) == y_te))
                modelo = getattr(cand, "modelo", None)
                reg = {"versao": __version__, "data": data, "dataset": dataset, "n_qubits": nq,
                       "topologia": topologia, "n_treino": nt, "n_teste": len(y_te), "sorteio": s,
                       "candidato": nome, "tipo": tipo, "acuracia": acc,
                       "tempo_s": round(time.perf_counter() - t0, 3),
                       "circuitos": int(getattr(modelo, "circuitos_executados", 0))}
                if nome == "orquestrador":
                    reg["escolha"] = modelo.relatorio["escolhido"]
                registros.append(reg)
                if verbose:
                    print(f"  {dataset} n={nt:3d} sorteio {s}  {nome:13s} {100 * acc:5.1f}%  "
                          f"{reg['tempo_s']:6.2f}s", flush=True)
    if historico:
        with open(historico, "a", encoding="utf-8") as f:
            for reg in registros:
                f.write(json.dumps(reg, ensure_ascii=False) + "\n")
    return registros


def resumir(registros):
    """Agrega por (dataset, n_qubits, n_treino): média por candidato e
    melhor quântico vs melhor clássico, pareado por sorteio."""
    grupos = {}
    for r in registros:
        grupos.setdefault((r["dataset"], r["n_qubits"], r["n_treino"]), []).append(r)
    saida = []
    for (ds, nq, nt), regs in sorted(grupos.items()):
        por = {}
        for r in regs:
            por.setdefault(r["candidato"], {"tipo": r["tipo"], "acc": {}})["acc"][r["sorteio"]] = r["acuracia"]
        linhas = {c: {"tipo": v["tipo"], "media": float(np.mean(list(v["acc"].values()))),
                      "desvio": float(np.std(list(v["acc"].values()))), "n": len(v["acc"])}
                  for c, v in por.items()}
        comp = None
        cl = [c for c in linhas if linhas[c]["tipo"] == "classico"]
        qu = [c for c in linhas if linhas[c]["tipo"] == "quantico"]
        if cl and qu:
            mc = max(cl, key=lambda c: linhas[c]["media"]); mq = max(qu, key=lambda c: linhas[c]["media"])
            comuns = sorted(set(por[mc]["acc"]) & set(por[mq]["acc"]))
            t = comparar_pareado([por[mq]["acc"][s] - por[mc]["acc"][s] for s in comuns])
            comp = {"melhor_classico": mc, "melhor_quantico": mq, "ganho_pp": round(100 * t["ganho"], 2),
                    "ic95_pp": [round(100 * t["ic"][0], 2), round(100 * t["ic"][1], 2)],
                    "p_valor": round(t["p_valor"], 4), "sorteios": len(comuns)}
        saida.append({"dataset": ds, "n_qubits": nq, "n_treino": nt, "candidatos": linhas, "comparacao": comp})
    return saida


def texto_resumo(resumo):
    out = []
    for g in resumo:
        out.append(f"\n{g['dataset']}  |  {g['n_qubits']} qubits  |  n_treino = {g['n_treino']}")
        for c, v in sorted(g["candidatos"].items(), key=lambda kv: -kv[1]["media"]):
            tag = {"classico": "C", "quantico": "Q", "hibrido": "H"}[v["tipo"]]
            out.append(f"  [{tag}] {c:14s} {100 * v['media']:5.1f}% ± {100 * v['desvio']:4.1f}  ({v['n']} sorteios)")
        cp = g["comparacao"]
        if cp:
            out.append(f"  quântico ({cp['melhor_quantico']}) - clássico ({cp['melhor_classico']}): "
                       f"{cp['ganho_pp']:+.2f} pp, IC95 [{cp['ic95_pp'][0]:+.2f}, {cp['ic95_pp'][1]:+.2f}], "
                       f"p={cp['p_valor']}")
    return "\n".join(out)


def ler_historico(caminho=HISTORICO_PADRAO):
    p = Path(caminho)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m quirios_engine.benchmark",
                                 description="Benchmark contínuo QURIOS: quântico vs clássico.")
    sub = ap.add_subparsers(dest="comando", required=True)
    pr = sub.add_parser("rodar", help="executa o protocolo e anexa ao histórico")
    pr.add_argument("--dataset", choices=DATASETS, default="breast_cancer")
    pr.add_argument("--n-treino", type=int, nargs="+", default=[10, 40])
    pr.add_argument("--sorteios", type=int, default=5)
    pr.add_argument("--n-qubits", type=int, default=8)
    pr.add_argument("--topologia", default="anel")
    pr.add_argument("--camadas", type=int, default=2, help="camadas de entrelaçamento")
    pr.add_argument("--candidatos", nargs="+", choices=CANDIDATOS,
                    default=["logistica", "rff", "svm", "quantico"])
    pr.add_argument("--historico", default=HISTORICO_PADRAO)
    ps = sub.add_parser("resumo", help="agrega o histórico")
    ps.add_argument("--historico", default=HISTORICO_PADRAO)
    ps.add_argument("--dataset", default=None)
    ps.add_argument("--json", action="store_true", help="imprime o resumo em JSON")
    a = ap.parse_args(argv)

    if a.comando == "rodar":
        regs = rodar(a.dataset, a.n_treino, a.sorteios, a.n_qubits, a.topologia, a.camadas,
                     a.candidatos, historico=a.historico)
        print(texto_resumo(resumir(regs)))
        print(f"\n{len(regs)} resultados anexados a {a.historico}")
    else:
        regs = ler_historico(a.historico)
        if a.dataset:
            regs = [r for r in regs if r["dataset"] == a.dataset]
        if not regs:
            print(f"histórico vazio: {a.historico}")
            return
        res = resumir(regs)
        print(json.dumps(res, indent=2, ensure_ascii=False) if a.json else texto_resumo(res))


if __name__ == "__main__":
    main()
