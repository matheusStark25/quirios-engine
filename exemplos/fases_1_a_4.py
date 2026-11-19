"""
Demonstração das fases 1 a 4 do QURIOS com dados reais (Breast Cancer).

    python exemplos/fases_1_a_4.py

1. Infra: o mesmo Ansatz em backends diferentes; 60 qubits com MPS; tarefa assíncrona.
2. Treino: custo em circuitos de parameter-shift vs SPSA; termômetro de barren plateau.
3. Dados: as 30 features originais em 8 qubits (re-uploading) e em 5 qubits (amplitude).
4. Orquestrador: clássico vs quântico com evidência estatística.
"""
import time

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from quirios_engine import (Ansatz, BackendExato, BackendMPS, QuantumConfig, VQCClassifier,
                            estimar_circuitos, variancia_gradiente)
from quirios_engine.orquestrador import Orquestrador, candidatos_padrao, escolher_backend


def titulo(t):
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


d = load_breast_cancer()
X_pool, X_te, y_pool, y_te = train_test_split(d.data, d.target, test_size=120, stratify=d.target,
                                              random_state=42)
r = np.random.default_rng(0)
sel = r.choice(len(y_pool), 60, replace=False)
pca = PCA(8, random_state=0).fit(X_pool)
mm = MinMaxScaler().fit(pca.transform(X_pool))
X8_tr, X8_te = mm.transform(pca.transform(X_pool[sel])), np.clip(mm.transform(pca.transform(X_te)), 0, 1)
mm30 = MinMaxScaler().fit(X_pool)
X30_tr, X30_te = mm30.transform(X_pool[sel]), np.clip(mm30.transform(X_te), 0, 1)
ss = StandardScaler().fit(X_pool)
Xa_tr, Xa_te = ss.transform(X_pool[sel]), ss.transform(X_te)
y_tr = y_pool[sel].astype(float)


def acc(m, X):
    return f"{100 * np.mean(m.predict(X) == y_te):.1f}%"


titulo("FASE 1 -- um Ansatz, vários backends")
ans = Ansatz(QuantumConfig(8, "anel", 2))
p = ans.params_iniciais(0)
F_ex = BackendExato().features(X8_te[:5], ans, p)
F_mps = BackendMPS().features(X8_te[:5], ans, p)
print(f"8 qubits: diferença exato vs MPS = {np.abs(F_ex - F_mps).max():.1e}")
grande = Ansatz(QuantumConfig(60, "linear", 1), n_features=30)
t0 = time.perf_counter()
F60 = BackendMPS().features(X30_te[:10], grande, grande.params_iniciais(0))
print(f"60 qubits (vetor de estado teria 2^60 amplitudes = 18 exabytes): "
      f"10 amostras em {time.perf_counter() - t0:.2f}s com MPS, features {F60.shape}")
for cfg in (QuantumConfig(12, "anel", 2), QuantumConfig(40, "linear", 1), QuantumConfig(40, "anel", 2)):
    print(f"roteador: {cfg.n_qubits} qubits, {cfg.topologia_nome}/{cfg.n_camadas} -> {escolher_backend(cfg)[1]}")
tarefa = BackendMPS().submeter(X30_te, grande, grande.params_iniciais(0))
print(f"tarefa assíncrona {tarefa.id}: {tarefa.status()} ... ", end="")
print(f"{tarefa.resultado().shape} -> {tarefa.status()}")

titulo("FASE 2 -- custo do treino: parameter-shift vs SPSA (8 qubits, 60 amostras, 20 épocas)")
for L in (1, 4):
    a = Ansatz(QuantumConfig(8, "anel", 2), n_camadas=L)
    ps, sp = estimar_circuitos(a, 60, 20, "parameter_shift"), estimar_circuitos(a, 60, 20, "spsa")
    print(f"{L} camada(s), {a.n_params:3d} parâmetros: parameter-shift {ps:>9,} circuitos | "
          f"SPSA {sp:>6,} circuitos ({ps / sp:.0f}x menos)")
for opt in ("parameter_shift", "spsa"):
    t0 = time.perf_counter()
    m = VQCClassifier(QuantumConfig(8, "anel", 2), epocas=20, seed=0, otimizador=opt).fit(X8_tr, y_tr)
    print(f"VQC {opt:15s}: teste {acc(m, X8_te)}  {m.circuitos_executados:>7,} circuitos  "
          f"{time.perf_counter() - t0:5.1f}s")
print("termômetro de barren plateau (variância do gradiente, 4 camadas de dados):")
for n in (4, 6, 8, 10):
    v = variancia_gradiente(Ansatz(QuantumConfig(n, "anel", 2), n_camadas=4), n_amostras=40)
    print(f"   {n:2d} qubits: {v:.5f}")

titulo("FASE 3 -- mais dados por qubit (as 30 features originais, sem PCA)")
t0 = time.perf_counter()
m = VQCClassifier(QuantumConfig(8, "anel", 1), epocas=10, seed=0).fit(X30_tr, y_tr)
print(f"re-uploading: 30 features em 8 qubits ({m.ansatz.n_camadas} camadas): teste {acc(m, X30_te)} "
      f"({time.perf_counter() - t0:.1f}s)")
t0 = time.perf_counter()
m = VQCClassifier(QuantumConfig(5, "linear", 1), codificacao="amplitude", n_camadas=3, epocas=10,
                  seed=0).fit(Xa_tr, y_tr)
print(f"amplitude: 30 features em 5 qubits: teste {acc(m, Xa_te)} ({time.perf_counter() - t0:.1f}s)")

titulo("FASE 4 -- orquestrador clássico vs quântico (60 amostras de treino)")
cfg = QuantumConfig(8, "anel", 2)
orq = Orquestrador(candidatos_padrao(cfg, incluir_vqc=True), repeticoes=3).fit(X8_tr, y_tr)
print(orq.texto())
print(f"teste do escolhido: {acc(orq, X8_te)}")
