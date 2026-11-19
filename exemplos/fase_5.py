"""
Demonstração da fase 5 do QURIOS: núcleo quântico auto-evolutivo.

    python exemplos/fase_5.py

1. Genoma: o mesmo problema com circuitos diferentes (IQP, RZZ, leitura Pauli).
2. Kernels: concentração do kernel de fidelidade vs projetado.
3. Auto-evolução: busca de arquitetura com promoção segura (registro temporário).
4. Hardware: mitigação de erro e curva de prontidão por geração.
5. Segurança: modelo assinado com ML-DSA (pós-quântico) e adulteração detectada.
"""
import tempfile
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from quirios_engine import (Ansatz, BackendExato, BackendQiskit, BuscaEvolutiva, ClassificadorKernel,
                            KernelQuantico, QuantumConfig, RegistroCampeoes, VQCClassifier,
                            concentracao, projetar_hardware)


def titulo(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


d = load_breast_cancer()
Xp, Xt, yp, yt = train_test_split(d.data, d.target, test_size=120, stratify=d.target, random_state=42)
pca = PCA(6, random_state=0).fit(Xp)
mm = MinMaxScaler().fit(pca.transform(Xp))
sel = np.random.default_rng(0).choice(len(yp), 80, replace=False)
Xtr, ytr = mm.transform(pca.transform(Xp[sel])), yp[sel].astype(float)
Xte = np.clip(mm.transform(pca.transform(Xt)), 0, 1)
cfg = QuantumConfig(6, "anel", 1)


def acc(m):
    return f"{100 * np.mean(m.predict(Xte) == yt):.1f}%"


titulo("1. GENOMA -- o mesmo VQC (SPSA, 20 épocas) com circuitos diferentes")
for nome, genes in [("padrão (ângulo/CNOT/Z)", {}),
                    ("IQP + RZZ treinável", {"codificacao": "iqp", "emaranhador": "rzz"}),
                    ("SU(2) + CZ + leitura Pauli", {"rotacoes": ("rz", "ry", "rz"), "emaranhador": "cz",
                                                    "leitura": "pauli"})]:
    m = VQCClassifier(cfg, otimizador="spsa", epocas=20, seed=0, **genes).fit(Xtr, ytr)
    print(f"  {nome:28s} teste {acc(m)}  ({m.ansatz.n_params} parâmetros, "
          f"{m.ansatz.portas_2q()} portas de 2 qubits)")

titulo("2. KERNELS -- concentração exponencial (variância fora da diagonal)")
r = np.random.default_rng(3)
for n in (4, 8, 12):
    X = r.uniform(0, 1, (20, n))
    c = QuantumConfig(n, "anel", 2)
    vf = concentracao(KernelQuantico(c, tipo="fidelidade", n_camadas=3).matriz(X))["variancia"]
    vp = concentracao(KernelQuantico(c, tipo="projetado", n_camadas=3).matriz(X))["variancia"]
    print(f"  {n:2d} qubits: fidelidade {vf:.1e}  projetado {vp:.1e}")
k = ClassificadorKernel(KernelQuantico(QuantumConfig(6, "anel", 0))).fit(Xtr, ytr)
print(f"  kernel projetado auto-calibrado (lambda={k.lam_}): teste {acc(k)}, "
      f"leave-one-out {100 * k.acuracia_loo_:.1f}%")

titulo("3. AUTO-EVOLUÇÃO -- 2 rodadas, promoção só com evidência")
with tempfile.TemporaryDirectory() as tmp:
    reg = RegistroCampeoes(Path(tmp) / "campeoes.json")
    for rodada in range(2):
        res = BuscaEvolutiva(6, populacao=8, geracoes=3, seed=rodada, registro=reg,
                             chave="breast_cancer:6q").executar(Xtr, ytr)
        m = res["melhor"]
        print(f"  rodada {rodada + 1}: melhor {100 * m['acuracia']:.1f}% ({m['descricao']}, "
              f"{m['custo']} portas 2q) -> {'PROMOVIDO' if res['promovido'] else 'mantido'}: {res['motivo']}")
    camp = reg.campeao("breast_cancer:6q")
    if camp["simulavel_classicamente"]:
        print("  aviso: o campeão não usa emaranhamento -- é simulável classicamente "
              "(use min_portas_2q para exigir circuitos quânticos)")

titulo("4. HARDWARE -- mitigação no hardware de hoje e curva de prontidão")
a = Ansatz(cfg, n_features=6)
p = a.params_iniciais(0)
ex = BackendExato().features(Xtr[:40], a, p)
for mit in [(), ("leitura",)]:
    F = BackendQiskit("parametrico", erros="hoje", shots=8000, seed=1, mitigacao=mit).features(Xtr[:40], a, p)
    print(f"  {'+'.join(mit) or 'sem mitigação':14s} erro médio {np.abs(F - ex).mean():.4f}")
for linha in projetar_hardware(Xtr[:40], ytr[:40], a, p, geracoes=["hoje", "dez_vezes"], shots=4000):
    print(f"  {linha['geracao']:12s} erro {linha['erro_features']:.4f}  acurácia CV {100 * linha['acuracia_cv']:.1f}%")

titulo("5. SEGURANÇA PÓS-QUÂNTICA -- ML-DSA-65 (NIST FIPS 204)")
try:
    from quirios_engine.seguranca import Assinador, Verificador, gerar_chaves
    with tempfile.TemporaryDirectory() as tmp:
        priv, pub = gerar_chaves(tmp)
        modelo = VQCClassifier(cfg, epocas=2, seed=0).fit(Xtr, ytr)
        arq = Path(tmp) / "modelo.json"
        modelo.salvar(arq, assinador=Assinador(priv))
        VQCClassifier.carregar(arq, verificador=Verificador(pub))
        print("  modelo assinado e verificado")
        arq.write_text(arq.read_text(encoding="utf-8").replace('"b": ', '"b": 1+'), encoding="utf-8")
        try:
            VQCClassifier.carregar(arq, verificador=Verificador(pub))
        except ValueError as e:
            print(f"  adulteração detectada: {e}")
except ImportError:
    print('  (instale o extra: pip install -e ".[pqc]")')
