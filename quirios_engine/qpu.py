"""
Execução numa QPU com COMPROVANTE.

O QURIOS roda num processador quântico e guarda um comprovante verificável:
ids dos jobs (conferíveis no painel da IBM pelo dono da conta), backend,
horários, tempo de QPU usado, hash SHA-256 das contagens brutas e os
circuitos em OpenQASM 3 -- tudo assinado com ML-DSA (pós-quântico).

    python -m quirios_engine.qpu evolucao --modo ruidoso         # ensaio local, ruído do ibm_kingston
    python -m quirios_engine.qpu evolucao --modo ibm             # QPU real (conta IBM salva)
    python -m quirios_engine.qpu dados    --modo ibm             # pipeline 100% quântico
    python -m quirios_engine.qpu memoria  --modo ibm             # aprender com memória quântica
    python -m quirios_engine.qpu verificar comprovante.json --chave-publica chave.pub.pem

evolucao  a População da evolução quântica (evolucao_quantica.py) é SORTEADA
          na QPU: cada geração = 1 job em que o circuito gerador é medido.
dados     pipeline 100% quântico: os estados de Ising são PREPARADOS na QPU
          (codificação "ising", cronograma de 6 passos) e lá mesmo medidos --
          (1) direto nas bases Z e X, como faria um laboratório (modelo
          clássico), e (2) depois do circuito do modelo quântico (kernel
          projetado). Mesmo orçamento total de medições para os dois. 2 jobs.
memoria   os estados rho_P = (I + eps P)/2^n são preparados na QPU e a tarefa é
          dizer se são simétricos por reversão temporal: com memória quântica
          (2 cópias medidas juntas na base de Bell) contra sem memória (cada
          cópia medida sozinha), com o mesmo número de cópias. 2 jobs.

Conta IBM (uma vez, no seu terminal -- a chave nunca passa pelo QURIOS):
    python -c "from qiskit_ibm_runtime import QiskitRuntimeService as S; S.save_account(channel='ibm_quantum_platform', token='SUA_API_KEY', set_as_default=True)"
O plano gratuito (Open Plan) dá 10 minutos de QPU a cada 28 dias; cada job
destes usa segundos, mas espera a fila (de minutos a horas por job).
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


def _backend(modo, dispositivo="ibm_kingston", shots=1000, seed=7):
    from .backends import BackendQiskit
    if modo in ("ibm", "iqm"):
        return BackendQiskit(modo, shots=shots, dispositivo=dispositivo)
    return BackendQiskit("ruidoso", shots=shots, dispositivo=dispositivo, seed=seed)


def _qasm(circuito):
    try:
        from qiskit import qasm3
        return qasm3.dumps(circuito)
    except Exception as e:                     # pragma: no cover
        return f"// OpenQASM indisponível: {e}"


def _selar(comprovante, caminho, assinador=None):
    """Hash do conteúdo + gravação + assinatura ML-DSA opcional."""
    corpo = json.dumps(comprovante, sort_keys=True, ensure_ascii=False, default=float)
    comprovante = dict(comprovante, sha256_conteudo=hashlib.sha256(corpo.encode()).hexdigest())
    p = Path(caminho)
    p.write_text(json.dumps(comprovante, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    if assinador is not None:
        assinador.assinar_arquivo(p)
    return p


def verificar(caminho, verificador=None):
    """Confere o hash do conteúdo e (se houver chave pública) a assinatura ML-DSA."""
    d = json.loads(Path(caminho).read_text(encoding="utf-8"))
    sha = d.pop("sha256_conteudo", None)
    corpo = json.dumps(d, sort_keys=True, ensure_ascii=False, default=float)
    ok_hash = sha == hashlib.sha256(corpo.encode()).hexdigest()
    ok_sig = verificador.verificar_arquivo(caminho) if verificador is not None else None
    return {"hash_ok": ok_hash, "assinatura_ok": ok_sig, "jobs": [j.get("job_id") for j in d.get("jobs", [])],
            "backend": d.get("backend"), "modo": d.get("modo")}


def _base(tipo, backend):
    from . import __version__
    return {"formato": "quirios.comprovante_qpu", "tipo": tipo, "versao_quirios": __version__,
            "modo": backend.modo, "backend": getattr(backend._get_backend(), "name", backend.nome),
            "data": datetime.datetime.now().isoformat(timespec="seconds")}


# ── evolução com a população sorteada na QPU ────────────────────────────────

def evolucao_na_qpu(X, y, backend, n_qubits=None, populacao=24, geracoes=3, seed=0, verbose=False):
    from .evolucao_quantica import EvolucaoQuantica
    X, y = np.asarray(X, float), np.asarray(y, float)
    eq = EvolucaoQuantica(n_qubits or X.shape[1], populacao=populacao, geracoes=geracoes, seed=seed,
                          backend_gerador=backend, dobras=4, repeticoes=2, verbose=verbose)
    res = eq.executar(X, y)
    m = res["melhor"]
    return dict(_base("evolucao", backend),
                resumo=eq.resumo(res), jobs=list(backend.registro_jobs),
                geracoes=[{k: h[k] for k in ("geracao", "melhor_acuracia", "entropia_bits", "genomas_distintos",
                                            "ligacao_max_bits")} for h in res["historico"]],
                melhor={"descricao": m["descricao"], "acuracia_cv": m["acuracia"], "portas_2q": m["custo"]},
                padrao={"acuracia_cv": res["padrao"]["acuracia"]},
                ligacoes=[{"genes": list(l["genes"]), "bits": l["informacao_mutua"]}
                          for l in res["ligacoes_aprendidas"][:3]],
                circuito_gerador_qasm3=_qasm(eq.gerador.circuito_qiskit(eq.params)),
                n_qubits_gerador=res["n_qubits_gerador"])


# ── pipeline 100% quântico: dados preparados e classificados na QPU ─────────

def _features_classicas(cont_z, cont_x, n):
    """[<Z_i Z_i+1>] + [<X_i>] + <|M|> a partir das contagens (qubit 0 à direita)."""
    def matriz(cont):
        bits = np.array([[int(s.replace(" ", "")[n - 1 - q]) for q in range(n)] for s in cont])
        return 1 - 2 * bits, np.array(list(cont.values()), float)
    z, wz = matriz(cont_z)
    x, wx = matriz(cont_x)
    zz = [float(np.sum(wz * z[:, i] * z[:, (i + 1) % n]) / wz.sum()) for i in range(n)]
    mx = [float(np.sum(wx * x[:, q]) / wx.sum()) for q in range(n)]
    mag = float(np.sum(wz * np.abs(z.sum(1))) / wz.sum() / n)
    return np.array(zz + mx + [mag])


def _cv(F, y, ajustar_prever, dobras=4, seed=0):
    from .orquestrador import divisoes_cv
    accs = []
    for tr, te in divisoes_cv(y, dobras, 2, seed)[0]:
        accs.append(float(np.mean(ajustar_prever(F[tr], y[tr], F[te]) == y[te])))
    return float(np.mean(accs)), float(np.std(accs))


def dados_na_qpu(backend, n_spins=6, n_sistemas=24, shots=60, h_min=0.85, h_max=1.2, seed=0):
    """Prepara os estados de Ising na QPU e compara, com o mesmo orçamento de
    medições, um modelo clássico (medidas diretas) e o kernel quântico projetado."""
    from .dados_quanticos import ansatz_preparo, estado_fundamental, estado_preparado
    from .kernels import ClassificadorKernel, KernelQuantico
    from .sistema import _ajustar
    from .vqc import _sig
    from .core import QuantumConfig
    r = np.random.default_rng(seed)
    hams, ys, fids = [], [], []
    for _ in range(n_sistemas):
        J = 1 + 0.3 * r.uniform(-1, 1, n_spins)
        h = r.uniform(h_min, h_max)
        hams.append(np.concatenate([J, [h]]))
        ys.append(float(h > np.exp(np.mean(np.log(J)))))
        fids.append(float(abs(np.vdot(estado_fundamental(J, h), estado_preparado(J, h))) ** 2))
    H, y = np.array(hams), np.array(ys)

    # (1) laboratório clássico: prepara na QPU e mede direto nas bases Z e X
    prep = ansatz_preparo(n_spins)
    p0 = np.zeros(prep.n_params)
    circs = []
    for x in H:
        c = prep.circuito_qiskit(x, p0, medir=False)
        cz = c.copy(); cz.measure_all()
        cx = c.copy(); cx.h(range(n_spins)); cx.measure_all()
        circs += [cz, cx]
    cont = backend.amostrar_lote(circs, shots, tipo="dados-classicos")
    Fc = np.array([_features_classicas(cont[2 * i], cont[2 * i + 1], n_spins) for i in range(n_sistemas)])

    # (2) modelo quântico: prepara na QPU + circuito do modelo + leitura X/Y/Z
    shots_q = max(1, round(2 * shots / 3))                        # mesmo total: 2 x shots
    kern = KernelQuantico(QuantumConfig(n_spins, "anel", 1), backend=backend, codificacao="ising", seed=seed)
    kern.preparar(H.shape[1])
    s_antes, backend.shots = backend.shots, shots_q
    Fq = kern.representar(H)
    backend.shots = s_antes
    Fq_exato = KernelQuantico(QuantumConfig(n_spins, "anel", 1), codificacao="ising", seed=seed).representar(H)

    def logistica(Ftr, ytr, Fte):
        mu, sd, w, b = _ajustar(Ftr, ytr, 1.0)
        return (_sig(((Fte - mu) / sd) @ w + b) > 0.5).astype(float)

    class _Fixo(KernelQuantico):                               # kernel sobre features já medidas
        def __init__(self, F):
            super().__init__(QuantumConfig(2, "anel", 0))
            self.F = F

        def representar(self, X, params=None):
            return self.F[np.asarray(X, int).ravel()]

    def kernel_em(F):
        def f(Itr, ytr, Ite):
            m = ClassificadorKernel(_Fixo(F)).fit(Itr, ytr)
            return m.predict(Ite).astype(float)
        return f

    idx = np.arange(n_sistemas, dtype=float)[:, None]
    acc_c = _cv(Fc, y, logistica)
    acc_q = _cv(idx, y, kernel_em(Fq))
    acc_teto = _cv(idx, y, kernel_em(Fq_exato))
    erro_hw = float(np.abs(Fq - Fq_exato).mean())
    return dict(_base("dados", backend),
                jobs=list(backend.registro_jobs), n_spins=n_spins, n_sistemas=n_sistemas,
                medicoes_por_sistema={"classico": 2 * shots, "quantico": 3 * shots_q},
                fidelidade_preparo_media=float(np.mean(fids)),
                acuracia_cv={"classico_medidas_na_qpu": acc_c, "kernel_quantico_na_qpu": acc_q,
                             "kernel_quantico_teto_sem_ruido_infinitos_shots": acc_teto},
                erro_medio_features_qpu_vs_ideal=erro_hw,
                circuito_preparo_exemplo_qasm3=_qasm(prep.circuito_qiskit(H[0], p0, medir=False)))


# ── aprender com memória quântica: o mesmo número de cópias, na QPU ─────────

GRADE_EPS = tuple(round(float(e), 2) for e in np.arange(0.05, 0.951, 0.05))


def mcnemar(acertos_a, acertos_b):
    """Teste exato de McNemar (bilateral) para duas decisões sobre os MESMOS casos:
    só os casos em que elas discordam contam. Devolve (só a acertou, só b acertou, p)."""
    import math
    a = np.asarray(acertos_a, bool)
    b = np.asarray(acertos_b, bool)
    so_a, so_b = int(np.sum(a & ~b)), int(np.sum(~a & b))
    k, t = min(so_a, so_b), so_a + so_b
    p = 1.0 if t == 0 else min(1.0, 2 * sum(math.comb(t, i) for i in range(k + 1)) / 2 ** t)
    return so_a, so_b, p


def _medidas_por_estado(inst, cont_q, cont_c, n, m, copias):
    """Para cada estado: os Paulis de Bell (com memória) e os bits de uma cópia (sem memória)."""
    from .memoria_quantica import pauli_de_bell
    out = []
    for i in range(len(inst)):
        Q = np.array([pauli_de_bell(next(iter(c)), n) for c in cont_q[i * m:(i + 1) * m]])
        bits = np.array([int(next(iter(c)).replace(" ", ""), 2) for c in cont_c[i * copias:(i + 1) * copias]])
        out.append((Q, bits))
    return out


def _calibrar_eps(inst, medidas, bases, n, grade=GRADE_EPS):
    """O sinal que sobra depois do ruído do chip, medido em estados de CALIBRAÇÃO (rótulo
    conhecido, fora do teste): para cada lado, o eps da grade que dá a maior verossimilhança
    ao rótulo certo. Os dois lados são calibrados do mesmo jeito e com os mesmos estados."""
    from .memoria_quantica import decidir_com_memoria, decidir_sem_memoria

    def melhor(decidir):
        notas = []
        for e in grade:
            ll = 0.0
            for i, (_, _, s) in enumerate(inst):
                p = min(max(decidir(i, e), 1e-6), 1 - 1e-6)
                ll += np.log(p if s > 0 else 1 - p)
            notas.append(ll)
        return float(grade[int(np.argmax(notas))])

    eq = melhor(lambda i, e: decidir_com_memoria(medidas[i][0], n, e))
    ec = melhor(lambda i, e: decidir_sem_memoria(bases[i][0], bases[i][1], medidas[i][1], n, e))
    return eq, ec


def memoria_na_qpu(backend, n_qubits=4, instancias=12, copias=32, eps=0.9, eps_decodificacao=None, seed=0,
                   calibracao=0, preparo="cnot"):
    """Os estados rho_P = (I + eps P)/2^n são preparados na QPU (cada disparo
    é uma cópia nova da fonte). Com memória: c/2 experimentos de 2 cópias
    medidas juntas na base de Bell. Sem memória: c cópias medidas uma a uma em
    bases de Pauli aleatórias. Mesmo número de cópias; 2 jobs.

    calibracao=K: K estados extras, no mesmo job (mesmo chip, mesma calibração da
    IBM), servem só para medir quanto sinal o ruído deixa, separadamente para cada
    lado; a acurácia é medida só nos `instancias` estados de teste, que não entram na
    calibração. Sem ela, os dois lados usam eps_decodificacao (0,6 fora do ideal).

    preparo="produto": as cópias saem sem portas de 2 qubits (mesmo estado médio; ver
    memoria_quantica._preparar_produto). Os dois lados recebem cópias feitas do mesmo jeito."""
    from .memoria_quantica import (_wilson, bases_aleatorias, circuito_com_memoria, circuito_sem_memoria,
                                   decidir_com_memoria, decidir_sem_memoria, sortear_instancias)
    rng = np.random.default_rng(seed)
    n, m = n_qubits, max(1, copias // 2)
    eps_dec = eps_decodificacao or (eps if backend.modo == "ideal" else 0.6)   # o ruído da QPU apaga parte do sinal
    cal = sortear_instancias(n, calibracao, rng) if calibracao else []
    teste = sortear_instancias(n, instancias, rng)
    inst = cal + teste                                              # calibração primeiro, no mesmo job
    circ_q, circ_c, bases = [], [], []
    for x, z, _ in inst:
        circ_q += [circuito_com_memoria(n, x, z, eps, rng, preparo) for _ in range(m)]
        xb, zb = bases_aleatorias(copias, n, rng)
        bases.append((xb, zb))
        circ_c += [circuito_sem_memoria(n, x, z, eps, a, b, rng, preparo) for a, b in zip(xb, zb)]
    cont_q = backend.amostrar_lote(circ_q, 1, tipo="memoria-bell")
    cont_c = backend.amostrar_lote(circ_c, 1, tipo="memoria-uma-copia")
    medidas = _medidas_por_estado(inst, cont_q, cont_c, n, m, copias)
    K = len(cal)
    extra = {}
    if K:
        eps_q, eps_c = _calibrar_eps(cal, medidas[:K], bases[:K], n)
        extra["calibracao"] = {"estados": K, "eps_com_memoria": eps_q, "eps_sem_memoria": eps_c,
                               "grade": list(GRADE_EPS),
                               "metodo": "máxima verossimilhança do rótulo certo nos estados de calibração"}
    else:
        eps_q = eps_c = eps_dec
    certo_q, certo_c, por_estado = [], [], []
    for i, (x, z, s) in enumerate(teste):
        Q, bits = medidas[K + i]
        pq = decidir_com_memoria(Q, n, eps_q)
        pc = decidir_sem_memoria(bases[K + i][0], bases[K + i][1], bits, n, eps_c)
        certo_q.append(bool((pq > 0.5) == (s > 0)))
        certo_c.append(bool((pc > 0.5) == (s > 0)))
        por_estado.append({"simetrico": s > 0, "p_simetrico_com_memoria": pq, "p_simetrico_sem_memoria": pc})
    ac_q, ac_c = sum(certo_q), sum(certo_c)
    so_q, so_c, p = mcnemar(certo_q, certo_c)
    return dict(_base("memoria", backend),
                jobs=list(backend.registro_jobs), n_qubits=n, instancias=instancias, eps=eps, preparo=preparo,
                eps_decodificacao={"com_memoria": eps_q, "sem_memoria": eps_c} if K else eps_dec,
                copias_por_estado={"com_memoria": 2 * m, "sem_memoria": copias}, **extra,
                acuracia={"com_memoria_quantica": ac_q / instancias, "sem_memoria": ac_c / instancias,
                          "ic95_com_memoria": _wilson(ac_q, instancias), "ic95_sem_memoria": _wilson(ac_c, instancias),
                          "mcnemar": {"so_com_memoria_acertou": so_q, "so_sem_memoria_acertou": so_c, "p_valor": p}},
                por_estado=por_estado,
                circuito_com_memoria_exemplo_qasm3=_qasm(circ_q[0]),
                circuito_sem_memoria_exemplo_qasm3=_qasm(circ_c[0]))


def _parser():
    ap = argparse.ArgumentParser(prog="python -m quirios_engine.qpu",
                                 description="Executa o QURIOS numa QPU e guarda um comprovante assinado.")
    sub = ap.add_subparsers(dest="acao", required=True)
    for nome in ("evolucao", "dados", "memoria"):
        p = sub.add_parser(nome)
        p.add_argument("--modo", choices=["ruidoso", "ibm", "iqm"], default="ruidoso")
        p.add_argument("--dispositivo", default="ibm_kingston",
                       help="ibm_kingston (IBM), garnet / garnet:mock (IQM), iqm_garnet (ensaio com o ruído da IQM)")
        p.add_argument("--saida", default=None)
        p.add_argument("--chave-privada", default=None, help="assina o comprovante com ML-DSA")
        p.add_argument("--sim", action="store_true", help="não pedir confirmação antes de usar a QPU")
        p.add_argument("--seed", type=int, default=0)
    pe = sub.choices["evolucao"]
    pe.add_argument("--dataset", default="breast_cancer")
    pe.add_argument("--n-qubits", type=int, default=6)
    pe.add_argument("--n-treino", type=int, default=80)
    pe.add_argument("--populacao", type=int, default=24)
    pe.add_argument("--geracoes", type=int, default=3)
    pd_ = sub.choices["dados"]
    pd_.add_argument("--n-spins", type=int, default=6)
    pd_.add_argument("--sistemas", type=int, default=24)
    pd_.add_argument("--shots", type=int, default=60)
    pm = sub.choices["memoria"]
    pm.add_argument("--n-qubits", type=int, default=4)
    pm.add_argument("--instancias", type=int, default=12)
    pm.add_argument("--copias", type=int, default=32)
    pm.add_argument("--eps-decodificacao", type=float, default=None)
    pm.add_argument("--preparo", choices=["cnot", "produto"], default="cnot",
                    help="produto: cópias sem portas de 2 qubits (mesmo estado, menos ruído)")
    pm.add_argument("--calibracao", type=int, default=0,
                    help="estados extras, fora do teste, para medir o ruído real do chip (mesmo job)")
    pv = sub.add_parser("verificar")
    pv.add_argument("arquivo")
    pv.add_argument("--chave-publica", default=None)
    pp = sub.add_parser("prova", help="os três experimentos de uma vez, com uma confirmação só")
    pp.add_argument("--modo", choices=["ibm", "iqm", "ruidoso"], default="ibm",
                    help="ibm: QPU real da IBM; iqm: QPU real da IQM; ruidoso: ensaio local")
    pp.add_argument("--dispositivo", default=None,
                    help="IBM: a QPU de menor fila; IQM: garnet (garnet:mock testa sem gastar créditos); "
                         "ensaio: ibm_kingston ou iqm_garnet")
    pp.add_argument("--pasta", default=str(Path("resultados") / "qpu_real"))
    pp.add_argument("--chave-privada", default=None, help="padrão: segredos/quirios_mldsa65.pem (criada se faltar)")
    pp.add_argument("--sim", action="store_true", help="não pedir confirmação antes de usar a QPU")
    pp.add_argument("--pequena", action="store_true", help="versões menores (ensaio rápido)")
    return ap


def _estimativa(a):
    if a.acao == "evolucao":
        return f"{a.geracoes + 1} jobs de 1 circuito ({a.populacao} shots) -- a população de cada geração"
    if a.acao == "memoria":
        k = a.instancias + a.calibracao
        return (f"2 jobs de 1 shot por circuito: {k * (a.copias // 2)} circuitos de "
                f"{2 * a.n_qubits} qubits + {k * a.copias} circuitos de {a.n_qubits} qubits"
                + (f" ({a.calibracao} estados de calibração + {a.instancias} de teste)" if a.calibracao else ""))
    return f"2 jobs: {2 * a.sistemas} circuitos x {a.shots} shots + {3 * a.sistemas} circuitos"


def _executar(a, assinador=None):
    """Roda um experimento (já confirmado) e grava o comprovante. Devolve o caminho."""
    if a.acao == "evolucao":
        from sklearn.model_selection import train_test_split
        from .benchmark import _preparar, carregar_dataset
        X, y = carregar_dataset(a.dataset)
        Xp, Xt, yp, _ = train_test_split(X, y, test_size=min(120, int(0.3 * len(y))), stratify=y, random_state=42)
        nq = min(a.n_qubits, X.shape[1])
        Xp, _ = _preparar(Xp, Xt, nq)
        sel = np.random.default_rng(a.seed).choice(len(yp), min(a.n_treino, len(yp)), replace=False)
        b = _backend(a.modo, a.dispositivo, shots=a.populacao)
        comp = evolucao_na_qpu(Xp[sel], yp[sel], b, nq, a.populacao, a.geracoes, a.seed, verbose=True)
        comp["dataset"] = a.dataset
        print(comp["resumo"])
    elif a.acao == "memoria":
        b = _backend(a.modo, a.dispositivo, shots=1)
        comp = memoria_na_qpu(b, a.n_qubits, a.instancias, a.copias, eps_decodificacao=a.eps_decodificacao,
                              seed=a.seed, calibracao=a.calibracao, preparo=a.preparo)
        print(json.dumps(comp["acuracia"], indent=2, ensure_ascii=False))
    else:
        b = _backend(a.modo, a.dispositivo, shots=a.shots)
        comp = dados_na_qpu(b, a.n_spins, a.sistemas, a.shots, seed=a.seed)
        print(json.dumps(comp["acuracia_cv"], indent=2, ensure_ascii=False))
    saida = a.saida or f"comprovante_qpu_{a.acao}_{a.modo}_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
    p = _selar(comp, saida, assinador)
    print(f"\njobs: {[j['job_id'] for j in comp['jobs']]}\ncomprovante: {p}"
          + (f" (assinado: {p}.sig)" if assinador else ""))
    return p


def _destaque(d):
    """Uma linha com o resultado principal de um comprovante."""
    if d.get("tipo") == "memoria":
        a = d["acuracia"]
        p = (a.get("mcnemar") or {}).get("p_valor")
        return (f"com memória quântica {a['com_memoria_quantica']:.0%} · sem memória {a['sem_memoria']:.0%}"
                + (f" (McNemar p = {p:.3f})" if p is not None else ""))
    if d.get("tipo") == "dados":
        a = d["acuracia_cv"]
        media = lambda v: v[0] if isinstance(v, (list, tuple)) else v          # _cv grava [média, desvio]
        return (f"kernel quântico {media(a['kernel_quantico_na_qpu']):.0%} · "
                f"clássico {media(a['classico_medidas_na_qpu']):.0%}")
    if d.get("tipo") == "evolucao":
        return f"campeão {d['melhor']['acuracia_cv']:.0%} · circuito padrão {d['padrao']['acuracia_cv']:.0%}"
    return ""


COMPLETA = {"memoria": ["--instancias", "40", "--copias", "64", "--calibracao", "20", "--preparo", "produto"],
            "dados": [], "evolucao": []}
PEQUENA = {"memoria": ["--instancias", "4", "--copias", "8", "--calibracao", "4"],
           "dados": ["--sistemas", "8", "--shots", "20"],
           "evolucao": ["--n-qubits", "4", "--n-treino", "24", "--populacao", "6", "--geracoes", "1"]}


def prova(modo="ibm", dispositivo=None, pasta=None, chave_privada=None, sim=False, pequena=False,
          perguntar=input):
    """A prova completa numa QPU, com uma confirmação só: memória quântica (a manchete),
    dados preparados na QPU e evolução sorteada na QPU. Escolhe a QPU de menor fila,
    assina cada comprovante com ML-DSA, confere tudo e grava um resumo assinado. Um
    experimento que falhar não apaga os outros. A chave da IBM nunca passa por aqui: a
    conta é a que você salvou com `python -m quirios_engine.conta_ibm`."""
    from .seguranca import Assinador, Verificador, gerar_chaves
    pasta = Path(pasta or Path("resultados") / "qpu_real")
    if modo == "ibm":
        from .conta_ibm import contas_salvas, processadores
        try:
            if not contas_salvas():
                raise RuntimeError("nenhuma conta salva")
            qpus = processadores()
        except Exception as e:
            raise SystemExit("Conta IBM não encontrada ou sem acesso. No seu terminal, rode:\n"
                             "    python -m quirios_engine.conta_ibm\n"
                             f"(a chave é digitada só lá e nunca aparece na tela). Detalhe: {type(e).__name__}") from e
        if dispositivo is None:
            aptas = [q for q in qpus if q["qubits"] >= 8 and q["fila"] is not None]
            if not aptas:
                raise SystemExit("nenhuma QPU disponível para a sua conta agora; tente mais tarde")
            dispositivo = aptas[0]["nome"]
            print(f"QPU escolhida: {dispositivo} (fila: {aptas[0]['fila']} jobs)")
    elif modo == "iqm":
        from .conta_iqm import DISPOSITIVO_PADRAO, carregar_token
        try:
            ok = carregar_token()
            import iqm.qiskit_iqm  # noqa: F401
        except ImportError as e:
            raise SystemExit('A IQM precisa do ambiente próprio dela (pip install "iqm-client[qiskit]"; '
                             'ver README.pt-BR.md, "Hardware quântico real").') from e
        if not ok:
            raise SystemExit("Token da IQM não encontrado. No seu terminal, no ambiente da IQM, rode:\n"
                             "    python -m quirios_engine.conta_iqm\n"
                             "(o token é digitado só lá e nunca aparece na tela).")
        dispositivo = dispositivo or DISPOSITIVO_PADRAO
        print(f"Computador quântico da IQM: {dispositivo}"
              + (" (simulado na nuvem da IQM: não gasta créditos)" if dispositivo.endswith(":mock") else ""))
    dispositivo = dispositivo or "ibm_kingston"

    acoes = ("memoria", "dados", "evolucao")
    args = {acao: _parser().parse_args([acao, "--modo", modo, "--dispositivo", dispositivo, "--sim"]
                                       + (PEQUENA if pequena else COMPLETA)[acao]) for acao in acoes}
    onde = {"ibm": f"na QPU real {dispositivo} (IBM)", "iqm": f"na QPU real {dispositivo} (IQM)",
            "ruidoso": f"em ensaio (ruído do {dispositivo})"}[modo]
    print(f"\nProva {onde}:")
    for acao in acoes:
        print(f"  {acao:9s} {_estimativa(args[acao])}")
    if modo in ("ibm", "iqm"):
        print({"ibm": "O plano gratuito da IBM dá 10 minutos de QPU a cada 28 dias.",
               "iqm": "O plano Starter da IQM dá até 30 créditos por mês."}[modo]
              + " Cada comprovante registra o tempo exato usado. Cada job espera a fila (de minutos a horas).")
        if not sim and perguntar("Continuar? [s/N] ").strip().lower() != "s":
            raise SystemExit("Cancelado.")

    if chave_privada:
        priv = Path(chave_privada)
    else:
        priv = Path("segredos") / "quirios_mldsa65.pem"
        if not priv.exists():
            priv, _ = gerar_chaves(priv.parent)
            print(f"chave de assinatura criada em {priv} (guarde em segredo; nunca vai para o git)")
    assinador = Assinador(priv)
    pasta.mkdir(parents=True, exist_ok=True)
    pub = pasta / "chave_publica_mldsa65.pub.pem"                 # a pública vai junto: qualquer um confere
    from cryptography.hazmat.primitives import serialization as s
    chave = s.load_pem_private_key(priv.read_bytes(), password=None)
    pub.write_bytes(chave.public_key().public_bytes(s.Encoding.PEM, s.PublicFormat.SubjectPublicKeyInfo))
    verificador = Verificador(pub)

    carimbo = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
    linhas = []
    for acao in acoes:
        a = args[acao]
        a.saida = str(pasta / f"comprovante_qpu_{acao}_{modo}_{carimbo}.json")
        print(f"\n-- {acao} --")
        try:
            p = _executar(a, assinador)
            d = json.loads(p.read_text(encoding="utf-8"))
            v = verificar(p, verificador)
            usos = [j.get("uso_qpu_s") for j in d.get("jobs", []) if j.get("uso_qpu_s") is not None]
            linhas.append({"experimento": acao, "comprovante": p.name, "backend": d.get("backend"),
                           "modo": d.get("modo"), "resultado": _destaque(d), "jobs": v["jobs"],
                           "uso_qpu_s": round(sum(usos), 2) if usos else None,
                           "verificado": bool(v["hash_ok"] and v["assinatura_ok"])})
        except Exception as e:                    # um experimento que falha não apaga os outros
            linhas.append({"experimento": acao, "erro": f"{type(e).__name__}: {e}"})
    from . import __version__
    resumo = {"formato": "quirios.prova_qpu", "versao_quirios": __version__, "modo": modo,
              "dispositivo": dispositivo, "data": datetime.datetime.now().isoformat(timespec="seconds"),
              "chave_publica": pub.name, "experimentos": linhas}
    _selar(resumo, pasta / f"resumo_prova_{modo}_{carimbo}.json", assinador)

    print(f"\n{'=' * 72}\nProva {onde}:")
    for l in linhas:
        if "erro" in l:
            print(f"  {l['experimento']:9s} FALHOU: {l['erro']}")
        else:
            uso = (f"{l['uso_qpu_s']} s de QPU" if l["uso_qpu_s"] is not None
                   else "tempo de QPU não informado" if l.get("modo") in ("ibm", "iqm") else "sem QPU (ensaio)")
            ok = "ok" if l["verificado"] else "FALHOU"
            print(f"  {l['experimento']:9s} {l['resultado']}  ({uso}; assinatura {ok})")
    feitos = [str(pasta / l["comprovante"]) for l in linhas if "comprovante" in l]
    if feitos:
        print(f"\nTudo em {pasta}. Para o painel de investidores:\n"
              f"  python exemplos/gerar_dados_investidores.py --comprovante {' '.join(feitos)}")
    return resumo


def main(argv=None):
    a = _parser().parse_args(argv)

    if a.acao == "verificar":
        ver = None
        if a.chave_publica:
            from .seguranca import Verificador
            ver = Verificador(a.chave_publica)
        print(json.dumps(verificar(a.arquivo, ver), indent=2, ensure_ascii=False))
        return
    if a.acao == "prova":
        prova(a.modo, a.dispositivo, a.pasta, a.chave_privada, a.sim, a.pequena)
        return

    if a.modo in ("ibm", "iqm") and not a.sim:
        print(f"Vai usar a QPU {a.dispositivo}: {_estimativa(a)}. Cada job espera a fila da IBM.")
        if input("Continuar? [s/N] ").strip().lower() != "s":
            raise SystemExit("Cancelado.")
    assinador = None
    if a.chave_privada:
        from .seguranca import Assinador
        assinador = Assinador(a.chave_privada)
    _executar(a, assinador)


if __name__ == "__main__":
    main()
