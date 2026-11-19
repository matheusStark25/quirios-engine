"""
Backends de execução do QURIOS -- todos seguem o mesmo contrato (Backend):

    features(X, ansatz, params)  -> matriz (amostras x ansatz.n_saida), bloqueante
    submeter(X, ansatz, params)  -> Tarefa (não bloqueia; ver tarefas.py)

    As features são os valores esperados dos observáveis da leitura do
    Ansatz (ansatz.observaveis()); na leitura padrão "z_zz":
    [P(qubit q = 1) para cada q] + [<Z_i Z_i+1> no anel].

Backends disponíveis:

    BackendExato   vetor de estado em numpy/numba. Exato, sem shots.
                   Custo 2^n: até MAX_QUBITS_VETORIAL (24) qubits.
    BackendMPS     tensor network (matrix product state) do Qiskit Aer.
                   Exato (sem shots) e o custo cresce com o ENTRELAÇAMENTO,
                   não com 2^n: linear, anel de 1 camada e blocos rodam 100
                   qubits em ~0,1 s/amostra. Anel com 2+ camadas entrelaça
                   demais (20 qubits: ~1000 s/amostra) -- use max_bond.
    BackendQiskit  circuito Qiskit com shots:
                     modo="ideal"       Aer, só ruído estatístico
                     modo="ruidoso"     Aer com o ruído calibrado de um processador IBM
                     modo="parametrico" Aer com taxas de erro escolhidas (hardware futuro,
                                        ver futuro.py)
                     modo="ibm"         hardware quântico real (conta e token)
                   Medição:
                     medicao="bases"    agrupa os observáveis em bases X, Y, Z globais
                     medicao="sombras"  sombras clássicas (Huang, Kueng & Preskill 2020):
                                        bases aleatórias por qubit; estima qualquer Pauli
                                        local de uma vez (leitura "pauli2")
                   Mitigação de erro (mitigacao=("leitura", "zne")):
                     "leitura"  calibra a matriz de confusão de cada qubit e inverte
                                (correlações de até 2 qubits, erros independentes)
                     "zne"      extrapolação para ruído zero: dobra cada porta de 2
                                qubits (G -> G G^-1 G) nas escalas 1, 3, 5... e ajusta
                                um polinômio até escala 0
                   Job assíncrono nativo; job IBM recuperável por id.
                   portas_fracionarias=True pede RZZ/RX nativos aos processadores
                   Heron (recurso experimental da IBM).

Backends Aer aceitam device="GPU" (exige qiskit-aer-gpu, só Linux + CUDA).

Backend novo (outro provedor ou simulador): herde de Backend e implemente
_calcular(X, ansatz, params). Para que criar_backend("nome") o encontre,
publique-o no grupo de entry points "quirios.backends" do seu pacote.
"""
import numpy as np

from .ansatz import Ansatz, pares_zz
from .core import MAX_QUBITS_VETORIAL
from .tarefas import TarefaLocal, TarefaQiskit, executor


def features_de_probs(pr, n, pares):
    """Compatibilidade: [P(q=1)] + [<Z_i Z_j>] a partir das probabilidades (convenção LSB)."""
    idx = np.arange(len(pr))
    bits = [(idx >> q) & 1 for q in range(n)]
    m = np.array([pr[bits[q] == 1].sum() for q in range(n)])
    zz = np.array([np.sum(pr * (1 - 2 * (bits[i] ^ bits[j]))) for i, j in pares])
    return np.concatenate([m, zz])


def dispositivos_disponiveis():
    """Dispositivos de simulação disponíveis nesta máquina ('CPU', e 'GPU' se houver)."""
    try:
        from qiskit_aer import AerSimulator
        return tuple(AerSimulator().available_devices())
    except ImportError:
        return ("CPU",)


def _checar_device(device):
    if device != "CPU" and device not in dispositivos_disponiveis():
        raise ValueError(f"device '{device}' indisponível nesta máquina (disponíveis: "
                         f"{dispositivos_disponiveis()}). GPU exige qiskit-aer-gpu (Linux + CUDA).")


# ── contrato ────────────────────────────────────────────────────────────────

class Backend:
    """Contrato de todo backend do QURIOS."""
    nome = "base"
    exato = False          # True: sem ruído de shots
    max_qubits = None      # limite prático (None = sem limite fixo)

    def _validar(self, X, ansatz):
        if not isinstance(ansatz, Ansatz):
            raise TypeError("ansatz deve ser um quirios_engine.Ansatz")
        X = np.atleast_2d(np.asarray(X, float))
        if X.shape[1] != ansatz.n_features:
            raise ValueError(f"X deve ter {ansatz.n_features} colunas, tem {X.shape[1]}")
        if self.max_qubits is not None and ansatz.n_qubits > self.max_qubits:
            raise ValueError(f"backend '{self.nome}' suporta até {self.max_qubits} qubits "
                             f"(pedido: {ansatz.n_qubits}); use BackendMPS ou BackendQiskit")
        return X

    def _calcular(self, X, ansatz, params):
        raise NotImplementedError

    def features(self, X, ansatz, params):
        return self._calcular(self._validar(X, ansatz), ansatz, np.asarray(params, float))

    def submeter(self, X, ansatz, params):
        X = self._validar(X, ansatz)
        return TarefaLocal(self.nome, executor().submit(self._calcular, X, ansatz,
                                                        np.asarray(params, float)))

    def capacidades(self):
        return {"nome": self.nome, "exato": self.exato, "max_qubits": self.max_qubits,
                "assincrono_nativo": type(self).submeter is not Backend.submeter}

    def __repr__(self):
        return f"{type(self).__name__}(nome='{self.nome}')"


# ── simuladores exatos ──────────────────────────────────────────────────────

class BackendExato(Backend):
    nome = "exato"
    exato = True
    max_qubits = MAX_QUBITS_VETORIAL

    def _calcular(self, X, ansatz, params):
        return np.array([ansatz.medir(ansatz.estado(x, params)) for x in X])


class BackendShots(Backend):
    """Simulação exata + ruído de shots EMULADO: cada feature é estimada com
    `shots` medições (binomial exata de cada observável +-1). Rápido para
    comparar modelos quânticos e clássicos com o MESMO orçamento de medições,
    sem montar circuitos no Aer. Aproximação: amostra cada observável de forma
    independente (ignora correlações entre features da mesma medição)."""
    max_qubits = MAX_QUBITS_VETORIAL

    def __init__(self, shots, seed=None, base=None):
        if shots < 1:
            raise ValueError("shots deve ser >= 1")
        self.shots, self.base = int(shots), base or BackendExato()
        self._rng = np.random.default_rng(seed)
        self.nome = f"shots({self.shots})"

    def _calcular(self, X, ansatz, params):
        F = self.base._calcular(X, ansatz, params)
        n, S = ansatz.n_qubits, self.shots
        out = F.copy()
        if ansatz.leitura in ("z", "z_zz"):               # P(qubit = 1) -> fração de 1s
            out[:, :n] = self._rng.binomial(S, np.clip(F[:, :n], 0, 1)) / S
            resto = slice(n, None)
        else:
            resto = slice(0, None)
        v = np.clip(F[:, resto], -1, 1)                    # valores esperados +-1
        out[:, resto] = 2 * self._rng.binomial(S, (1 + v) / 2) / S - 1
        return out


def _sparse(ansatz):
    from qiskit.quantum_info import SparsePauliOp
    n = ansatz.n_qubits
    return [SparsePauliOp.from_sparse_list([(ops, list(qs), 1.0)], n) for ops, qs in ansatz.observaveis()]


class BackendMPS(Backend):
    def __init__(self, max_bond=None, truncamento=None, device="CPU"):
        _checar_device(device)
        self.max_bond, self.truncamento, self.device = max_bond, truncamento, device
        self.exato = max_bond is None
        self.nome = "mps" + ("" if max_bond is None else f"(chi={max_bond})") + \
                    ("" if device == "CPU" else f"-{device.lower()}")

    def _opcoes(self):
        o = {"method": "matrix_product_state", "device": self.device}
        if self.max_bond is not None:
            o["matrix_product_state_max_bond_dimension"] = int(self.max_bond)
        if self.truncamento is not None:
            o["matrix_product_state_truncation_threshold"] = float(self.truncamento)
        return o

    def _calcular(self, X, ansatz, params):
        from qiskit import transpile
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import EstimatorV2
        obs = _sparse(ansatz)
        circs = [ansatz.circuito_qiskit(x, params, medir=False) for x in X]
        if ansatz.codificacao == "amplitude":      # StatePreparation -> portas do Aer
            circs = transpile(circs, AerSimulator(method="matrix_product_state"), optimization_level=0)
        est = EstimatorV2(options={"backend_options": self._opcoes()})
        res = est.run([(c, obs) for c in circs]).result()
        return np.array([ansatz.converter(np.asarray(r.data.evs, float)) for r in res])


# ── Qiskit com shots: medição, mitigação e ZNE ──────────────────────────────

def dobrar_portas(qc, escala):
    """Zero-noise extrapolation: cada porta de 2 qubits G vira G (G^-1 G)^k,
    escala = 2k + 1. Barreiras impedem o transpilador de cancelar as cópias."""
    from qiskit import QuantumCircuit
    if escala == 1:
        return qc
    if escala < 1 or escala % 2 == 0:
        raise ValueError("escalas de ZNE devem ser ímpares (1, 3, 5, ...)")
    k = (escala - 1) // 2
    novo = QuantumCircuit(*qc.qregs, *qc.cregs)
    for inst in qc.data:
        novo.append(inst.operation, inst.qubits, inst.clbits)
        if inst.operation.num_qubits == 2 and inst.operation.name in ("cx", "cz", "rzz"):
            for _ in range(k):
                novo.barrier(inst.qubits)
                novo.append(inst.operation.inverse(), inst.qubits)
                novo.barrier(inst.qubits)
                novo.append(inst.operation, inst.qubits)
    return novo


def _rotacionar_base(qc, base):
    """Acrescenta a mudança de base por qubit (base: string 'X'/'Y'/'Z' por qubit) e mede."""
    qc = qc.copy()
    for q, b in enumerate(base):
        if b == "X":
            qc.h(q)
        elif b == "Y":
            qc.sdg(q); qc.h(q)
    qc.measure_all()
    return qc


def _contagens(r, n):
    """counts -> (z: (k, n) com +-1, w: pesos)."""
    counts = r.data.meas.get_counts()
    bits = np.array([[int(s[n - 1 - q]) for q in range(n)] for s in counts])   # qubit 0 à direita
    return 1 - 2 * bits, np.array(list(counts.values()), float)


def _esperanca(z, w, qs, cal=None):
    """<prod z_q> sobre as contagens; com cal=(a, b) inverte o erro de leitura
    (E[z_obs] = a z + b por qubit, independentes) para 1 ou 2 qubits."""
    def bruto(S):
        return float(np.sum(w * np.prod(z[:, list(S)], axis=1)) / w.sum())
    if cal is None:
        return bruto(qs)
    a, b = cal
    if len(qs) == 1:
        (i,) = qs
        return (bruto(qs) - b[i]) / a[i]
    if len(qs) == 2:
        i, j = qs
        zi, zj = (bruto((i,)) - b[i]) / a[i], (bruto((j,)) - b[j]) / a[j]
        return (bruto(qs) - a[i] * b[j] * zi - b[i] * a[j] * zj - b[i] * b[j]) / (a[i] * a[j])
    raise ValueError("mitigação de leitura implementada para observáveis de até 2 qubits")


MAX_CIRCUITOS_IQM = 200          # circuitos por job na IQM Resonance (jobs menores atravessam a fila melhor)


def _iqm_simulado(nome):
    """Chip IQM simulado com o ruído dele (pacote iqm-client): 'iqm_garnet' -> IQMFakeGarnet."""
    import iqm.qiskit_iqm as q
    classe = "IQMFake" + nome.replace("iqm_", "").capitalize()
    if not hasattr(q, classe):
        from iqm.qiskit_iqm import fake_backends as q
    return getattr(q, classe)()


class _ContagensPub:
    """O mesmo formato de um resultado do SamplerV2: r.data.meas.get_counts()."""

    def __init__(self, contagens):
        self._contagens = {k.replace(" ", ""): int(v) for k, v in contagens.items()}
        self.data = type("Dados", (), {"meas": self})()

    def get_counts(self):
        return dict(self._contagens)


class JobPorRun:
    """Circuitos enviados por backend.run (IQM), embrulhados como um job do SamplerV2:
    result() devolve a lista de resultados na ordem dos circuitos. Agrupa por shots
    e, na IQM real, em jobs de até `max_por_job` circuitos. metrics() traz os horários
    e o tempo de execução no processador (timeline da IQM), para o comprovante."""

    def __init__(self, backend, circuitos, shots, max_por_job=None):
        self._partes = []                                   # (índices, job)
        grupos = {}
        for i, s in enumerate(shots):
            grupos.setdefault(int(s), []).append(i)
        passo = max_por_job or len(circuitos) or 1
        for s, idx in grupos.items():
            for k in range(0, len(idx), passo):
                parte = idx[k:k + passo]
                self._partes.append((parte, backend.run([circuitos[i] for i in parte], shots=s)))
        self._n, self._resultado = len(circuitos), None

    def job_id(self):
        ids = [j.job_id() for _, j in self._partes if hasattr(j, "job_id")]
        return ",".join(str(i) for i in ids) if ids else None

    def status(self):
        nomes = [getattr(j.status(), "name", str(j.status())).upper() for _, j in self._partes]
        nome = "ERROR" if "ERROR" in nomes else "DONE" if all(n == "DONE" for n in nomes) else "RUNNING"
        return type("Status", (), {"name": nome})()

    def cancel(self):
        for _, j in self._partes:
            j.cancel()

    def result(self):
        if self._resultado is None:
            saida = [None] * self._n
            self._resultados = []
            for idx, job in self._partes:
                r = job.result()
                self._resultados.append(r)
                for pos, i in enumerate(idx):
                    saida[i] = _ContagensPub(r.get_counts(pos))
            self._resultado = saida
        return self._resultado

    def metrics(self):
        """Horários e segundos de execução no processador, somados entre os jobs (IQM)."""
        import datetime
        self.result()
        horarios, segundos = [], 0.0
        for r in self._resultados:
            linha = getattr(r, "timeline", None) or []
            marca = {}
            for t in linha:
                status = getattr(t, "status", None) or (t.get("status") if isinstance(t, dict) else None)
                quando = getattr(t, "timestamp", None) or (t.get("timestamp") if isinstance(t, dict) else None)
                if status and quando:
                    marca[status] = quando
                    horarios.append({"status": status, "timestamp": str(quando)})
            ini, fim = marca.get("execution_started"), marca.get("execution_ended")
            if ini and fim:
                if isinstance(ini, str):
                    ini, fim = (datetime.datetime.fromisoformat(x.replace("Z", "+00:00")) for x in (ini, fim))
                segundos += (fim - ini).total_seconds()
        return {"timestamps": horarios, "usage": {"quantum_seconds": round(segundos, 3) if horarios else None}}


class BackendQiskit(Backend):
    def __init__(self, modo="ideal", shots=4096, dispositivo="ibm_kingston",
                 token=None, instancia=None, otimizacao=1, seed=None, device="CPU",
                 erros=None, mitigacao=(), escalas_zne=(1, 3, 5), grau_zne=1,
                 medicao="bases", n_sombras=64, portas_fracionarias=False):
        if modo not in ("ideal", "ruidoso", "parametrico", "ibm", "iqm"):
            raise ValueError("modo deve ser 'ideal', 'ruidoso', 'parametrico', 'ibm' ou 'iqm'")
        if medicao not in ("bases", "sombras"):
            raise ValueError("medicao deve ser 'bases' ou 'sombras'")
        mitigacao = tuple(mitigacao)
        if any(m not in ("leitura", "zne") for m in mitigacao):
            raise ValueError("mitigacao aceita 'leitura' e/ou 'zne'")
        if modo not in ("ibm", "iqm"):
            _checar_device(device)
        if modo == "parametrico":
            from .futuro import resolver_erros
            erros = resolver_erros(erros)
        self.modo, self.shots, self.dispositivo = modo, shots, dispositivo
        self.token, self.instancia, self.otimizacao, self.seed = token, instancia, otimizacao, seed
        self.device, self.erros = device, erros
        self.mitigacao, self.escalas_zne, self.grau_zne = mitigacao, tuple(escalas_zne), grau_zne
        self.medicao, self.n_sombras = medicao, n_sombras
        self.portas_fracionarias = portas_fracionarias
        sufixo = {"ideal": "", "ruidoso": f"-{dispositivo}", "ibm": f"-{dispositivo}", "iqm": f"-{dispositivo}",
                  "parametrico": f"-p2q={erros['erro_2q']:.0e}" if erros else ""}[modo]
        extras = ("+" + "+".join(mitigacao) if mitigacao else "") + ("+sombras" if medicao == "sombras" else "")
        self.nome = f"qiskit-{modo}{sufixo}{extras}"
        self._backend = None
        self._servico_ibm = None
        self.ultimo_job_id = None
        self.registro_jobs = []        # comprovante: um item por job enviado (ver qpu.py)

    def _registrar_job(self, job, tipo, n_circuitos, shots, contagens=None):
        """Guarda os metadados verificáveis de um job (id, backend, horários, uso de QPU)."""
        import datetime
        import hashlib
        import json as _json
        item = {"job_id": job.job_id() if hasattr(job, "job_id") else None, "tipo": tipo,
                "backend": getattr(self._backend, "name", str(self._backend)), "modo": self.modo,
                "circuitos": int(n_circuitos), "shots": int(shots),
                "registrado_em": datetime.datetime.now().isoformat(timespec="seconds")}
        if contagens is not None:
            bruto = _json.dumps(contagens, sort_keys=True).encode()
            item["contagens_sha256"] = hashlib.sha256(bruto).hexdigest()
        try:                                       # IBM Runtime: uso real de QPU e horários
            m = job.metrics()
            item["uso_qpu_s"] = (m.get("usage") or {}).get("quantum_seconds")
            item["horarios"] = m.get("timestamps")
        except Exception:
            pass
        if item.get("uso_qpu_s") is None and hasattr(job, "usage"):
            try:                                   # qiskit-ibm-runtime novo: o tempo saiu de metrics() para usage()
                item["uso_qpu_s"] = float(job.usage())
            except Exception:
                pass
        self.registro_jobs.append(item)
        return item

    def _servico(self):
        if self._servico_ibm is None:
            from qiskit_ibm_runtime import QiskitRuntimeService
            kw = {"channel": "ibm_quantum_platform"}
            if self.token: kw["token"] = self.token
            if self.instancia: kw["instance"] = self.instancia
            self._servico_ibm = QiskitRuntimeService(**kw)
        return self._servico_ibm

    def _get_backend(self):
        if self._backend is not None:
            return self._backend
        if self.modo == "iqm":                    # IQM Resonance (token em IQM_TOKEN; ver conta_iqm.py)
            from iqm.qiskit_iqm import IQMProvider
            from .conta_iqm import URL_RESONANCE, carregar_token
            if not carregar_token():
                raise RuntimeError("sem token da IQM: rode `python -m quirios_engine.conta_iqm`")
            self._backend = IQMProvider(URL_RESONANCE, quantum_computer=self.dispositivo).get_backend()
            return self._backend
        if self.modo == "ruidoso" and self.dispositivo.startswith("iqm_"):    # ensaio com o ruído de um chip IQM
            self._backend = _iqm_simulado(self.dispositivo)
            return self._backend
        from qiskit_aer import AerSimulator
        if self.modo == "ideal":
            self._backend = AerSimulator(seed_simulator=self.seed, device=self.device)
        elif self.modo == "ruidoso":
            import qiskit_ibm_runtime.fake_provider as fp
            classe = "Fake" + "".join(p.capitalize() for p in self.dispositivo.replace("ibm_", "").split("_"))
            self._backend = AerSimulator.from_backend(getattr(fp, classe)(), seed_simulator=self.seed,
                                                      device=self.device)
        elif self.modo == "parametrico":
            from .futuro import modelo_de_ruido
            self._backend = AerSimulator(noise_model=modelo_de_ruido(**self.erros),
                                         seed_simulator=self.seed, device=self.device)
        else:
            kw = {"use_fractional_gates": True} if self.portas_fracionarias else {}
            self._backend = self._servico().backend(self.dispositivo, **kw)
        return self._backend

    def _sampler(self, backend):
        if self.modo == "ibm":
            from qiskit_ibm_runtime import SamplerV2
            return SamplerV2(mode=backend)
        # O SamplerV2 do Aer cria o próprio simulador: o modelo de ruído
        # precisa ir explícito (bug da v0.3: o modo "ruidoso" rodava sem ruído).
        from qiskit_aer.primitives import SamplerV2
        opcoes = {"device": self.device}
        if self.modo in ("ruidoso", "parametrico"):
            opcoes["noise_model"] = backend.options.noise_model
        return SamplerV2(seed=self.seed, options={"backend_options": opcoes})

    def _via_run(self):
        """IQM não tem o Sampler da IBM: os circuitos vão por backend.run."""
        return self.modo == "iqm" or (self.modo == "ruidoso" and self.dispositivo.startswith("iqm_"))

    def _transpilar(self, circuitos, backend):
        if self._via_run():                       # o transpile do Qiskit chama o plugin da IQM (portas nativas)
            from qiskit import transpile
            return transpile(list(circuitos), backend=backend, optimization_level=self.otimizacao,
                             seed_transpiler=self.seed)
        from qiskit.transpiler import generate_preset_pass_manager
        # semente do compilador: sem ela a escolha de qubits muda a cada execução e o ensaio não se repete
        return generate_preset_pass_manager(optimization_level=self.otimizacao, backend=backend,
                                            seed_transpiler=self.seed).run(list(circuitos))

    def _executar(self, backend, isa, shots):
        """Envia os circuitos (um shots por circuito) e devolve um job no formato do SamplerV2."""
        if self._via_run():
            return JobPorRun(backend, isa, shots, max_por_job=MAX_CIRCUITOS_IQM if self.modo == "iqm" else None)
        return self._sampler(backend).run([(c, None, s) for c, s in zip(isa, shots)])

    def amostrar_lote(self, circuitos, shots=None, tipo="amostragem"):
        """Mede vários circuitos (com measure_all) num ÚNICO job e devolve a lista
        de contagens {bitstring: vezes}. O job fica em registro_jobs."""
        backend = self._get_backend()
        s = int(shots or self.shots)
        isa = self._transpilar(circuitos, backend)
        job = self._executar(backend, isa, [s] * len(isa))
        contagens = [r.data.meas.get_counts() for r in job.result()]
        self.ultimo_job_id = job.job_id() if hasattr(job, "job_id") else None
        self._registrar_job(job, tipo, len(isa), s, contagens)
        return contagens

    def amostrar(self, circuito, shots=None):
        """Mede um circuito qualquer (com measure_all) neste backend e devolve
        as contagens {bitstring: vezes}. Usado, por exemplo, para rodar o
        circuito gerador da evolução quântica numa QPU."""
        return self.amostrar_lote([circuito], shots, tipo="amostragem")[0]

    def _configuracoes(self, ansatz, rng):
        """Bases de medição por amostra: lista de strings (uma letra por qubit)."""
        n = ansatz.n_qubits
        if self.medicao == "sombras":
            return [["".join(rng.choice(list("XYZ"), n)) for _ in range(self.n_sombras)]]
        letras = []
        for ops, _ in ansatz.observaveis():
            if len(set(ops)) != 1:
                raise ValueError(f"leitura '{ansatz.leitura}' mistura bases num observável ({ops}); "
                                 f"use medicao='sombras'")
            if ops[0] not in letras:
                letras.append(ops[0])
        return [[L * n for L in letras]]

    def submeter(self, X, ansatz, params):
        """Transpila e envia TODOS os circuitos (amostras x bases x escalas ZNE +
        calibração) num único job; não espera o resultado."""
        from qiskit import QuantumCircuit
        X = self._validar(X, ansatz)
        n = ansatz.n_qubits
        backend = self._get_backend()
        rng = np.random.default_rng(self.seed)
        escalas = self.escalas_zne if "zne" in self.mitigacao else (1,)
        sombras = self.medicao == "sombras"
        shots = max(1, self.shots // self.n_sombras) if sombras else self.shots

        bases_por_amostra = [self._configuracoes(ansatz, rng)[0] for _ in X] if sombras else \
            self._configuracoes(ansatz, rng) * len(X)
        circs, plano = [], []
        if "leitura" in self.mitigacao:                       # calibração: |0...0> e |1...1>
            zero, um = QuantumCircuit(n), QuantumCircuit(n)
            um.x(range(n))
            zero.measure_all(); um.measure_all()
            circs += [zero, um]
        for s, x in enumerate(X):
            base_circ = ansatz.circuito_qiskit(x, params, medir=False)
            for e in escalas:
                dobrado = dobrar_portas(base_circ, e)
                for base in bases_por_amostra[s]:
                    circs.append(_rotacionar_base(dobrado, base))
                    plano.append((s, e, base))
        isa = self._transpilar(circs, backend)
        job = self._executar(backend, isa, [self.shots if i < 2 and "leitura" in self.mitigacao else shots
                                            for i in range(len(isa))])
        tarefa = TarefaQiskit(self.nome, job, self._posprocessador(ansatz, len(X), escalas, plano))
        tarefa.n_circuitos, tarefa.shots_por_circuito = len(isa), shots     # para o comprovante do job
        self.ultimo_job_id = tarefa.id
        return tarefa

    def _posprocessador(self, ansatz, n_amostras, escalas, plano):
        n, obs = ansatz.n_qubits, ansatz.observaveis()
        mit_leitura = "leitura" in self.mitigacao
        sombras = self.medicao == "sombras"
        grau = min(self.grau_zne, len(escalas) - 1)

        def converter(res):
            res = list(res)
            cal = None
            if mit_leitura:
                z0, w0 = _contagens(res[0], n)
                z1, w1 = _contagens(res[1], n)
                p10 = np.array([np.sum(w0[z0[:, q] == -1]) / w0.sum() for q in range(n)])  # leu 1 sendo 0
                p01 = np.array([np.sum(w1[z1[:, q] == 1]) / w1.sum() for q in range(n)])   # leu 0 sendo 1
                a = np.clip(1 - p10 - p01, 1e-3, None)
                cal = (a, p01 - p10)
                res = res[2:]
            soma = np.zeros((n_amostras, len(escalas), len(obs)))
            conta = np.zeros((n_amostras, len(escalas), len(obs)))
            for r, (s, e, base) in zip(res, plano):
                z, w = _contagens(r, n)
                ie = escalas.index(e)
                for k, (ops, qs) in enumerate(obs):
                    if all(base[q] == P for P, q in zip(ops, qs)):
                        v = _esperanca(z, w, qs, cal)
                        soma[s, ie, k] += (3 ** len(qs)) * v if sombras else v
                    if sombras or all(base[q] == P for P, q in zip(ops, qs)):
                        conta[s, ie, k] += 1
            evs = soma / np.maximum(conta, 1)               # sombras: média sobre TODAS as bases
            if len(escalas) > 1:                             # ZNE: ajuste polinomial até escala 0
                x = np.asarray(escalas, float)
                evs = np.array([[np.polyval(np.polyfit(x, evs[s, :, k], grau), 0.0)
                                 for k in range(len(obs))] for s in range(n_amostras)])
            else:
                evs = evs[:, 0, :]
            return np.array([ansatz.converter(np.clip(v, -1, 1)) for v in evs])
        return converter

    def features(self, X, ansatz, params):
        t = self.submeter(X, ansatz, params)
        F = t.resultado()
        # Circuitos de verdade (amostras x bases x escalas ZNE + calibração), não amostras.
        contagens = [r.data.meas.get_counts() for r in t.job.result()]
        self._registrar_job(t.job, "features", t.n_circuitos, t.shots_por_circuito, contagens)
        return F

    def recuperar(self, job_id, ansatz):
        """Retoma um job IBM enviado antes (até por outro processo) pelo id.
        Aceita o `ansatz` usado no envio ou, por compatibilidade, n_qubits
        (leitura padrão). Só para medicao='bases' sem mitigação."""
        if self.modo != "ibm":
            raise ValueError("recuperar só existe para modo='ibm' (jobs Aer vivem só no processo)")
        if self.mitigacao or self.medicao != "bases":
            raise ValueError("recuperar não suporta mitigação nem sombras (o plano de circuitos não é salvo)")
        if isinstance(ansatz, (int, np.integer)):
            from .core import QuantumConfig
            ansatz = Ansatz(QuantumConfig(int(ansatz), "linear", 1))
        bases = self._configuracoes(ansatz, np.random.default_rng(0))[0]

        def converter(res):                    # nº de amostras só é conhecido com o resultado
            res = list(res)
            m = len(res) // len(bases)
            plano = [(s, 1, b) for s in range(m) for b in bases]
            return self._posprocessador(ansatz, m, (1,), plano)(res)
        return TarefaQiskit(self.nome, self._servico().job(job_id), converter)


# ── fábrica e plugins ───────────────────────────────────────────────────────

def _plugins():
    from importlib.metadata import entry_points
    return {ep.name: ep for ep in entry_points(group="quirios.backends")}


def backends_registrados():
    """Nomes aceitos por criar_backend (internos + plugins instalados)."""
    return (["exato", "mps", "qiskit-ideal", "qiskit-ruidoso", "parametrico", "ibm",
             "estimador-local", "ibm-estimador"] + sorted(_plugins()))


def criar_backend(nome, **kw):
    """Fábrica por nome: 'exato', 'mps', 'qiskit-ideal', 'qiskit-ruidoso',
    'parametrico', 'ibm', 'estimador-local', 'ibm-estimador' ou o nome de um
    plugin (entry point 'quirios.backends')."""
    if nome == "exato":
        return BackendExato()
    if nome == "mps":
        return BackendMPS(**kw)
    if nome in ("estimador-local", "ibm-estimador"):
        from .ibm_estimador import BackendEstimador
        return BackendEstimador("local" if nome == "estimador-local" else "ibm", **kw)
    modos = {"qiskit-ideal": "ideal", "qiskit-ruidoso": "ruidoso", "parametrico": "parametrico", "ibm": "ibm"}
    if nome in modos:
        return BackendQiskit(modo=modos[nome], **kw)
    plugins = _plugins()
    if nome in plugins:
        classe = plugins[nome].load()
        b = classe(**kw)
        if not isinstance(b, Backend):
            raise TypeError(f"plugin '{nome}' não devolveu um quirios_engine.Backend")
        return b
    raise ValueError(f"backend desconhecido: '{nome}' (disponíveis: {backends_registrados()})")


def circuito_qiskit(x, theta, phi, alpha, config, medir=True):
    """Compatibilidade v0.3/v0.4: circuito de 1 camada a partir de theta/phi/alpha soltos."""
    ans = Ansatz(config, camada_pos=alpha is not None)
    return ans.circuito_qiskit(x, ans.empacotar(theta, phi, alpha), medir=medir)
