"""
Caminho IBM profundo: valores esperados pelo primitivo Estimator, com a
pilha de mitigação da IBM e operator backpropagation.

    b = BackendEstimador("local", obp=True)                    # desenvolvimento/testes
    b = BackendEstimador("ibm", resiliencia=2, obp=True)       # QPU real
    F = b.features(X, ansatz, params)
    b.ultima_reducao     # profundidade do circuito antes/depois do OBP

Modos:
    "local"  Aer Estimator: exato, ou com o ruído de uma geração de hardware
             (erros="hoje", ver futuro.py). Testa todo o caminho sem conta.
    "ibm"    QPU IBM com o Estimator do Runtime (o primitivo novo, de
             processamento no cliente, quando instalado; senão o EstimatorV2).
             resiliencia 0 = nada, 1 = TREX (mitigação de leitura com
             twirling), 2 = TREX + ZNE. desacoplamento = dynamical decoupling
             (XpXm) nos tempos ociosos; twirling = twirling de Pauli nas portas.

Operator backpropagation (obp=True, extra [ibm]): as últimas fatias do
circuito são absorvidas CLASSICAMENTE no observável (qiskit-addon-obp), até
um orçamento de erro de truncamento. O circuito que vai para a QPU fica mais
raso -- menos portas, menos ruído -- ao custo de medir mais termos de Pauli.
Funciona melhor em circuitos perto de Clifford; o orçamento controla a troca
entre precisão e profundidade.
"""
import numpy as np

from .backends import Backend, _sparse
from .tarefas import TarefaQiskit


class BackendEstimador(Backend):
    def __init__(self, modo="local", erros=None, dispositivo="ibm_kingston", token=None, instancia=None,
                 shots=4096, resiliencia=1, desacoplamento=True, twirling=True, obp=False, obp_erro_max=0.02,
                 obp_max_grupos=16, otimizacao=1, seed=None, opcoes_extra=None):
        if modo not in ("local", "ibm"):
            raise ValueError("modo deve ser 'local' ou 'ibm'")
        if resiliencia not in (0, 1, 2):
            raise ValueError("resiliencia deve ser 0, 1 ou 2")
        if modo == "local" and erros is not None:
            from .futuro import resolver_erros
            erros = resolver_erros(erros)
        self.modo, self.erros, self.dispositivo = modo, erros, dispositivo
        self.token, self.instancia, self.shots = token, instancia, shots
        self.resiliencia, self.desacoplamento, self.twirling = resiliencia, desacoplamento, twirling
        self.obp, self.obp_erro_max, self.obp_max_grupos = obp, obp_erro_max, obp_max_grupos
        self.otimizacao, self.seed, self.opcoes_extra = otimizacao, seed, opcoes_extra or {}
        self.exato = modo == "local" and erros is None
        self.nome = (f"estimador-{modo}" + (f"-{dispositivo}" if modo == "ibm" else "")
                     + ("-ruidoso" if modo == "local" and erros else "") + ("+obp" if obp else ""))
        self.ultima_reducao = None
        self.ultimo_job_id = None
        self._servico_ibm = None

    # ── opções da IBM ───────────────────────────────────────────────────────

    def opcoes_ibm(self):
        """Dicionário de EstimatorOptions do Runtime para esta configuração."""
        o = {"default_shots": int(self.shots), "resilience_level": int(self.resiliencia),
             "dynamical_decoupling": {"enable": bool(self.desacoplamento), "sequence_type": "XpXm"},
             "twirling": {"enable_gates": bool(self.twirling), "enable_measure": self.resiliencia >= 1}}
        for k, v in self.opcoes_extra.items():
            if isinstance(v, dict) and isinstance(o.get(k), dict):
                o[k] = {**o[k], **v}
            else:
                o[k] = v
        return o

    def _servico(self):
        if self._servico_ibm is None:
            from qiskit_ibm_runtime import QiskitRuntimeService
            kw = {"channel": "ibm_quantum_platform"}
            if self.token: kw["token"] = self.token
            if self.instancia: kw["instance"] = self.instancia
            self._servico_ibm = QiskitRuntimeService(**kw)
        return self._servico_ibm

    # ── operator backpropagation ────────────────────────────────────────────

    def _reduzir(self, circ, observaveis):
        """Absorve as últimas fatias do circuito nos observáveis (qiskit-addon-obp)."""
        try:
            from qiskit_addon_obp import backpropagate
            from qiskit_addon_obp.utils.simplify import OperatorBudget
            from qiskit_addon_obp.utils.truncating import setup_budget
            from qiskit_addon_utils.slicing import combine_slices, slice_by_depth
        except ImportError as e:
            raise ImportError('obp=True exige o extra: pip install -e ".[ibm]"') from e
        from qiskit import QuantumCircuit
        fatias = slice_by_depth(circ, 1)
        orc = setup_budget(max_error_total=self.obp_erro_max, num_slices=len(fatias))
        obs_bp, resto, _ = backpropagate(list(observaveis), fatias, truncation_error_budget=orc,
                                         operator_budget=OperatorBudget(max_qwc_groups=self.obp_max_grupos))
        reduzido = combine_slices(resto, include_barriers=False) if resto else QuantumCircuit(circ.num_qubits)
        return reduzido, obs_bp

    # ── execução ────────────────────────────────────────────────────────────

    def _preparar_pubs(self, X, ansatz, params):
        from qiskit import transpile
        obs = _sparse(ansatz)
        pubs, prof = [], []
        for x in X:
            c = ansatz.circuito_qiskit(x, params, medir=False)
            if ansatz.codificacao == "amplitude":
                c = transpile(c, basis_gates=["ry", "rz", "cx", "h", "t", "rzz", "cz", "rx"], optimization_level=0)
            p0 = c.depth()
            o = obs
            if self.obp:
                c, o = self._reduzir(c, obs)
            prof.append((p0, c.depth()))
            pubs.append((c, o))
        self.ultima_reducao = {"profundidade_original": float(np.mean([a for a, _ in prof])),
                               "profundidade_executada": float(np.mean([b for _, b in prof]))}
        return pubs

    def submeter(self, X, ansatz, params):
        X = self._validar(X, ansatz)
        params = np.asarray(params, float)
        pubs = self._preparar_pubs(X, ansatz, params)
        if self.modo == "local":
            from qiskit_aer.primitives import EstimatorV2
            opcoes = {"backend_options": {}}
            if self.erros:
                from .futuro import modelo_de_ruido
                opcoes["backend_options"]["noise_model"] = modelo_de_ruido(**self.erros)
            if self.seed is not None:
                opcoes["run_options"] = {"seed_simulator": self.seed}
            job = EstimatorV2(options=opcoes).run(pubs)
        else:
            from qiskit.transpiler import generate_preset_pass_manager
            try:                                  # primitivo novo (processamento no cliente)
                from qiskit_ibm_runtime.executor_estimator import Estimator
            except ImportError:                   # versões anteriores do Runtime
                from qiskit_ibm_runtime import EstimatorV2 as Estimator
            backend = self._servico().backend(self.dispositivo)
            pm = generate_preset_pass_manager(optimization_level=self.otimizacao, backend=backend)
            isa = []
            for c, o in pubs:
                ci = pm.run(c)
                isa.append((ci, [ob.apply_layout(ci.layout) for ob in o]))
            job = Estimator(mode=backend, options=self.opcoes_ibm()).run(isa)

        def converter(res):
            return np.array([ansatz.converter(np.clip(np.asarray(r.data.evs, float).ravel(), -1, 1))
                             for r in res])
        tarefa = TarefaQiskit(self.nome, job, converter)
        self.ultimo_job_id = tarefa.id
        return tarefa

    def features(self, X, ansatz, params):
        return self.submeter(X, ansatz, params).resultado()
