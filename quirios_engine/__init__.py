"""
QURIOS Engine -- camada de infraestrutura entre software clássico e quântico.

Uso básico:

    from quirios_engine import QuantumConfig, VQCClassifier
    from quirios_engine.entropy import entropia_emaranhamento, mutual_information

    config = QuantumConfig(n_qubits=8, topologia="anel")
    clf = VQCClassifier(config)
    clf.fit(X_treino, y_treino)
    preds = clf.predict(X_teste)
"""
__version__ = "0.8.0"

from .core import (
    QuantumConfig,
    TOPOLOGIAS,
    MAX_QUBITS_VETORIAL,
    apply_gate,
    apply_perm,
    gate_H,
    gate_T,
    gate_Ry,
    gate_Rz,
    build_cnot_perm,
    compose_perms,
    topologia_anel,
    topologia_linear,
    topologia_todos_pares,
    topologia_blocos,
)
from .ansatz import Ansatz, esperanca_pauli, qubits_para_amplitude
from .entropy import (
    entropia_medicao,
    entropia_emaranhamento,
    mutual_information,
    pureza,
)
from .tarefas import Tarefa, em_segundo_plano
from .backends import (Backend, BackendExato, BackendMPS, BackendQiskit, backends_registrados,
                       circuito_qiskit, criar_backend, dispositivos_disponiveis, dobrar_portas)
from .otimizadores import SPSA, ParameterShift
from .vqc import (VQCClassifier, circuito_treinavel, grad_parameter_shift,
                  cone_de_luz, loss_e_grad, ajustar_leitura, logistica_l2)
from .diagnostico import variancia_gradiente, estimar_circuitos
from .sistema import ClassificadorQURIOS, Brutos, RFF, Quantico
from .kernels import ClassificadorKernel, KernelQuantico, alinhamento, concentracao, treinar_alinhamento
from .futuro import GERACOES, projetar_hardware

from .ibm_estimador import BackendEstimador

# Módulos que também rodam como linha de comando (python -m ...) são importados
# sob demanda: assim o runpy não os encontra já carregados pelo pacote.
_PREGUICOSOS = {
    "BuscaEvolutiva": "evolucao", "RegistroCampeoes": "evolucao", "evoluir_continuamente": "evolucao",
    "EvolucaoQuantica": "evolucao_quantica", "GeradorQuantico": "evolucao_quantica",
    "CodificacaoGenoma": "evolucao_quantica",
    "fases_ising": "dados_quanticos", "comparar_dados_quanticos": "dados_quanticos",
    "comparar_memoria_quantica": "memoria_quantica",
    "vetor_para_mps": "redes_tensoriais", "mps_para_circuito": "redes_tensoriais",
    "CamadaTT": "ia_tensorial", "comprimir_modelo": "ia_tensorial",
}


def __getattr__(nome):
    if nome in _PREGUICOSOS:
        import importlib
        mod = importlib.import_module(f".{_PREGUICOSOS[nome]}", __name__)
        return getattr(mod, "comparar" if nome.startswith("comparar_") else nome)
    raise AttributeError(f"module 'quirios_engine' has no attribute '{nome}'")

__all__ = [
    # fase 7: memória quântica, redes tensoriais e IA tensorial
    "comparar_memoria_quantica",
    "vetor_para_mps",
    "mps_para_circuito",
    "CamadaTT",
    "comprimir_modelo",
    # fase 6: evolução quântica, dados quânticos, caminho IBM profundo
    "EvolucaoQuantica",
    "GeradorQuantico",
    "CodificacaoGenoma",
    "fases_ising",
    "comparar_dados_quanticos",
    "BackendEstimador",
    # fase 5: núcleo quântico auto-evolutivo e tecnologia emergente
    "esperanca_pauli",
    "backends_registrados",
    "dobrar_portas",
    "KernelQuantico",
    "ClassificadorKernel",
    "alinhamento",
    "concentracao",
    "treinar_alinhamento",
    "BuscaEvolutiva",
    "RegistroCampeoes",
    "evoluir_continuamente",
    "GERACOES",
    "projetar_hardware",
    # configuração e circuito
    "QuantumConfig",
    "Ansatz",
    "TOPOLOGIAS",
    "MAX_QUBITS_VETORIAL",
    "qubits_para_amplitude",
    "apply_gate",
    "apply_perm",
    "gate_H",
    "gate_T",
    "gate_Ry",
    "gate_Rz",
    "build_cnot_perm",
    "compose_perms",
    "topologia_anel",
    "topologia_linear",
    "topologia_todos_pares",
    "topologia_blocos",
    # backends e tarefas assíncronas
    "Backend",
    "BackendExato",
    "BackendMPS",
    "BackendQiskit",
    "criar_backend",
    "dispositivos_disponiveis",
    "circuito_qiskit",
    "Tarefa",
    "em_segundo_plano",
    # treino
    "VQCClassifier",
    "SPSA",
    "ParameterShift",
    "circuito_treinavel",
    "grad_parameter_shift",
    "loss_e_grad",
    "ajustar_leitura",
    "logistica_l2",
    # diagnóstico e métricas
    "cone_de_luz",
    "variancia_gradiente",
    "estimar_circuitos",
    "entropia_medicao",
    "entropia_emaranhamento",
    "mutual_information",
    "pureza",
    # sistema de geradores
    "ClassificadorQURIOS",
    "Brutos",
    "RFF",
    "Quantico",
]
