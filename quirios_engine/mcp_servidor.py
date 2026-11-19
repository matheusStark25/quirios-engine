"""
Servidor MCP (Model Context Protocol) do QURIOS: agentes de IA -- Claude,
IDEs, outros clientes MCP -- usam o QURIOS como ferramenta, sem código.

    pip install -e ".[mcp]"
    python -m quirios_engine.mcp_servidor          # transporte stdio

Exemplo de configuração de cliente (claude_desktop_config.json / .mcp.json):

    {"mcpServers": {"quirios": {"command": "python",
                                "args": ["-m", "quirios_engine.mcp_servidor"]}}}

Ferramentas: status_quirios, treinar_vqc, prever, orquestrar, evoluir_circuito
(motor quântico ou clássico), campeoes, projetar_hardware, diagnosticar_kernel,
comparar_dados_quanticos. Dados entram como listas
JSON (X: lista de amostras, y: 0/1). Os modelos ficam na memória do servidor.
Registro de campeões: QURIOS_REGISTRO (padrão campeoes_quirios.json).

Proteção (protecao.Guarda): QURIOS_MCP_FERRAMENTAS (lista das liberadas, separadas por
vírgula), QURIOS_MCP_CHAMADAS_HORA (padrão 600), QURIOS_AUDITORIA (trilha encadeada).
Nenhuma ferramenta executa código, abre caminhos de arquivo ou muda configuração a
partir do que o agente envia: tudo o que chega é tratado como dado.
"""
import os
import uuid

import numpy as np

try:
    from mcp.server.mcpserver import MCPServer
except ImportError as e:                      # pragma: no cover
    raise ImportError('O servidor MCP exige o extra: pip install -e ".[mcp]"') from e

from . import __version__
from .backends import backends_registrados, dispositivos_disponiveis
from .core import QuantumConfig
from .futuro import GERACOES
from .protecao import Guarda

servidor = MCPServer(
    "quirios",
    title="QURIOS Engine",
    version=__version__,
    instructions=("Infraestrutura entre software clássico e quântico. Treine classificadores "
                  "quânticos variacionais, compare clássico vs quântico com evidência estatística, "
                  "evolua circuitos automaticamente e projete o desempenho em hardware futuro. "
                  "X deve ser uma lista de amostras numéricas; y, rótulos 0/1."),
)
_modelos = {}
MAX_QUBITS = 16
# Agentes de IA são clientes não confiáveis: lista de ferramentas, orçamento, limites e auditoria.
_guarda = Guarda.do_ambiente()


def _dados(X, y=None, n_qubits=None):
    X = np.asarray(X, float)
    if X.ndim != 2 or len(X) < 4:
        raise ValueError("X deve ser uma lista de pelo menos 4 amostras")
    if y is not None:
        y = np.asarray(y, float)
        if len(y) != len(X) or set(np.unique(y)) != {0.0, 1.0}:
            raise ValueError("y deve ter um rótulo 0/1 por amostra, com as duas classes")
    lo, hi = X.min(0), X.max(0)
    Xs = (X - lo) / np.where(hi > lo, hi - lo, 1.0)            # codificação por ângulo pede [0, 1]
    nq = int(n_qubits or min(X.shape[1], 8))
    if not 2 <= nq <= MAX_QUBITS:
        raise ValueError(f"n_qubits deve estar entre 2 e {MAX_QUBITS}")
    return Xs, y, nq, (lo, hi)


def _escalar(X, escala):
    lo, hi = escala
    return np.clip((np.asarray(X, float) - lo) / np.where(hi > lo, hi - lo, 1.0), 0, 1)


def _registro():
    from .evolucao import RegistroCampeoes
    return RegistroCampeoes(os.environ.get("QURIOS_REGISTRO", "campeoes_quirios.json"))


@servidor.tool()
@_guarda.proteger
def status_quirios() -> dict:
    """Versão do QURIOS, backends disponíveis, dispositivos e gerações de hardware simuláveis."""
    return {"versao": __version__, "backends": backends_registrados(),
            "dispositivos": list(dispositivos_disponiveis()), "geracoes_hardware": GERACOES,
            "modelos_em_memoria": len(_modelos)}


@servidor.tool()
@_guarda.proteger
def treinar_vqc(X: list[list[float]], y: list[int], n_qubits: int = 0, topologia: str = "anel",
                camadas_entrelacamento: int = 1, otimizador: str = "spsa", epocas: int = 20,
                codificacao: str = "angulo", emaranhador: str = "cnot", leitura: str = "z") -> dict:
    """Treina um classificador quântico variacional (simulação exata). X é escalado
    para [0, 1] automaticamente. Devolve o id do modelo, acurácia de treino e custo."""
    from .vqc import VQCClassifier
    Xs, y_, nq, esc = _dados(X, y, n_qubits or None)
    clf = VQCClassifier(QuantumConfig(nq, topologia, camadas_entrelacamento), otimizador=otimizador,
                        epocas=epocas, codificacao=codificacao, emaranhador=emaranhador,
                        leitura=leitura, seed=0).fit(Xs, y_)
    mid = uuid.uuid4().hex[:12]
    _modelos[mid] = (clf, esc)
    return {"modelo_id": mid, "acuracia_treino": float(np.mean(clf.predict(Xs) == y_)),
            "circuitos_executados": int(clf.circuitos_executados),
            "parametros": int(clf.ansatz.n_params), "ansatz": repr(clf.ansatz)}


@servidor.tool()
@_guarda.proteger
def prever(modelo_id: str, X: list[list[float]]) -> dict:
    """Predições (0/1) e probabilidades de um modelo treinado nesta sessão."""
    if modelo_id not in _modelos:
        raise ValueError(f"modelo '{modelo_id}' não encontrado")
    modelo, esc = _modelos[modelo_id]
    p = modelo.predict_proba(_escalar(X, esc))
    return {"predicoes": (p > 0.5).astype(int).tolist(), "probabilidades": [float(v) for v in p]}


@servidor.tool()
@_guarda.proteger
def orquestrar(X: list[list[float]], y: list[int], n_qubits: int = 0, incluir_kernel: bool = True,
               incluir_vqc: bool = False, chave_campeao: str = "") -> dict:
    """Compara candidatos clássicos e quânticos na validação cruzada e escolhe o
    quântico só com evidência estatística (p < 0,05). Devolve o relatório e o id
    do modelo escolhido."""
    from .orquestrador import Orquestrador, candidatos_padrao
    Xs, y_, nq, esc = _dados(X, y, n_qubits or None)
    cfg = QuantumConfig(nq, "anel", 1)             # d > n_qubits: re-uploading automático
    reg = _registro() if chave_campeao else None
    orq = Orquestrador(candidatos_padrao(cfg, incluir_vqc=incluir_vqc, incluir_kernel=incluir_kernel,
                                         registro=reg, chave=chave_campeao or None),
                       repeticoes=2).fit(Xs, y_)
    mid = uuid.uuid4().hex[:12]
    _modelos[mid] = (orq, esc)
    return {"modelo_id": mid, "relatorio": orq.relatorio, "texto": orq.texto()}


@servidor.tool()
@_guarda.proteger
def evoluir_circuito(X: list[list[float]], y: list[int], chave: str, n_qubits: int = 0,
                     geracoes: int = 4, populacao: int = 8, modo: str = "reservatorio",
                     motor: str = "quantico") -> dict:
    """Auto-evolução do circuito do problema `chave`. motor="quantico": a população
    de arquiteturas vive em superposição num circuito gerador que aprende (e
    emaranha) os genes bons; "classico": algoritmo genético. O campeão só é
    promovido se a melhora for comprovada."""
    from .evolucao import BuscaEvolutiva
    from .evolucao_quantica import EvolucaoQuantica
    if motor not in ("quantico", "classico"):
        raise ValueError("motor deve ser 'quantico' ou 'classico'")
    Xs, y_, nq, _ = _dados(X, y, n_qubits or None)
    classe = EvolucaoQuantica if motor == "quantico" else BuscaEvolutiva
    busca = classe(nq, populacao=populacao, geracoes=geracoes, modo=modo, registro=_registro(), chave=chave)
    res = busca.executar(Xs, y_)
    m = res["melhor"]
    out = {"motor": motor, "promovido": res["promovido"], "motivo": res["motivo"],
           "melhor": {"descricao": m["descricao"], "acuracia_cv": m["acuracia"], "portas_2q": m["custo"]},
           "padrao_quirios": {"acuracia_cv": res["padrao"]["acuracia"], "portas_2q": res["padrao"]["custo"]},
           "genomas_avaliados": res["avaliados"], "tempo_s": res["tempo_s"]}
    if motor == "quantico":
        out["resumo"] = busca.resumo(res)
    return out


@servidor.tool()
@_guarda.proteger
def comparar_dados_quanticos(n_spins: int = 6, n_amostras: int = 120, shots: int = 10,
                             h_min: float = 0.85, h_max: float = 1.2) -> dict:
    """Benchmark com dados quânticos: classificar a fase (ferro/paramagnética) de
    sistemas de Ising. Clássicos veem medidas com `shots` repetições; quânticos
    recebem o estado. Devolve acurácias e o teste pareado quântico vs clássico."""
    from .dados_quanticos import comparar, fases_ising, texto_comparacao
    if not 3 <= n_spins <= 10:
        raise ValueError("n_spins deve estar entre 3 e 10")
    res = comparar(fases_ising(n_spins, n_amostras, h_min=h_min, h_max=h_max, shots=shots),
                   n_treino=min(40, n_amostras // 3), repeticoes=3)
    return {**res, "texto": texto_comparacao(res)}


@servidor.tool()
@_guarda.proteger
def campeoes() -> dict:
    """Campeões registrados pela auto-evolução, por problema."""
    reg = _registro()
    return {ch: {k: reg.campeao(ch).get(k) for k in ("descricao", "acuracia", "custo", "data", "motivo", "motor")}
            for ch in reg.chaves() if reg.campeao(ch)}


@servidor.tool()
@_guarda.proteger
def projetar_hardware(X: list[list[float]], y: list[int], n_qubits: int = 0,
                      geracoes: list[str] | None = None, shots: int = 2000) -> dict:
    """Simula o circuito padrão em gerações de hardware (hoje, próxima, 10x, 100x,
    tolerante a falhas) e mede erro das features e acurácia em cada uma."""
    from .ansatz import Ansatz
    from .futuro import projetar_hardware as projetar
    Xs, y_, nq, _ = _dados(X, y, n_qubits or None)
    ans = Ansatz(QuantumConfig(nq, "anel", 1), n_features=Xs.shape[1])
    return {"linhas": projetar(Xs, y_, ans, ans.params_iniciais(0), geracoes=geracoes, shots=shots)}


@servidor.tool()
@_guarda.proteger
def diagnosticar_kernel(X: list[list[float]], y: list[int], n_qubits: int = 0,
                        tipo: str = "projetado") -> dict:
    """Alinhamento kernel-alvo e concentração exponencial de um kernel quântico."""
    from .kernels import KernelQuantico, alinhamento, concentracao
    Xs, y_, nq, _ = _dados(X, y, n_qubits or None)
    K = KernelQuantico(QuantumConfig(nq, "anel", 1), tipo=tipo).matriz(Xs)
    return {"tipo": tipo, "alinhamento": alinhamento(K, y_), **concentracao(K)}


def main():
    servidor.run("stdio")


if __name__ == "__main__":
    main()
