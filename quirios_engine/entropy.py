"""
Duas métricas DIFERENTES, com nomes diferentes de propósito -- confundi-las
nas versões anteriores do QURIOS (v9-v12) foi um erro de nomenclatura real:

  entropia_medicao(state)
      Entropia de Shannon da distribuição de medição |amplitude|^2 do
      sistema INTEIRO. Mede "espalhamento"/superposição. É o que as
      versões anteriores chamavam de "entropia" -- válido e útil, mas
      NÃO é entrelaçamento. Para um estado puro, a entropia de Von
      Neumann do sistema completo é SEMPRE zero, por definição.

  entropia_emaranhamento(state, n_qubits, subsistema)
      Entropia de Von Neumann da matriz de densidade REDUZIDA (traço
      parcial sobre os qubits fora de 'subsistema'). Essa SIM mede
      entrelaçamento real entre partes do sistema. Validada contra o
      valor analítico conhecido: par de Bell = exatamente 1 bit,
      estado produto = exatamente 0.
"""
import numpy as np


def entropia_medicao(state):
    """Entropia de Shannon da distribuição de medição do sistema inteiro."""
    probs = np.abs(state) ** 2
    nz = probs[probs > 1e-15]
    return float(-np.sum(nz * np.log2(nz)))


def pureza(state):
    probs = np.abs(state) ** 2
    return float(np.sum(probs ** 2))


def matriz_reduzida(state, n_qubits, subsistema):
    """Matriz de densidade reduzida sobre `subsistema` (traço parcial do resto),
    direto do vetor de estado: custo O(2^n * 2^k), sem montar a matriz 4^n.

    Convenção LSB (qubit q <-> bit q), a mesma de apply_gate e dos CNOTs: o
    eixo do tensor de q é n-1-q. (Até a v0.6 a entropia usava a convenção
    oposta e, com 3+ qubits, media o qubit errado.)"""
    keep = list(subsistema)
    if len(set(keep)) != len(keep) or any(not 0 <= q < n_qubits for q in keep):
        raise ValueError(f"subsistema inválido: {subsistema}")
    T = np.asarray(state).reshape([2] * n_qubits)
    eixos = [n_qubits - 1 - q for q in keep]
    M = np.moveaxis(T, eixos, list(range(len(keep)))).reshape(2 ** len(keep), -1)
    return M @ M.conj().T


def entropia_emaranhamento(state, n_qubits, subsistema):
    """
    Entropia de Von Neumann da matriz de densidade reduzida sobre
    'subsistema' (lista de índices de qubit). Mede entrelaçamento real
    entre 'subsistema' e o resto do sistema.

    Validado contra valores analíticos conhecidos:
      - par de Bell: 1.000000 (máximo para 1 qubit)
      - estado produto (sem entrelaçamento): 0.00000000
    """
    autovalores = np.linalg.eigvalsh(matriz_reduzida(state, n_qubits, subsistema)).real
    autovalores = autovalores[autovalores > 1e-12]
    if len(autovalores) == 0:
        return 0.0
    return float(-np.sum(autovalores * np.log2(autovalores)))


def mutual_information(state, n_qubits, subsistema_a, subsistema_b):
    """
    I(A:B) = S(A) + S(B) - S(A∪B).
    Exige A e B disjuntos. Para o estado global puro, se A∪B = todo o
    sistema, S(A∪B) = 0 (estado puro completo) e I(A:B) = S(A) + S(B)
    = 2*S(A) (já que para um estado puro global, S(A) = S(B_complement)
    quando B = complemento de A).
    """
    if set(subsistema_a) & set(subsistema_b):
        raise ValueError("subsistema_a e subsistema_b devem ser disjuntos")
    S_a = entropia_emaranhamento(state, n_qubits, subsistema_a)
    S_b = entropia_emaranhamento(state, n_qubits, subsistema_b)
    uniao = sorted(set(subsistema_a) | set(subsistema_b))
    S_ab = entropia_emaranhamento(state, n_qubits, uniao)
    return S_a + S_b - S_ab
