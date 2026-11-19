"""
Orquestrador híbrido: decide, com evidência estatística, se uma tarefa de IA
roda no clássico ou no quântico -- e em qual backend.

    orq = Orquestrador([
        DeGerador(Brutos()),                                  # clássico: logística nos dados
        DeGerador(RFF(escala=1.0)),                           # clássico: features aleatórias
        DeGerador(Quantico(cfg)),                             # quântico: mapa de features
        DeModelo("vqc-spsa", lambda: VQCClassifier(cfg, otimizador="spsa"), quantico=True),
    ]).fit(X, y)
    print(orq.texto())        # relatório legível
    orq.relatorio             # o mesmo, como dict (JSON)
    orq.predict(X_novo)

Política "evidencia" (padrão): todos os candidatos são avaliados nas MESMAS
dobras de uma validação cruzada repetida (só no treino). Um candidato
quântico só é escolhido se vencer o MELHOR clássico com p < alfa no teste t
pareado corrigido de Nadeau-Bengio. Sem evidência, fica o melhor clássico:
o quântico precisa provar que ajuda.

Política "melhor_media": escolhe a maior média (como o ClassificadorQURIOS).

orcamento_circuitos limita quantos circuitos um candidato pode executar na
avaliação (importante para hardware real); quem estoura é pulado e aparece
no relatório.

escolher_backend(config) faz o roteamento: simulação vetorial exata, MPS
exato ou MPS truncado, conforme o tamanho e o entrelaçamento do circuito.
"""
import copy
import time

import numpy as np

from .backends import BackendExato, BackendMPS
from .core import MAX_QUBITS_VETORIAL
from .estatistica import comparar_pareado
from .sistema import _ajustar, _folds
from .tarefas import em_segundo_plano
from .vqc import _sig


# ── roteamento de backend ───────────────────────────────────────────────────

def _tem_aer():
    try:
        import qiskit_aer  # noqa: F401
        return True
    except ImportError:
        return False


def escolher_backend(config, permitir_aproximacao=True):
    """Devolve (backend, motivo) para um QuantumConfig."""
    n = config.n_qubits
    pouco_entrelacado = (config.topologia_nome in ("linear", "blocos")
                         or (config.topologia_nome == "anel" and config.n_camadas <= 1))
    if n <= 16:
        return BackendExato(), f"vetor de estado exato ({n} qubits <= 16)"
    if pouco_entrelacado and _tem_aer():
        return BackendMPS(), f"MPS exato: topologia '{config.topologia_nome}' de baixo entrelaçamento"
    if n <= MAX_QUBITS_VETORIAL:
        return BackendExato(), f"vetor de estado exato ({n} qubits <= {MAX_QUBITS_VETORIAL})"
    if not _tem_aer():
        raise ValueError(f"{n} qubits exigem BackendMPS: instale o extra [qiskit]")
    if permitir_aproximacao:
        return BackendMPS(max_bond=64), "MPS truncado (chi=64): aproximado, entrelaçamento alto"
    raise ValueError(f"{n} qubits com entrelaçamento alto: sem backend exato viável "
                     f"(use permitir_aproximacao=True ou topologia linear/blocos)")


# ── candidatos ──────────────────────────────────────────────────────────────

def _proba(modelo, X):
    p = np.asarray(modelo.predict_proba(X))
    return p[:, 1] if p.ndim == 2 else p


class DeGerador:
    """Gerador de features (Brutos, RFF, Quantico...) + regressão logística L2
    sobre [dados, features]. As features são calculadas UMA vez para todo o
    treino (geradores não olham y) e fatiadas nas dobras."""

    def __init__(self, gerador, C=1.0, nome=None):
        self.gerador, self.C = gerador, C
        self.nome = nome or gerador.nome
        self.quantico = bool(getattr(gerador, "quantico", False))

    def custo_estimado(self, n_amostras):
        return n_amostras if self.quantico else 0

    def _Z(self, X, y):
        chave = X.tobytes()
        if getattr(self, "_chave", None) != chave:   # avaliar + ajustar no mesmo X: 1 execução só
            self.gerador.preparar(X, y)
            self._chave, self._Zx = chave, np.hstack([X, self.gerador.transformar(X)])
        return self._Zx

    def avaliar(self, X, y, divisoes):
        Z = self._Z(X, y)
        accs = []
        for tr, te in divisoes:
            mu, sd, w, b = _ajustar(Z[tr], y[tr], self.C)
            accs.append(np.mean((_sig(((Z[te] - mu) / sd) @ w + b) > 0.5) == y[te]))
        return np.array(accs), self.custo_estimado(len(X))

    def ajustar(self, X, y):
        Z = self._Z(X, y)
        self.mu, self.sd, self.w, self.b = _ajustar(Z, y, self.C)
        return self

    def predict_proba(self, X):
        Z = np.hstack([X, self.gerador.transformar(X)])
        return _sig(((Z - self.mu) / self.sd) @ self.w + self.b)


class DeModelo:
    """Qualquer modelo com fit/predict_proba (VQCClassifier, scikit-learn...).
    fabrica() devolve um modelo novo, não treinado (um por dobra)."""

    def __init__(self, nome, fabrica, quantico=False, custo=None):
        self.nome, self.fabrica, self.quantico = nome, fabrica, quantico
        self._custo = custo

    def custo_estimado(self, n_amostras):
        if self._custo is None:
            return 0
        return self._custo(n_amostras) if callable(self._custo) else int(self._custo)

    def avaliar(self, X, y, divisoes):
        accs, circuitos = [], 0
        for tr, te in divisoes:
            m = self.fabrica().fit(X[tr], y[tr])
            accs.append(np.mean((_proba(m, X[te]) > 0.5) == y[te]))
            circuitos += getattr(m, "circuitos_executados", 0)
        return np.array(accs), circuitos

    def ajustar(self, X, y):
        self.modelo = self.fabrica().fit(X, y)
        return self

    def predict_proba(self, X):
        return _proba(self.modelo, X)


# ── orquestrador ────────────────────────────────────────────────────────────

def divisoes_cv(y, dobras=5, repeticoes=3, seed=0):
    """Validação cruzada estratificada repetida: [(treino, teste), ...] e k."""
    y = np.asarray(y, float)
    k = max(2, min(dobras, int(np.bincount(y.astype(int)).min())))
    out = []
    for r in range(repeticoes):
        partes = _folds(y, k, seed + r)
        for i in range(k):
            tr = np.concatenate([partes[j] for j in range(k) if j != i])
            if len(np.unique(y[tr])) == 2:
                out.append((tr, partes[i]))
    return out, k


class Orquestrador:
    def __init__(self, candidatos, politica="evidencia", alfa=0.05, dobras=5, repeticoes=3,
                 orcamento_circuitos=None, seed=0):
        if politica not in ("evidencia", "melhor_media"):
            raise ValueError("politica deve ser 'evidencia' ou 'melhor_media'")
        if not candidatos:
            raise ValueError("informe pelo menos um candidato")
        self.candidatos = candidatos
        self.politica, self.alfa = politica, alfa
        self.dobras, self.repeticoes = dobras, repeticoes
        self.orcamento_circuitos, self.seed = orcamento_circuitos, seed

    def _divisoes(self, y):
        return divisoes_cv(y, self.dobras, self.repeticoes, self.seed)

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        divisoes, k = self._divisoes(y)
        linhas, accs = [], {}
        for c in self.candidatos:
            linha = {"nome": c.nome, "quantico": c.quantico}
            estimado = c.custo_estimado(len(X))
            if self.orcamento_circuitos is not None and estimado > self.orcamento_circuitos:
                linha.update(status="pulado", motivo=f"orçamento: estimados {estimado} circuitos "
                                                     f"> {self.orcamento_circuitos}")
                linhas.append(linha)
                continue
            t0 = time.perf_counter()
            a, circuitos = c.avaliar(X, y, divisoes)
            accs[c.nome] = a
            linha.update(status="avaliado", acuracia=float(a.mean()), desvio=float(a.std()),
                         tempo_s=round(time.perf_counter() - t0, 3), circuitos=int(circuitos))
            linhas.append(linha)

        avaliados = [c for c in self.candidatos if c.nome in accs]
        if not avaliados:
            raise RuntimeError("nenhum candidato coube no orçamento")
        media = lambda c: (round(float(accs[c.nome].mean()), 6), -self.candidatos.index(c))
        classicos = [c for c in avaliados if not c.quantico]
        quanticos = [c for c in avaliados if c.quantico]
        comparacao = None
        if classicos and quanticos:
            mc, mq = max(classicos, key=media), max(quanticos, key=media)
            n_tr = len(divisoes[0][0]); n_te = len(divisoes[0][1])
            t = comparar_pareado(accs[mq.nome] - accs[mc.nome], correcao=n_te / n_tr)
            comparacao = {"melhor_classico": mc.nome, "melhor_quantico": mq.nome,
                          "ganho_pp": round(100 * t["ganho"], 2),
                          "ic95_pp": [round(100 * t["ic"][0], 2), round(100 * t["ic"][1], 2)],
                          "p_valor": round(t["p_valor"], 4),
                          "evidencia": bool(t["ganho"] > 0 and t["p_valor"] < self.alfa)}

        if self.politica == "melhor_media" or not (classicos and quanticos):
            escolhido = max(avaliados, key=media)
            motivo = "maior acurácia média na validação cruzada"
        elif comparacao["evidencia"]:
            escolhido = max(quanticos, key=media)
            motivo = (f"quântico venceu o melhor clássico com evidência: "
                      f"+{comparacao['ganho_pp']} pp, p={comparacao['p_valor']}")
        else:
            escolhido = max(classicos, key=media)
            motivo = (f"quântico não superou o clássico com evidência "
                      f"({comparacao['ganho_pp']:+} pp, p={comparacao['p_valor']}); "
                      f"mantido o clássico")

        self.escolhido = escolhido.ajustar(X, y)
        self.relatorio = {"politica": self.politica, "alfa": self.alfa, "n_amostras": len(X),
                          "dobras": k, "repeticoes": self.repeticoes, "candidatos": linhas,
                          "comparacao": comparacao, "escolhido": escolhido.nome,
                          "escolhido_quantico": escolhido.quantico, "motivo": motivo}
        return self

    def submeter(self, X, y):
        """fit em segundo plano; devolve uma Tarefa cujo resultado é o próprio orquestrador."""
        return em_segundo_plano(self.fit, X, y, origem="orquestrador")

    def predict_proba(self, X):
        if not hasattr(self, "escolhido"):
            raise RuntimeError("orquestrador não treinado: chame fit() antes")
        return self.escolhido.predict_proba(np.asarray(X, float))

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

    def texto(self):
        r = self.relatorio
        out = [f"Orquestrador QURIOS ({r['politica']}, {r['repeticoes']}x{r['dobras']} dobras, "
               f"{r['n_amostras']} amostras)"]
        for c in r["candidatos"]:
            tipo = "Q" if c["quantico"] else "C"
            if c["status"] == "avaliado":
                out.append(f"  [{tipo}] {c['nome']:28s} {100 * c['acuracia']:5.1f}% ± "
                           f"{100 * c['desvio']:4.1f}  {c['tempo_s']:7.2f}s  {c['circuitos']} circuitos")
            else:
                out.append(f"  [{tipo}] {c['nome']:28s} pulado -- {c['motivo']}")
        if r["comparacao"]:
            cp = r["comparacao"]
            out.append(f"  quântico vs clássico: {cp['ganho_pp']:+} pp "
                       f"(IC95 {cp['ic95_pp'][0]:+} a {cp['ic95_pp'][1]:+}), p={cp['p_valor']}")
        out.append(f"  -> escolhido: {r['escolhido']} ({r['motivo']})")
        return "\n".join(out)


def candidatos_padrao(config, backend=None, incluir_vqc=False, seed=0, incluir_kernel=False,
                      registro=None, chave=None):
    """Conjunto de partida: 2 clássicos + mapa quântico (+ VQC com SPSA, + kernel
    projetado). Com registro/chave, inclui o circuito CAMPEÃO da auto-evolução
    (evolucao.py): o melhor circuito já descoberto entra sem mudar código."""
    from .sistema import RFF, Brutos, Quantico
    from .vqc import VQCClassifier
    if backend is None:
        backend, _ = escolher_backend(config)
    c = [DeGerador(Brutos()), DeGerador(RFF(escala=1.0, seed=seed)),
         DeGerador(Quantico(config, backend, seed=seed))]
    if registro is not None and chave and registro.campeao(chave):
        g = registro.campeao(chave)["genoma"]
        c.append(DeGerador(Quantico.de_genoma(g, config.n_qubits, backend, seed=seed,
                                              nome="quantico[campeao]")))
    if incluir_kernel:
        from .kernels import ClassificadorKernel, KernelQuantico
        c.append(DeModelo("kernel-projetado", lambda: ClassificadorKernel(
            KernelQuantico(copy.deepcopy(config), backend=backend, seed=seed)), quantico=True))
    if incluir_vqc:
        c.append(DeModelo("vqc-spsa", lambda: VQCClassifier(copy.deepcopy(config), otimizador="spsa",
                                                            backend=backend, epocas=15, seed=seed),
                          quantico=True))
    return c
