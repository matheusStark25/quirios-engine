"""
Sistema QURIOS: geradores de features plugáveis + seleção automática.

O classificador final é sempre uma regressão logística L2 sobre
[dados brutos] + [features do gerador]. O sistema compara os geradores por
validação cruzada SÓ nos dados de treino e escolhe o melhor -- o conjunto de
teste nunca participa da escolha.

Geradores:
    Brutos()                      -- só os dados originais (baseline clássico)
    RFF(escala)                   -- features aleatórias clássicas cos(Wx+b)
    Quantico(config, backend)     -- mapa de features do circuito QURIOS
                                     (circuito fixo: 1 execução por amostra)

Resultados de referência (Breast Cancer, N=8, 20 sorteios):
  quântico vs mesmo modelo sem ele: +5,3pp com 10 exemplos (p=0,005);
  quântico vs RFF com escala ótima: sem diferença significativa.
"""
import numpy as np
from .ansatz import Ansatz
from .vqc import logistica_l2, _sig
from .backends import BackendExato


class Brutos:
    nome = "brutos"
    quantico = False
    def preparar(self, X, y=None): return self
    def transformar(self, X): return np.zeros((len(X), 0))


class RFF:
    quantico = False

    def __init__(self, n_features=16, escala=1.0, seed=0):
        self.n_features, self.escala, self.seed = n_features, escala, seed
        self.nome = f"rff(escala={escala:g})"
    def preparar(self, X, y=None):
        r = np.random.default_rng(self.seed)
        self.W = r.normal(0, self.escala, (X.shape[1], self.n_features))
        self.b = r.uniform(0, 2 * np.pi, self.n_features)
        return self
    def transformar(self, X): return np.cos(X @ self.W + self.b)


class Quantico:
    """Circuito fixo (modo reservatório): sem treino por gradiente, que nos
    testes não superou o circuito fixo e custa ~49x mais simulações.

    Com mais features que qubits, o Ansatz distribui os dados em camadas
    (data re-uploading): n_camadas=None usa ceil(d / n_qubits).

    genes (rotacoes, emaranhador, leitura, camada_pos) definem o circuito;
    Quantico.de_genoma(...) monta um a partir de um genoma da busca evolutiva."""
    quantico = True

    def __init__(self, config, backend=None, seed=0, escala_alpha=np.pi / 2,
                 n_camadas=None, codificacao="angulo", nome=None, repeticoes=1, **genes):
        self.config, self.backend = config, backend or BackendExato()
        self.seed, self.escala_alpha = seed, escala_alpha
        self.n_camadas, self.codificacao, self.genes = n_camadas, codificacao, genes
        self.repeticoes = repeticoes
        self.nome = nome or f"quantico[{self.backend.nome}]"
        self._cache = {}

    @classmethod
    def de_genoma(cls, genoma, n_qubits, backend=None, seed=0, nome=None):
        from .evolucao import config_do_genoma
        g = dict(genoma)
        return cls(config_do_genoma(g, n_qubits), backend, seed=seed,
                   escala_alpha=g["escala_alpha"], repeticoes=g["repeticoes_dados"],
                   codificacao=g["codificacao"], nome=nome, rotacoes=tuple(g["rotacoes"]),
                   emaranhador=g["emaranhador"], leitura=g["leitura"], camada_pos=g["camada_pos"])

    def preparar(self, X, y=None):
        L = self.n_camadas
        if L is None and self.repeticoes > 1:      # reenvia cada bloco de features `repeticoes` vezes
            base = 1 if self.codificacao == "amplitude" else -(-X.shape[1] // self.config.n_qubits)
            L = base * self.repeticoes
        self.ansatz = Ansatz(self.config, n_features=X.shape[1], n_camadas=L,
                             codificacao=self.codificacao, **self.genes)
        self.params = self.ansatz.params_iniciais(self.seed, escala_alpha=self.escala_alpha)
        self._cache = {}
        return self

    def transformar(self, X):
        chave = X.tobytes()
        if chave not in self._cache:      # evita reexecutar circuitos (importante em hardware real)
            self._cache[chave] = self.backend.features(X, self.ansatz, self.params)
        return self._cache[chave]


def _ajustar(Z, y, C):
    mu, sd = Z.mean(0), Z.std(0) + 1e-9
    w, b = logistica_l2((Z - mu) / sd, y, C)
    return mu, sd, w, b


def _folds(y, k, seed):
    r = np.random.default_rng(seed)
    dobras = [[] for _ in range(k)]
    for classe in np.unique(y):
        idx = r.permutation(np.where(y == classe)[0])
        for i, j in enumerate(idx):
            dobras[i % k].append(j)
    return [np.array(d) for d in dobras]


class ClassificadorQURIOS:
    def __init__(self, geradores, cv=5, C=1.0, seed=0):
        self.geradores, self.cv, self.C, self.seed = geradores, cv, C, seed
        self.relatorio_cv = {}

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        k = max(2, min(self.cv, int(np.bincount(y.astype(int)).min())))
        dobras = _folds(y, k, self.seed)
        for g in self.geradores:
            g.preparar(X, y)
            F = g.transformar(X)                     # calculado 1x, fatiado nas dobras
            Z = np.hstack([X, F])
            accs = []
            for i in range(k):
                te = dobras[i]; tr = np.concatenate([dobras[j] for j in range(k) if j != i])
                if len(np.unique(y[tr])) < 2: continue
                mu, sd, w, b = _ajustar(Z[tr], y[tr], self.C)
                accs.append(np.mean((_sig(((Z[te] - mu) / sd) @ w + b) > 0.5) == y[te]))
            self.relatorio_cv[g.nome] = (float(np.mean(accs)), float(np.std(accs)))
        # empate -> o mais simples (ordem da lista)
        self.escolhido = max(self.geradores, key=lambda g: (round(self.relatorio_cv[g.nome][0], 6),
                                                            -self.geradores.index(g)))
        Z = np.hstack([X, self.escolhido.transformar(X)])
        self.mu, self.sd, self.w, self.b = _ajustar(Z, y, self.C)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        Z = np.hstack([X, self.escolhido.transformar(X)])
        return _sig(((Z - self.mu) / self.sd) @ self.w + self.b)

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

    def relatorio(self):
        linhas = [f"  {nome:32s} CV = {m*100:5.1f}% ± {s*100:4.1f}" for nome, (m, s) in self.relatorio_cv.items()]
        return "Seleção por validação cruzada (só treino):\n" + "\n".join(linhas) + \
               f"\n  -> escolhido: {self.escolhido.nome}"
