"""
Circuito variacional treinável do QURIOS.

Arquitetura (v0.2):
    H, T -> Ry(pi*x) [dados] -> Ry(theta), Rz(phi) -> CNOTs -> Ry(alpha) -> mede TODOS os qubits
    p(y=1) = sigmoid(w . m + b), m_q = P(qubit q = 1)

Correções em relação à v0.1 (todas verificadas numericamente):
  1. Leitura de 1 qubit só "enxergava" os qubits no seu cone de luz
     (anel 1 camada: 2 de 8; linear: 1 de N; blocos N=16: 2 de 16).
     Agora todos os qubits são medidos e combinados com pesos.
  2. Sem rotação depois dos CNOTs, Rz(phi) não altera nenhuma medida
     (efeito 2e-16) e o circuito equivale exatamente a bits clássicos
     independentes combinados por XOR. A camada Ry(alpha) pós-entrelaçamento
     cria interferência real (phi passa a alterar as medidas em ~0.24).
  3. Os pesos (w, b) da leitura são resolvidos exatamente a cada época
     (regressão logística L2, convexa), pois as medidas variam pouco entre
     amostras (desvio ~0.01-0.04) e gradiente simples não chega aos pesos
     grandes necessários -- sem isso o modelo colapsa na classe majoritária.

Gradientes de theta/phi/alpha: parameter-shift exato (validado contra
diferença finita, diff ~1e-10).
"""
import json
from pathlib import Path

import numpy as np
from .ansatz import Ansatz
from .backends import BackendExato, BackendMPS
from .core import apply_gate, apply_perm, gate_H, gate_T, gate_Ry, gate_Rz
from .entropy import entropia_medicao, pureza
from .otimizadores import SPSA, ParameterShift, criar_otimizador


# ── circuito ────────────────────────────────────────────────────────────────

def estado(x, theta, phi, alpha, config):
    n = config.n_qubits
    st = np.zeros(config.dim, dtype=complex)
    st[0] = 1.0
    H, T = gate_H(), gate_T()
    for q in range(n): apply_gate(st, H, q, n)
    for q in range(n): apply_gate(st, T, q, n)
    for q in range(n): apply_gate(st, gate_Ry(np.pi * x[q]), q, n)
    for q in range(n): apply_gate(st, gate_Ry(theta[q]), q, n)
    for q in range(n): apply_gate(st, gate_Rz(phi[q]), q, n)
    for camada in config.camadas_cnot:
        apply_perm(st, camada)
    if alpha is not None:
        for q in range(n): apply_gate(st, gate_Ry(alpha[q]), q, n)
    return st


def marginais(state, n):
    """m_q = P(qubit q = 1), convenção LSB (qubit q <-> bit q)."""
    pr = (np.abs(state) ** 2).reshape([2] * n)
    out = np.empty(n)
    for q in range(n):
        eixos = tuple(a for a in range(n) if a != n - 1 - q)
        out[q] = pr.sum(axis=eixos)[1]
    return out


def _marg_lento(x, th, ph, al, config):
    return marginais(estado(x, th, ph, al, config), config.n_qubits)


_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_T = np.diag([1, np.exp(1j * np.pi / 4)])


def _rhos(x, th, ph, config):
    """Matrizes reduzidas 2x2 de cada qubit após os CNOTs (antes de alpha).
    Antes dos CNOTs o estado é produto -> montado por Kronecker (exato)."""
    n = config.n_qubits
    st = np.ones(1, dtype=complex)
    for q in range(n - 1, -1, -1):                     # qubit n-1 = bit mais significativo
        v = _T @ (_H @ np.array([1, 0], dtype=complex))
        v = gate_Ry(np.pi * x[q] + th[q]) @ v           # Ry(a)Ry(b) = Ry(a+b)
        v = gate_Rz(ph[q]) @ v
        st = np.kron(st, v)
    for camada in config.camadas_cnot:
        apply_perm(st, camada)
    T = st.reshape([2] * n)
    rhos = np.empty((n, 2, 2), dtype=complex)
    for q in range(n):
        v = np.moveaxis(T, n - 1 - q, 0).reshape(2, -1)
        rhos[q] = v @ v.conj().T
    return rhos


def _m_de_rhos(rhos, al):
    if al is None:
        return rhos[:, 1, 1].real.copy()
    out = np.empty(len(rhos))
    for q in range(len(rhos)):
        R = gate_Ry(al[q])
        out[q] = (R @ rhos[q] @ R.conj().T)[1, 1].real
    return out


def _marg(x, th, ph, al, config):
    """Caminho rápido e exato (validado contra _marg_lento, diff ~1e-15)."""
    return _m_de_rhos(_rhos(x, th, ph, config), al)


# ── leitura: regressão logística L2 exata (Newton), sem dependência externa ──

def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def logistica_l2(Z, y, C=1.0, iters=50):
    """Minimiza 0.5*||w||^2 + C*sum(logloss), intercepto sem penalidade
    (mesmo objetivo do sklearn LogisticRegression(C=C))."""
    n, d = Z.shape
    A = np.hstack([Z, np.ones((n, 1))])
    beta = np.zeros(d + 1)
    reg = np.ones(d + 1) / C
    reg[-1] = 0.0
    for _ in range(iters):
        p = _sig(A @ beta)
        g = A.T @ (p - y) + reg * beta
        Hm = (A * (p * (1 - p))[:, None]).T @ A + np.diag(reg) + 1e-10 * np.eye(d + 1)
        passo = np.linalg.solve(Hm, g)
        beta -= passo
        if np.abs(passo).max() < 1e-10:
            break
    return beta[:-1], beta[-1]


def ajustar_leitura(M, y, C=1.0):
    mu, sd = M.mean(0), M.std(0) + 1e-9
    w_s, b_s = logistica_l2((M - mu) / sd, y, C)
    return w_s / sd, b_s - np.sum(w_s * mu / sd)


# ── gradiente ───────────────────────────────────────────────────────────────

def loss_e_grad(x, y, theta, phi, alpha, w, b, config):
    """Perda BCE e gradiente exato (parameter-shift) de theta, phi, alpha.
    Deslocamentos de alpha reaproveitam as matrizes reduzidas do ponto base."""
    n = config.n_qubits
    rhos0 = _rhos(x, theta, phi, config)
    m = _m_de_rhos(rhos0, alpha)
    p = _sig(w @ m + b)
    pc = min(max(p, 1e-9), 1 - 1e-9)
    loss = -(y * np.log(pc) + (1 - y) * np.log(1 - pc))
    dLdm = (p - y) * w
    g_t, g_p = np.zeros(n), np.zeros(n)
    s2 = np.pi / 2
    for i in range(n):
        tp, tm = theta.copy(), theta.copy(); tp[i] += s2; tm[i] -= s2
        g_t[i] = dLdm @ ((_m_de_rhos(_rhos(x, tp, phi, config), alpha)
                          - _m_de_rhos(_rhos(x, tm, phi, config), alpha)) / 2)
        if alpha is not None:
            pp, pm = phi.copy(), phi.copy(); pp[i] += s2; pm[i] -= s2
            g_p[i] = dLdm @ ((_m_de_rhos(_rhos(x, theta, pp, config), alpha)
                              - _m_de_rhos(_rhos(x, theta, pm, config), alpha)) / 2)
    grads = [g_t, g_p]
    if alpha is not None:
        g_a = np.zeros(n)
        for i in range(n):          # alpha_i só afeta m_i
            R_p, R_m = gate_Ry(alpha[i] + s2), gate_Ry(alpha[i] - s2)
            mp = (R_p @ rhos0[i] @ R_p.conj().T)[1, 1].real
            mm = (R_m @ rhos0[i] @ R_m.conj().T)[1, 1].real
            g_a[i] = dLdm[i] * (mp - mm) / 2
        grads.append(g_a)
    return loss, grads


# ── diagnóstico ─────────────────────────────────────────────────────────────

def cone_de_luz(config, readout_qubit=0):
    """Qubits cuja feature afeta a medida de UM qubit (sem camada pós-CNOT).
    Útil para mostrar por que leitura de 1 qubit desperdiça features."""
    n = config.n_qubits
    rng = np.random.default_rng(0)
    th = rng.uniform(-1, 1, n)
    vivos = set()
    for _ in range(3):
        x = rng.uniform(0, 1, n)
        base = _marg(x, th, np.zeros(n), None, config)[readout_qubit]
        for q in range(n):
            x2 = x.copy(); x2[q] = (x2[q] + 0.37) % 1.0
            if abs(_marg(x2, th, np.zeros(n), None, config)[readout_qubit] - base) > 1e-12:
                vivos.add(q)
    return sorted(vivos)


# ── classificador ──────────────────────────────────────────────────────────

class VQCClassifier:
    """Interface sklearn-like: fit / predict / predict_proba.

    camada_pos=True (padrão): camada Ry(alpha) após o entrelaçamento -- o
    circuito tem interferência quântica real. camada_pos=False reproduz o
    circuito equivalente a bits clássicos + XOR (útil como controle).

    Escala:
      backend             None = simulação exata (até 24 qubits); BackendMPS
                          para 50-100+ qubits; BackendQiskit para shots, ruído
                          ou hardware IBM.
      otimizador          "parameter_shift" (gradiente exato, 2 circuitos por
                          parâmetro) ou "spsa" (2 circuitos por passo, qualquer
                          número de parâmetros). Aceita instâncias de
                          ParameterShift/SPSA com hiperparâmetros próprios.
      n_camadas           camadas de dados (re-uploading); None = ceil(d / n_qubits).
      codificacao         "angulo", "iqp" ou "amplitude" (d features em log2(d+1) qubits).
      rotacoes, emaranhador, leitura
                          genes do circuito (ver ansatz.py). leitura="z" (padrão)
                          usa P(qubit=1) de cada qubit; "pauli" usa <X>, <Y>, <Z>.
      inicializacao       "padrao" ou "pequena" (mitiga barren plateau).
      treino_por_camadas  treina uma camada de cada vez, depois todas juntas.
      lote                amostras por passo (None: todas; com SPSA: 16).

    Com simulação exata, genoma padrão de 1 camada, d = n_qubits e
    parameter-shift, o treino usa o caminho rápido da v0.2 (resultados
    idênticos aos de antes). circuitos_executados conta execuções de circuito
    equivalentes em hardware.
    """

    def __init__(self, config, lr=0.3, epocas=25, paciencia=6, camada_pos=True,
                 escala_alpha=np.pi / 2, C=1.0, seed=None, backend=None,
                 otimizador="parameter_shift", n_camadas=None, codificacao="angulo",
                 inicializacao="padrao", treino_por_camadas=False, lote=None,
                 rotacoes=("ry", "rz"), emaranhador="cnot", leitura="z", constante_amplitude=1.0,
                 cronograma_ising=None):
        self.config = config
        self.lr = lr
        self.epocas = epocas
        self.paciencia = paciencia
        self.camada_pos = camada_pos
        self.escala_alpha = escala_alpha
        self.C = C
        self.seed = seed
        self.backend = backend
        self.otimizador = otimizador
        self.n_camadas = n_camadas
        self.codificacao = codificacao
        self.inicializacao = inicializacao
        self.treino_por_camadas = treino_por_camadas
        self.lote = lote
        self.rotacoes, self.emaranhador, self.leitura = tuple(rotacoes), emaranhador, leitura
        self.constante_amplitude = constante_amplitude      # None: dados já são estados quânticos
        self.cronograma_ising = cronograma_ising            # codificacao="ising" (ansatz.py)
        self.historico_loss = []
        self.circuitos_executados = 0

    # ── parâmetros no formato da v0.2-v0.4 (theta/phi/alpha) ───────────────

    @property
    def theta(self):
        t = self.ansatz.desempacotar(self.params)[0]
        return t[0] if self.ansatz.n_camadas == 1 else t

    @property
    def phi(self):
        p = self.ansatz.desempacotar(self.params)[1]
        return p[0] if self.ansatz.n_camadas == 1 else p

    @property
    def alpha(self):
        return self.ansatz.desempacotar(self.params)[2]

    # ── avaliação do circuito ───────────────────────────────────────────────

    def _vetorial(self):
        return self.backend is None or isinstance(self.backend, BackendExato)

    def _rapido(self):
        a = self.ansatz
        return (self._vetorial() and a.padrao and a.leitura == "z" and a.n_camadas == 1
                and a.n_features == a.n_qubits)

    def _M(self, X, params=None):
        """Features da leitura de cada amostra (amostras x ansatz.n_saida);
        na leitura padrão "z", as marginais P(qubit q = 1)."""
        params = self.params if params is None else params
        X = np.asarray(X, float)
        self.circuitos_executados += len(X)
        if self._rapido():
            th, ph, al = self.ansatz.desempacotar(params)
            return np.array([_marg(x, th[0], ph[0], al, self.config) for x in X])
        if self._vetorial():
            return np.array([self.ansatz.medir(self.ansatz.estado(x, params)) for x in X])
        return self.backend.features(X, self.ansatz, params)

    def _perda(self, params, X, y):
        p = np.clip(_sig(self._M(X, params) @ self.w + self.b), 1e-9, 1 - 1e-9)
        return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))

    def _grad_shift(self, params, X, y, mascara):
        """Parameter-shift para qualquer ansatz/backend: 2 avaliações do lote
        por parâmetro treinável, regra da cadeia pela leitura logística."""
        p = _sig(self._M(X, params) @ self.w + self.b)
        pc = np.clip(p, 1e-9, 1 - 1e-9)
        perda = float(np.mean(-(y * np.log(pc) + (1 - y) * np.log(1 - pc))))
        dLdm = (p - y)[:, None] * self.w[None, :]
        g = np.zeros_like(params)
        for i in np.flatnonzero(mascara):
            e = np.zeros_like(params); e[i] = np.pi / 2
            dM = (self._M(X, params + e) - self._M(X, params - e)) / 2
            g[i] = np.mean(np.sum(dLdm * dM, axis=1))
        return g, perda

    def _passo_rapido(self, params, X, y):
        """Caminho da v0.2: matrizes reduzidas + parameter-shift analítico."""
        th, ph, al = self.ansatz.desempacotar(params)
        th, ph = th[0], ph[0]
        acc, perda = None, 0.0
        for xi, yi in zip(X, y):
            L, gs = loss_e_grad(xi, yi, th, ph, al, self.w, self.b, self.config)
            perda += L
            acc = gs if acc is None else [a + g for a, g in zip(acc, gs)]
        th = th - self.lr * acc[0] / len(X)
        if self.camada_pos:
            ph = ph - self.lr * acc[1] / len(X)
            al = al - self.lr * acc[2] / len(X)
        n = self.config.n_qubits
        self.circuitos_executados += len(X) * (1 + 2 * (3 * n if self.camada_pos else n))
        return self.ansatz.empacotar(th, ph, al), perda / len(X)

    def _estagios(self):
        """[(épocas, máscara de parâmetros treináveis), ...]"""
        a = self.ansatz
        tudo = np.ones(a.n_params, dtype=bool)
        if not self.camada_pos and a.n_camadas == 1 and a.padrao:
            tudo[a.n_qubits:2 * a.n_qubits] = False     # phi sem efeito (equivalência clássica)
        if not self.treino_por_camadas or a.n_camadas == 1:
            return [(self.epocas, tudo)]
        por = max(1, self.epocas // (a.n_camadas + 1))
        est = [(por, a.mascara_camada(l) & tudo) for l in range(a.n_camadas)]
        if self.epocas - por * a.n_camadas > 0:
            est.append((self.epocas - por * a.n_camadas, tudo))
        return est

    # ── treino ──────────────────────────────────────────────────────────────

    def fit(self, X, y, X_val=None, y_val=None, verbose=False):
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.ansatz = Ansatz(self.config, n_features=X.shape[1], n_camadas=self.n_camadas,
                             codificacao=self.codificacao, camada_pos=self.camada_pos,
                             rotacoes=self.rotacoes, emaranhador=self.emaranhador, leitura=self.leitura,
                             constante_amplitude=self.constante_amplitude,
                             cronograma_ising=self.cronograma_ising)
        params = self.ansatz.params_iniciais(self.seed, self.inicializacao, self.escala_alpha)
        if not self.camada_pos and self.ansatz.n_camadas == 1 and self.ansatz.padrao:
            n = self.config.n_qubits
            params[n:2 * n] = 0.0      # sem camada pós, phi não tem efeito -- não treinar
        self.params = params
        self.historico_loss, self.circuitos_executados = [], 0
        self._opt = criar_otimizador(self.otimizador, self.lr, self.seed)
        spsa = isinstance(self._opt, SPSA)
        rapido = self._rapido() and isinstance(self._opt, ParameterShift) and not self.treino_por_camadas
        lote = len(X) if self.lote is None and not spsa else min(len(X), self.lote or 16)
        rng = np.random.default_rng(self.seed)
        estagios = self._estagios()

        self.w, self.b = ajustar_leitura(self._M(X), y, self.C)
        if spsa:
            cal = rng.permutation(len(X))[:lote]
            self._opt.iniciar(lambda p: self._perda(p, X[cal], y[cal]), self.params, estagios[0][1],
                              sum(e for e, _ in estagios) * -(-len(X) // lote))

        melhor, melhor_loss, sem, parar, ep = None, np.inf, 0, False, 0
        for n_ep, mascara in estagios:
            for _ in range(n_ep):
                ordem = rng.permutation(len(X)) if lote < len(X) else np.arange(len(X))
                perdas = []
                for i0 in range(0, len(X), lote):
                    idx = ordem[i0:i0 + lote]
                    Xb, yb = X[idx], y[idx]
                    if rapido:
                        self.params, L = self._passo_rapido(self.params, Xb, yb)
                    elif spsa:
                        self.params, L = self._opt.passo(lambda p: self._perda(p, Xb, yb),
                                                         self.params, mascara)
                    else:
                        g, L = self._grad_shift(self.params, Xb, yb, mascara)
                        self.params = self._opt.aplicar(self.params, g, mascara)
                    perdas.append(L)
                self.w, self.b = ajustar_leitura(self._M(X), y, self.C)
                self.historico_loss.append(float(np.mean(perdas)))
                if verbose:
                    print(f"    época {ep:2d} | loss={self.historico_loss[-1]:.4f} "
                          f"| circuitos={self.circuitos_executados}")
                ep += 1
                if X_val is not None:
                    pv = np.clip(self.predict_proba(X_val), 1e-9, 1 - 1e-9)
                    yv = np.asarray(y_val, float)
                    lv = float(np.mean(-(yv * np.log(pv) + (1 - yv) * np.log(1 - pv))))
                    if lv < melhor_loss - 1e-4:
                        melhor_loss, sem = lv, 0
                        melhor = (self.params.copy(), self.w.copy(), self.b)
                    else:
                        sem += 1
                        if sem >= self.paciencia:
                            parar = True
                            break
            if parar:
                break
        if melhor is not None:
            self.params, self.w, self.b = melhor
        return self

    def _checar_treinado(self):
        if not hasattr(self, "w"):
            raise RuntimeError("modelo não treinado: chame fit() antes")

    def predict_proba(self, X):
        self._checar_treinado()
        M = self._M(np.asarray(X, float))
        return _sig(M @ self.w + self.b)

    def predict(self, X, limiar=0.5):
        return (self.predict_proba(X) > limiar).astype(int)

    # ── persistência ────────────────────────────────────────────────────────

    def salvar(self, caminho, assinador=None):
        """Salva o modelo treinado em JSON (sem pickle: seguro de abrir)."""
        Path(caminho).write_text(self.para_json(), encoding="utf-8")
        if assinador is not None:                    # ML-DSA (seguranca.py): <caminho>.sig
            assinador.assinar_arquivo(caminho)

    def para_json(self):
        """O modelo treinado como texto JSON (o que salvar() grava; protecao.Cofre cifra)."""
        from . import __version__
        self._checar_treinado()
        nome_opt = self.otimizador if isinstance(self.otimizador, str) else self.otimizador.nome
        d = {"formato": "quirios.VQCClassifier", "versao": __version__,
             "ansatz": self.ansatz.para_dict(), "params": self.params.tolist(),
             "w": self.w.tolist(), "b": float(self.b),
             "hiper": {"lr": self.lr, "epocas": self.epocas, "paciencia": self.paciencia,
                       "camada_pos": self.camada_pos, "escala_alpha": self.escala_alpha,
                       "C": self.C, "seed": self.seed, "otimizador": nome_opt,
                       "n_camadas": self.n_camadas, "codificacao": self.codificacao,
                       "inicializacao": self.inicializacao,
                       "treino_por_camadas": self.treino_por_camadas, "lote": self.lote,
                       "rotacoes": list(self.rotacoes), "emaranhador": self.emaranhador,
                       "leitura": self.leitura, "constante_amplitude": self.constante_amplitude,
                       "cronograma_ising": self.cronograma_ising},
             "backend": None if self.backend is None else self.backend.nome,
             "historico_loss": self.historico_loss,
             "circuitos_executados": self.circuitos_executados}
        return json.dumps(d, indent=2, ensure_ascii=False)

    @classmethod
    def carregar(cls, caminho, backend=None, verificador=None, conteudo=None):
        """Carrega um modelo salvo. Modelos de backends Qiskit/IBM precisam
        que o backend (com token) seja passado de novo. Com verificador, só
        carrega se a assinatura pós-quântica bater. `conteudo` (texto JSON já
        decifrado) dispensa ler `caminho`."""
        if verificador is not None and not verificador.verificar_arquivo(caminho):
            raise ValueError(f"assinatura pós-quântica inválida ou ausente em {caminho}")
        d = json.loads(conteudo if conteudo is not None else Path(caminho).read_text(encoding="utf-8"))
        if d.get("formato") != "quirios.VQCClassifier":
            raise ValueError(f"{caminho} não é um VQCClassifier salvo pelo QURIOS")
        if backend is None and d["backend"] not in (None, "exato"):
            if d["backend"] == "mps":
                backend = BackendMPS()
            else:
                raise ValueError(f"modelo treinado com o backend '{d['backend']}': "
                                 f"passe backend=... ao carregar")
        ans = Ansatz.de_dict(d["ansatz"])
        clf = cls(ans.config, backend=backend, **d["hiper"])
        clf.ansatz, clf.params = ans, np.asarray(d["params"], float)
        clf.w, clf.b = np.asarray(d["w"], float), float(d["b"])
        clf.historico_loss = d["historico_loss"]
        clf.circuitos_executados = d.get("circuitos_executados", 0)
        return clf

    # ── estado e features ───────────────────────────────────────────────────

    def estado_quantico(self, x):
        """Vetor de estado completo (2^n amplitudes) do circuito treinado,
        incluindo a camada pós-entrelaçamento. Use com entropia_emaranhamento
        e mutual_information para medir o entrelaçamento do modelo real."""
        self._checar_treinado()
        return self.ansatz.estado(np.asarray(x, float), self.params)

    def extrair_features(self, X):
        """Features quânticas para alimentar um classificador clássico.

        Colunas: [P(qubit q = 1) para cada q] + [entropia_medicao, pureza].
        Shape: (n_amostras, n_qubits + 2). Usa o estado completo (2^n)."""
        n = self.config.n_qubits
        out = []
        for x in np.asarray(X, float):
            st = self.estado_quantico(x)
            out.append(np.concatenate([marginais(st, n), [entropia_medicao(st), pureza(st)]]))
        return np.array(out)


# Compatibilidade com v0.1 (leitura de 1 qubit, sem camada pós) -- mantida
# só para reproduzir resultados antigos.
def circuito_treinavel(x, theta, phi, config, readout_qubit=0):
    st = estado(x, theta, phi, None, config)
    return float(marginais(st, config.n_qubits)[readout_qubit]), st


def grad_parameter_shift(x, y, theta, phi, config, readout_qubit=0):
    n = config.n_qubits
    p1, _ = circuito_treinavel(x, theta, phi, config, readout_qubit)
    p1c = min(max(p1, 1e-9), 1 - 1e-9)
    dL = -(y / p1c - (1 - y) / (1 - p1c))
    g_t, g_p = np.zeros(n), np.zeros(n)
    for i in range(n):
        tp, tm = theta.copy(), theta.copy(); tp[i] += np.pi / 2; tm[i] -= np.pi / 2
        g_t[i] = dL * (circuito_treinavel(x, tp, phi, config, readout_qubit)[0]
                       - circuito_treinavel(x, tm, phi, config, readout_qubit)[0]) / 2
    loss = -(y * np.log(p1c) + (1 - y) * np.log(1 - p1c))
    return loss, g_t, g_p, p1
