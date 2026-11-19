"""
Evolução QUÂNTICA: o próprio circuito quântico conduz a evolução.

A população de circuitos candidatos não é uma lista num computador clássico:
ela vive em SUPERPOSIÇÃO dentro de um circuito gerador, com um qubit por bit
do genoma (16 qubits no espaço padrão: as 15.552 arquiteturas possíveis ao
mesmo tempo). Cada etapa da evolução é uma operação quântica:

    população   |psi(theta)> = sum_g a_g |g>, superposição de TODOS os genomas
    indivíduo   MEDIR o gerador colapsa a superposição num genoma g com
                probabilidade |a_g|^2 (regra de Born)
    mutação     rotações aleatórias pequenas nos ângulos do gerador antes de medir
    seleção     os melhores genomas (aptidão = validação cruzada, a mesma da
                BuscaEvolutiva) formam a elite
    herança     o gerador é ajustado -- gradiente exato por parameter-shift --
                para aumentar a amplitude da elite: o circuito aprende onde
                estão os bons circuitos e passa a sorteá-los mais
    ligação     portas RZZ entre os qubits do gerador EMARANHAM genes: o
                circuito aprende quais escolhas funcionam juntas. A informação
                mútua quântica entre os registros de dois genes mede essa
                ligação aprendida (emaranhamento=False desliga e vira o
                algoritmo evolutivo de Han & Kim, sem ligação entre genes)

O gerador pode ser medido em qualquer backend com amostragem -- simulação
exata (padrão), Aer com ruído ou a QPU da IBM (backend_gerador=
BackendQiskit("ibm")): a população é sorteada num processador quântico de
verdade. O ajuste dos ângulos usa o vetor de estado (exato até ~20 qubits).

O campeão só é promovido com a regra da BuscaEvolutiva: evidência
estatística, ou economia de portas sem perda de acurácia.

Base: algoritmos evolutivos com Q-bits e portas de rotação (Han & Kim, IEEE
Trans. Evol. Comput. 2002), otimização guiada por modelos geradores -- GEO,
Alcazar, Ghazi Vakili, Kalayci & Perdomo-Ortiz, Nat. Commun. 2024 -- e
geradores quânticos de Born (QCBM); o emaranhamento entre genes generaliza o
"emaranhamento artificial" de Choy, Nguyen & Thawonmas.
"""
import time

import numpy as np

from .core import apply_gate, gate_Ry
from .entropy import entropia_emaranhamento
from .evolucao import (ESPACO_PADRAO, GENOMA_PADRAO, BuscaEvolutiva, chave_genoma, descrever,
                       normalizar)


_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)


def _sinal(idx, q):
    return 1 - 2 * ((idx >> q) & 1)


class CodificacaoGenoma:
    """Genoma <-> bits: um gene com k opções usa ceil(log2 k) qubits; o valor
    lido nos bits, módulo k, escolhe a opção."""

    def __init__(self, espaco):
        self.espaco = {k: tuple(v) for k, v in espaco.items()}
        self.genes, i = [], 0
        for nome, opcoes in self.espaco.items():
            b = int(np.ceil(np.log2(len(opcoes)))) if len(opcoes) > 1 else 0
            self.genes.append((nome, opcoes, list(range(i, i + b))))
            i += b
        self.n_bits = i

    @property
    def n_genomas(self):
        return int(np.prod([len(o) for _, o, _ in self.genes]))

    def decodificar(self, x):
        g = {}
        for nome, opcoes, qs in self.genes:
            v = sum(((int(x) >> q) & 1) << k for k, q in enumerate(qs))
            g[nome] = opcoes[v % len(opcoes)]
        return normalizar(g)

    def codificar(self, genoma):
        g, x = normalizar(genoma), 0
        for nome, opcoes, qs in self.genes:
            v = opcoes.index(g[nome]) if g[nome] in opcoes else 0
            for k, q in enumerate(qs):
                x |= ((v >> k) & 1) << q
        return x

    def registros(self):
        return {nome: qs for nome, _, qs in self.genes if qs}


class GeradorQuantico:
    """Circuito de Born: Ry em cada qubit, depois (RZZ nos pares + Ry) por camada.
    Com os ângulos iniciais (1ª camada Ry = pi/2, Ry seguintes = 0) a
    distribuição é uniforme: todos os genomas em superposição com o mesmo peso.

    As RZZ começam com ângulos ALEATÓRIOS: com todas em zero o estado é real,
    a derivada em relação a cada RZZ é imaginária pura e o gradiente delas é
    exatamente zero (ponto de sela) -- o emaranhamento nunca nasceria. Fases
    antes da última camada não mudam as probabilidades iniciais, então a
    população continua uniforme."""

    def __init__(self, n_qubits, pares, camadas=2):
        if n_qubits > 22:
            raise ValueError("gerador com mais de 22 qubits: o ajuste exato fica caro demais")
        self.n, self.pares, self.camadas = n_qubits, list(pares), max(1, camadas)
        self.n_params = n_qubits + (self.camadas - 1) * (len(self.pares) + n_qubits)
        idx = np.arange(1 << n_qubits)
        self._zz = [(_sinal(idx, a) * _sinal(idx, b)).astype(np.int8) for a, b in self.pares]

    def params_iniciais(self, seed=None):
        p = np.zeros(self.n_params)
        p[:self.n] = np.pi / 2
        r = np.random.default_rng(seed)
        for tipo, _, k in self._portas():
            if tipo == "rzz":
                p[k] = r.uniform(-np.pi / 2, np.pi / 2)
        return p

    def estado(self, params):
        n = self.n
        st = np.zeros(1 << n, dtype=complex)
        st[0] = 1.0
        for q in range(n):
            apply_gate(st, gate_Ry(params[q]), q, n)
        i = n
        for _ in range(1, self.camadas):
            for k in range(len(self.pares)):
                # ZZ só vale +-1: exp(-i t ZZ / 2) = cos(t/2) - i sen(t/2) ZZ
                st *= np.cos(params[i] / 2) - 1j * np.sin(params[i] / 2) * self._zz[k]
                i += 1
            for q in range(n):
                apply_gate(st, gate_Ry(params[i + q]), q, n)
            i += n
        return st

    def probabilidades(self, params):
        p = np.abs(self.estado(params)) ** 2
        return p / p.sum()

    def entropia(self, params):
        """Entropia de Shannon da população em bits (n = tudo em aberto, 0 = convergiu)."""
        p = self.probabilidades(params)
        p = p[p > 1e-15]
        return float(-np.sum(p * np.log2(p)))

    def circuito_qiskit(self, params):
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(self.n)
        for q in range(self.n):
            qc.ry(float(params[q]), q)
        i = self.n
        for _ in range(1, self.camadas):
            for a, b in self.pares:
                qc.rzz(float(params[i]), a, b)
                i += 1
            for q in range(self.n):
                qc.ry(float(params[i + q]), q)
            i += self.n
        qc.measure_all()
        return qc

    def amostrar(self, params, n, rng, backend=None):
        """Mede o gerador n vezes. backend=None: simulação exata; senão qualquer
        backend com .amostrar(circuito, shots) (BackendQiskit: Aer ou QPU IBM)."""
        if backend is None:
            return rng.choice(1 << self.n, size=n, p=self.probabilidades(params))
        contagens = backend.amostrar(self.circuito_qiskit(params), shots=n)
        return np.array([int(s.replace(" ", ""), 2) for s, c in contagens.items() for _ in range(c)])

    def _portas(self):
        """[(tipo, alvo, índice do parâmetro)] na ordem de aplicação."""
        out = [("ry", q, q) for q in range(self.n)]
        i = self.n
        for _ in range(1, self.camadas):
            for k in range(len(self.pares)):
                out.append(("rzz", k, i)); i += 1
            out += [("ry", q, i + q) for q in range(self.n)]
            i += self.n
        return out

    def _aplicar(self, st, porta, theta):
        tipo, alvo, _ = porta
        if tipo == "ry":
            apply_gate(st, gate_Ry(theta), alvo, self.n)
        else:
            st *= np.cos(theta / 2) - 1j * np.sin(theta / 2) * self._zz[alvo]

    def _pauli(self, st, porta):
        """P|st> para o gerador da porta (Y para Ry, ZZ para RZZ)."""
        tipo, alvo, _ = porta
        if tipo == "ry":
            v = st.copy()
            apply_gate(v, _Y, alvo, self.n)
            return v
        return st * self._zz[alvo]

    def gradiente(self, params, dL_dp):
        """Gradiente de qualquer L(p) dado dL/dp, por diferenciação ADJUNTA:
        uma passada para frente e uma para trás (~3 simulações), em vez de
        2 por parâmetro. Exato: para G = exp(-i theta P / 2),
        dL/dtheta = Im <beta| P |psi>, com beta = (dL/dp) * psi propagado de volta."""
        psi = self.estado(params)
        beta = np.asarray(dL_dp, float) * psi
        g = np.zeros_like(params, dtype=float)
        for porta in reversed(self._portas()):
            k = porta[2]
            g[k] = np.imag(np.vdot(beta, self._pauli(psi, porta)))
            self._aplicar(psi, porta, -params[k])           # desfaz a porta (G^-1 = G(-theta))
            self._aplicar(beta, porta, -params[k])
        return g

    def objetivo(self, params, xs, pesos, temperatura=0.0):
        """L = -sum w log p(x_elite) - temperatura * H(p) e dL/dp.
        A temperatura (entropia em bits) impede o colapso prematuro da população."""
        p = self.probabilidades(params)
        dL = np.zeros_like(p)
        np.add.at(dL, np.asarray(xs), -np.asarray(pesos, float) / (p[np.asarray(xs)] + 1e-12))
        L = float(-np.sum(np.asarray(pesos) * np.log(p[np.asarray(xs)] + 1e-12)))
        if temperatura > 0:
            pp = np.maximum(p, 1e-15)
            H = float(-np.sum(pp * np.log2(pp)))
            L -= temperatura * H
            dL += temperatura * (np.log2(pp) + 1 / np.log(2))
        return L, dL

    def nll_e_gradiente(self, params, xs, pesos, temperatura=0.0):
        L, dL = self.objetivo(params, xs, pesos, temperatura)
        return L, self.gradiente(params, dL)

    def gradiente_parameter_shift(self, params, xs, pesos, temperatura=0.0):
        """Mesmo gradiente pela regra de parameter-shift (2 simulações por
        parâmetro) -- usado para validar a diferenciação adjunta. Regra da
        cadeia: dL/dtheta = sum_x dL/dp_x * [p_x(theta + pi/2) - p_x(theta - pi/2)] / 2."""
        _, dL = self.objetivo(params, xs, pesos, temperatura)
        g = np.zeros_like(params)
        for k in range(len(params)):
            e = np.zeros_like(params); e[k] = np.pi / 2
            g[k] = (dL @ self.probabilidades(params + e) - dL @ self.probabilidades(params - e)) / 2
        return g


class EvolucaoQuantica:
    """Mesma interface e mesmo registro de campeões da BuscaEvolutiva, com a
    população em superposição num circuito gerador (ver o topo do módulo)."""

    def __init__(self, n_qubits, espaco=None, populacao=12, geracoes=6, elite=0.3, passos=5, lr=0.25,
                 mutacao=0.08, temperatura=0.15, camadas_gerador=2, emaranhamento=True, backend_gerador=None,
                 modo="reservatorio", backend=None, dobras=5, repeticoes=2, penalidade_custo=0.5,
                 seed=0, tempo_max_s=None, registro=None, chave=None, alfa=0.05, min_portas_2q=0,
                 verbose=False, ao_progresso=None):
        self.avaliador = BuscaEvolutiva(n_qubits, espaco=espaco or ESPACO_PADRAO, modo=modo, backend=backend,
                                        dobras=dobras, repeticoes=repeticoes,
                                        penalidade_custo=penalidade_custo, seed=seed, registro=registro,
                                        chave=chave, alfa=alfa, min_portas_2q=min_portas_2q)
        self.cod = CodificacaoGenoma(self.avaliador.espaco)
        n = self.cod.n_bits
        pares = [(i, (i + 1) % n) for i in range(n)] if (emaranhamento and n > 1) else []
        self.gerador = GeradorQuantico(n, pares, camadas_gerador if emaranhamento else 1)
        self.n_qubits, self.populacao, self.geracoes = n_qubits, populacao, geracoes
        self.elite, self.passos, self.lr, self.mutacao = elite, passos, lr, mutacao
        self.temperatura = temperatura
        self.emaranhamento, self.backend_gerador = emaranhamento, backend_gerador
        self.seed, self.tempo_max_s, self.verbose = seed, tempo_max_s, verbose
        self.ao_progresso = ao_progresso        # callback(evento) a cada geração (painel ao vivo)
        self.registro, self.chave = registro, chave
        self.params = self.gerador.params_iniciais(seed)

    def ligacoes_aprendidas(self, top=5):
        """Ligação entre genes que o circuito aprendeu: informação mútua (bits)
        entre os registros de dois genes na POPULAÇÃO que o gerador sorteia.
        Começa em 0 (população uniforme) e só cresce se o circuito aprender
        que dois genes andam juntos. Um gerador sem emaranhamento produz genes
        independentes -- nele a ligação é sempre 0: quem cria as ligações é
        o emaranhamento."""
        p = self.gerador.probabilidades(self.params)
        idx = np.arange(len(p))
        regs = list(self.cod.registros().items())
        valores = {nome: sum(((idx >> q) & 1) << k for k, q in enumerate(qs)) for nome, qs in regs}
        out = []
        for i in range(len(regs)):
            for j in range(i + 1, len(regs)):
                a, b = regs[i][0], regs[j][0]
                na, nb = 1 << len(regs[i][1]), 1 << len(regs[j][1])
                conj = np.bincount(valores[a] * nb + valores[b], weights=p, minlength=na * nb).reshape(na, nb)
                pa, pb = conj.sum(1, keepdims=True), conj.sum(0, keepdims=True)
                m = conj > 1e-15
                mi = float(np.sum(conj[m] * np.log2(conj[m] / (pa @ pb)[m])))
                out.append({"genes": (a, b), "informacao_mutua": max(mi, 0.0)})
        return sorted(out, key=lambda d: -d["informacao_mutua"])[:top]

    def emaranhamento_gerador(self):
        """Entropia de emaranhamento (bits) entre a metade inferior e a superior
        dos qubits do gerador -- quanto do circuito está emaranhado."""
        n = self.gerador.n
        return entropia_emaranhamento(self.gerador.estado(self.params), n, list(range(n // 2)))

    def executar(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        t0 = time.perf_counter()
        av = self.avaliador
        rng = np.random.default_rng(self.seed)
        divisoes = av._divisoes(y)
        campeao_reg = self.registro.campeao(self.chave) if self.registro else None
        for g in [GENOMA_PADRAO] + ([campeao_reg["genoma"]] if campeao_reg else []):
            av.avaliar(normalizar(g), X, y, divisoes)
        H0 = self.gerador.entropia(self.params)
        historico = []
        for ger in range(self.geracoes + 1):
            mutado = self.params + rng.normal(0, self.mutacao, self.params.shape)     # mutação quântica
            xs = self.gerador.amostrar(mutado, self.populacao, rng, self.backend_gerador)
            geracao = [av.avaliar(self.cod.decodificar(x), X, y, divisoes) for x in xs]
            todos = sorted(av._cache.values(), key=lambda a: -a["fitness"])
            lig = self.ligacoes_aprendidas(top=1) if self.emaranhamento else []
            historico.append({"geracao": ger, "melhor": todos[0]["fitness"],
                              "melhor_acuracia": todos[0]["acuracia"],
                              "media_geracao": float(np.mean([a["fitness"] for a in geracao])),
                              "entropia_bits": self.gerador.entropia(self.params),
                              "genomas_distintos": int(len(set(int(x) for x in xs))),
                              "ligacao_max_bits": lig[0]["informacao_mutua"] if lig else 0.0,
                              "emaranhamento_bits": self.emaranhamento_gerador() if self.emaranhamento else 0.0,
                              "melhor_descricao": todos[0]["descricao"]})
            if self.ao_progresso is not None:
                self.ao_progresso(dict(historico[-1], tipo="geracao", total_geracoes=self.geracoes,
                                       padrao_acuracia=av._cache[chave_genoma(GENOMA_PADRAO)]["acuracia"],
                                       melhor_portas_2q=todos[0]["custo"],
                                       ligacoes=[{"genes": list(l["genes"]), "bits": l["informacao_mutua"]}
                                                 for l in (self.ligacoes_aprendidas(top=5)
                                                           if self.emaranhamento else [])]))
            if self.verbose:
                h = historico[-1]
                print(f"  geração {ger}: melhor {100 * h['melhor_acuracia']:.1f}% ({h['melhor_descricao']}) "
                      f"| população: {h['genomas_distintos']}/{self.populacao} distintos, "
                      f"entropia {h['entropia_bits']:.1f} bits, ligação máx {h['ligacao_max_bits']:.3f}",
                      flush=True)
            if ger == self.geracoes or (self.tempo_max_s and time.perf_counter() - t0 > self.tempo_max_s):
                break
            # herança: a elite (melhores já avaliados) puxa as amplitudes do gerador
            validos = [a for a in todos if a["fitness"] > -1]
            k = max(2, int(round(self.elite * self.populacao)))
            elite = validos[:k]
            xs_e = [self.cod.codificar(a["genoma"]) for a in elite]
            pesos = np.arange(len(elite), 0, -1, dtype=float)
            pesos /= pesos.sum()
            for _ in range(self.passos):
                _, grad = self.gerador.nll_e_gradiente(self.params, xs_e, pesos, self.temperatura)
                self.params = self.params - self.lr * grad

        todos = sorted(av._cache.values(), key=lambda a: -a["fitness"])
        melhor = todos[0]
        resultado = {"motor": "quantico", "melhor": melhor,
                     "padrao": av._cache[chave_genoma(GENOMA_PADRAO)],
                     "avaliados": len(av._cache), "historico": historico,
                     "n_qubits_gerador": self.gerador.n, "genomas_possiveis": self.cod.n_genomas,
                     "entropia_inicial_bits": H0, "entropia_final_bits": self.gerador.entropia(self.params),
                     "ligacoes_aprendidas": self.ligacoes_aprendidas() if self.emaranhamento else [],
                     "backend_gerador": getattr(self.backend_gerador, "nome", "exato"),
                     "tempo_s": round(time.perf_counter() - t0, 2), "promovido": None, "motivo": None}
        if self.registro is not None:
            resultado["promovido"], resultado["motivo"] = av._promover(
                melhor, campeao_reg, X, y, divisoes, motor="quantico")
        return resultado

    def resumo(self, resultado):
        r = resultado
        linhas = [f"Evolução quântica: gerador de {r['n_qubits_gerador']} qubits "
                  f"({r['genomas_possiveis']:,} arquiteturas em superposição), backend {r['backend_gerador']}",
                  f"  entropia da população: {r['entropia_inicial_bits']:.1f} -> {r['entropia_final_bits']:.1f} bits",
                  f"  melhor: {100 * r['melhor']['acuracia']:.1f}% ({descrever(r['melhor']['genoma'])}, "
                  f"{r['melhor']['custo']} portas 2q) | padrão: {100 * r['padrao']['acuracia']:.1f}%"]
        for lig in r["ligacoes_aprendidas"][:3]:
            if lig["informacao_mutua"] > 1e-3:
                linhas.append(f"  ligação aprendida: {lig['genes'][0]} <-> {lig['genes'][1]} "
                              f"(informação mútua {lig['informacao_mutua']:.3f} bits)")
        if r["promovido"] is not None:
            linhas.append(f"  -> {'PROMOVIDO' if r['promovido'] else 'mantido'}: {r['motivo']}")
        return "\n".join(linhas)
