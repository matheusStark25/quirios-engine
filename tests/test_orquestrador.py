"""Fase 4: estatística, roteamento de backend, orquestrador e benchmark contínuo."""
import json

import numpy as np
import pytest

from quirios_engine import QuantumConfig, Quantico
from quirios_engine.estatistica import comparar_pareado, t_critico, t_sf
from quirios_engine.orquestrador import DeGerador, DeModelo, Orquestrador, escolher_backend
from quirios_engine.sistema import RFF, Brutos


# ── estatística ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("t, gl", [(0.0, 5), (1.0, 3), (2.5, 10), (-1.7, 7), (4.0, 30)])
def test_t_sf_igual_ao_scipy(t, gl):
    st = pytest.importorskip("scipy.stats")
    assert abs(t_sf(t, gl) - st.t.sf(t, gl)) < 1e-10


def test_t_critico():
    assert abs(t_critico(10) - 2.228139) < 1e-5
    assert abs(t_critico(1000) - 1.962339) < 1e-4


def test_comparar_pareado():
    assert comparar_pareado([0.1] * 8)["p_valor"] == 0.0
    simetrico = comparar_pareado([0.1, -0.1, 0.2, -0.2])
    assert abs(simetrico["p_valor"] - 0.5) < 1e-9
    r = comparar_pareado([0.05, 0.02, 0.04, 0.03, 0.06])
    assert r["ic"][0] > 0 and r["p_valor"] < 0.01
    # a correção de Nadeau-Bengio é mais conservadora
    assert comparar_pareado([0.05, 0.0, 0.04, 0.01, 0.06], correcao=0.25)["p_valor"] > \
        comparar_pareado([0.05, 0.0, 0.04, 0.01, 0.06])["p_valor"]


# ── roteamento ──────────────────────────────────────────────────────────────

def test_escolher_backend():
    assert escolher_backend(QuantumConfig(8, "anel", 2))[0].nome == "exato"
    assert escolher_backend(QuantumConfig(20, "anel", 2))[0].nome == "exato"
    pytest.importorskip("qiskit_aer")
    assert escolher_backend(QuantumConfig(40, "linear", 1))[0].nome == "mps"
    b, motivo = escolher_backend(QuantumConfig(40, "anel", 2))
    assert b.nome == "mps(chi=64)" and "aproximado" in motivo
    with pytest.raises(ValueError):
        escolher_backend(QuantumConfig(40, "anel", 2), permitir_aproximacao=False)


# ── orquestrador ────────────────────────────────────────────────────────────

class _Produto:
    """Gerador 'quântico' de mentira com a feature certa para XOR (x0*x1)."""
    nome, quantico = "produto", True
    def preparar(self, X, y=None): return self
    def transformar(self, X): return (X[:, :1] - 0.5) * (X[:, 1:2] - 0.5)


class _Ruido:
    nome, quantico = "ruido", True
    def preparar(self, X, y=None):
        self.r = np.random.default_rng(0); return self
    def transformar(self, X): return np.random.default_rng(len(X)).normal(size=(len(X), 3))


def _xor(n=80, seed=0):
    r = np.random.default_rng(seed)
    X = r.uniform(0, 1, (n, 2))
    return X, ((X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)).astype(float)


def test_quantico_com_evidencia_e_escolhido():
    X, y = _xor()
    orq = Orquestrador([DeGerador(Brutos()), DeGerador(_Produto())]).fit(X, y)
    assert orq.relatorio["escolhido"] == "produto"
    assert orq.relatorio["comparacao"]["evidencia"] is True
    assert (orq.predict(X) == y).mean() > 0.9


def test_quantico_sem_evidencia_perde_para_o_classico():
    r = np.random.default_rng(1)
    X = r.uniform(0, 1, (60, 3)); y = (X[:, 0] > 0.5).astype(float)
    orq = Orquestrador([DeGerador(Brutos()), DeGerador(_Ruido())]).fit(X, y)
    assert orq.relatorio["escolhido"] == "brutos"
    assert orq.relatorio["comparacao"]["evidencia"] is False
    assert "mantido o clássico" in orq.relatorio["motivo"]


def test_politica_melhor_media():
    X, y = _xor()
    orq = Orquestrador([DeGerador(Brutos()), DeGerador(_Produto())], politica="melhor_media").fit(X, y)
    assert orq.relatorio["escolhido"] == "produto"


def test_orcamento_pula_candidato_caro():
    X, y = _xor(n=40)
    orq = Orquestrador([DeGerador(Brutos()), DeGerador(_Produto())], orcamento_circuitos=10).fit(X, y)
    linha = [c for c in orq.relatorio["candidatos"] if c["nome"] == "produto"][0]
    assert linha["status"] == "pulado" and "orçamento" in linha["motivo"]
    assert orq.relatorio["escolhido"] == "brutos"


def test_orquestrador_com_quantico_real_e_relatorio_serializavel():
    r = np.random.default_rng(2)
    X = r.uniform(0, 1, (40, 4)); y = (X[:, 0] + X[:, 1] > 1).astype(float)
    cfg = QuantumConfig(4, "anel", 1)
    from quirios_engine import VQCClassifier
    orq = Orquestrador([DeGerador(Brutos()), DeGerador(RFF()), DeGerador(Quantico(cfg)),
                        DeModelo("vqc-spsa", lambda: VQCClassifier(cfg, otimizador="spsa", epocas=3, seed=0),
                                 quantico=True)], repeticoes=1).fit(X, y)
    rel = json.loads(json.dumps(orq.relatorio))
    assert {c["nome"] for c in rel["candidatos"]} == {"brutos", "rff(escala=1)", "quantico[exato]", "vqc-spsa"}
    vqc = [c for c in rel["candidatos"] if c["nome"] == "vqc-spsa"][0]
    assert vqc["circuitos"] > 0
    assert "escolhido" in orq.texto()


def test_orquestrador_assincrono():
    X, y = _xor(n=40)
    t = Orquestrador([DeGerador(Brutos()), DeGerador(_Produto())]).submeter(X, y)
    orq = t.resultado(timeout=120)
    assert t.status() == "concluida" and orq.relatorio["escolhido"] == "produto"


# ── benchmark contínuo ──────────────────────────────────────────────────────

def test_benchmark_rodar_resumir_e_cli(tmp_path, capsys):
    pytest.importorskip("sklearn")
    from quirios_engine import benchmark as bm
    hist = tmp_path / "hist.jsonl"
    regs = bm.rodar("iris", n_treino=[12], sorteios=2, n_qubits=4,
                    candidatos=["logistica", "quantico"], historico=str(hist), verbose=False)
    assert len(regs) == 4 and all(0 <= r["acuracia"] <= 1 for r in regs)
    assert len(bm.ler_historico(hist)) == 4
    res = bm.resumir(regs)
    assert res[0]["comparacao"]["melhor_classico"] == "logistica"
    assert res[0]["comparacao"]["sorteios"] == 2
    bm.main(["resumo", "--historico", str(hist)])
    assert "iris" in capsys.readouterr().out


def test_benchmark_orquestrador_registra_escolha(tmp_path):
    pytest.importorskip("sklearn")
    from quirios_engine import benchmark as bm
    regs = bm.rodar("wine", n_treino=[20], sorteios=1, n_qubits=4, candidatos=["orquestrador", "svm"],
                    historico=None, verbose=False)
    orq = [r for r in regs if r["candidato"] == "orquestrador"][0]
    assert orq["tipo"] == "hibrido" and orq["escolha"]
