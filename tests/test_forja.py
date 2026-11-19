import numpy as np
import pytest

from quirios_engine import forja


def _alvo(genoma):
    """Problema de brinquedo: fitness = genes iguais a um projeto secreto (com interação entre a e b)."""
    certo = {"a": 3, "b": "y", "c": 1, "d": 2}
    acertos = sum(genoma[k] == v for k, v in certo.items())
    bonus = 1.0 if (genoma["a"] == 3 and genoma["b"] == "y") else 0.0
    return {"fitness": acertos + bonus, "acuracia": acertos / 4, "compressao": 1.0}


ESPACO = {"a": (0, 1, 2, 3), "b": ("x", "y"), "c": (0, 1, 2, 3), "d": (0, 1, 2, 3)}


@pytest.mark.parametrize("motor", ["quantico", "classico", "aleatorio"])
def test_os_tres_motores_buscam_com_o_mesmo_orcamento(motor):
    r = forja.Busca(ESPACO, _alvo, motor, populacao=8, geracoes=6, seed=1).executar()
    assert len(r["historico"]) == 7 and r["avaliados"] <= 8 * 7
    melhores = [h["melhor_fitness"] for h in r["historico"]]
    assert melhores == sorted(melhores)                      # o melhor já visto nunca piora
    assert r["melhor"]["fitness"] >= 3
    if motor == "quantico":
        assert r["n_qubits_gerador"] == 7 and r["historico"][-1]["entropia_bits"] < 7
        assert r["ligacoes_aprendidas"] and r["projetos_possiveis"] == 4 * 2 * 4 * 4


def test_codificacao_ida_e_volta_e_canonico():
    cod = forja._Codificacao(ESPACO)
    g = {"a": 2, "b": "y", "c": 0, "d": 3}
    assert cod.decodificar(cod.codificar(g)) == g
    c = forja.canonico_compressao({"0:metodo": "densa", "0:nivel": 3, "0:nucleos": 6,
                                   "2:metodo": "posto_baixo", "2:nivel": 2, "2:nucleos": 4})
    assert c == {"0:metodo": "densa", "0:nivel": 0, "0:nucleos": 3,
                 "2:metodo": "posto_baixo", "2:nivel": 2, "2:nucleos": 3}
    with pytest.raises(ValueError):
        forja.Busca(ESPACO, _alvo, "magico")


def test_projeto_aplicado_num_modelo_e_avaliado():
    torch = pytest.importorskip("torch")
    from torch import nn
    from quirios_engine.ia_tensorial import CamadaPostoBaixo, CamadaTT, contar_params
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2))
    espaco = forja.espaco_compressao(m)
    assert set(espaco) == {"0:metodo", "0:nivel", "0:nucleos", "2:metodo", "2:nivel", "2:nucleos"}
    proj = {"0:metodo": "tt", "0:nivel": 1, "0:nucleos": 3, "2:metodo": "posto_baixo", "2:nivel": 0, "2:nucleos": 3}
    novo = forja.aplicar_projeto(m, proj)
    assert isinstance(novo[0], CamadaTT) and isinstance(novo[2], CamadaPostoBaixo) and isinstance(m[0], nn.Linear)
    assert "TT 3 núcleos, ligação 4" in forja.descrever(proj)
    X = torch.randn(400, 64)
    y = (X[:, 0] > 0).long()
    prob = forja.ProblemaCompressao(m, X[:300], y[:300], X[300:], y[300:], alvo=2.0, passos_ajuste=30, lote=32)
    r = prob(proj)
    assert r["parametros"] == contar_params(novo) and r["compressao"] > 2 and 0 <= r["acuracia"] <= 1
    assert prob({**proj, "0:metodo": "densa", "2:metodo": "densa"})["fitness"] < 0     # não bate a meta
