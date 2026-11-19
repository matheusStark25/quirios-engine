"""Fase 5: auto-evolução do circuito (busca evolutiva + campeões com promoção por evidência)."""
import json

import numpy as np
import pytest

from quirios_engine import BuscaEvolutiva, QuantumConfig, RegistroCampeoes, evoluir_continuamente
from quirios_engine.evolucao import (ESPACO_PADRAO, GENOMA_PADRAO, chave_genoma, espaco_valido,
                                     normalizar)


def _dados(n=48, seed=0):
    r = np.random.default_rng(seed)
    X = r.uniform(0, 1, (n, 4))
    return X, ((X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)).astype(float)


def _busca(**kw):
    base = dict(n_qubits=4, populacao=5, geracoes=2, dobras=3, repeticoes=1, seed=0)
    base.update(kw)
    return BuscaEvolutiva(**base)


def test_busca_nunca_termina_pior_que_o_padrao():
    X, y = _dados()
    res = _busca().executar(X, y)
    assert res["melhor"]["fitness"] >= res["padrao"]["fitness"]
    assert res["avaliados"] >= 5
    assert len(res["historico"]) == 3
    assert [h["melhor"] for h in res["historico"]] == sorted(h["melhor"] for h in res["historico"])


def test_operadores_geneticos_ficam_no_espaco():
    b = _busca()
    esp = b.espaco
    for _ in range(30):
        g = b._aleatorio()
        m = b._mutar(g)
        assert chave_genoma(m) != chave_genoma(g)
        f = b._cruzar(g, m)
        for k, v in esp.items():
            assert m[k] in v and f[k] in (g[k], m[k])


def test_espaco_valido_remove_todos_pares_acima_de_8_qubits():
    assert "todos_pares" in espaco_valido(ESPACO_PADRAO, 8)["topologia"]
    assert "todos_pares" not in espaco_valido(ESPACO_PADRAO, 9)["topologia"]


def test_registro_so_promove_com_evidencia_sobre_o_padrao(tmp_path):
    # XOR: o circuito padrão é fraco e a evolução o vence com evidência
    X, y = _dados(n=80)
    reg = RegistroCampeoes(tmp_path / "c.json")
    r1 = _busca(registro=reg, chave="xor", dobras=4, repeticoes=3).executar(X, y)
    assert r1["promovido"] is True and "circuito padrão" in r1["motivo"]
    r2 = _busca(registro=reg, chave="xor", dobras=4, repeticoes=3).executar(X, y)   # nada a promover
    assert r2["promovido"] is False
    assert len(reg.historico("xor")) == 2
    reaberto = RegistroCampeoes(tmp_path / "c.json")
    assert chave_genoma(reaberto.campeao("xor")["genoma"]) == chave_genoma(r1["melhor"]["genoma"])


def test_sem_evidencia_o_circuito_padrao_continua(tmp_path):
    # Regressão v0.6: o primeiro campeão era promovido sem evidência e podia ser pior que o padrão
    r = np.random.default_rng(1)
    X = r.uniform(0, 1, (60, 4)); y = (X[:, 0] > 0.5).astype(float)       # linear: o padrão já vai bem
    reg = RegistroCampeoes(tmp_path / "c.json")
    res = _busca(registro=reg, chave="linear", dobras=4, repeticoes=3, geracoes=2).executar(X, y)
    assert res["promovido"] is False and "circuito padrão mantido" in res["motivo"]
    assert reg.campeao("linear") is None and len(reg.historico("linear")) == 1


def test_regra_de_promocao():
    b = _busca()
    camp = {"genoma": normalizar(GENOMA_PADRAO), "accs": np.full(15, 0.80), "custo": 16}
    outro = normalizar(dict(GENOMA_PADRAO, emaranhador="cz"))
    ganho_claro = {"genoma": outro, "accs": np.full(15, 0.80) + np.linspace(0.08, 0.12, 15), "custo": 16}
    assert b._decidir(ganho_claro, camp, 40, 10)[0] is True
    ruido = {"genoma": outro, "accs": 0.80 + np.random.default_rng(0).normal(0, 0.05, 15), "custo": 16}
    assert b._decidir(ruido, camp, 40, 10)[0] is False
    mais_barato = {"genoma": outro, "accs": np.full(15, 0.80) + np.linspace(-0.005, 0.005, 15), "custo": 8}
    ok, motivo = b._decidir(mais_barato, camp, 40, 10)
    assert ok is True and "portas" in motivo


def test_evoluir_continuamente_acumula_historico(tmp_path):
    X, y = _dados()
    reg = RegistroCampeoes(tmp_path / "c.json")
    res = evoluir_continuamente(X, y, 4, reg, "xor", rodadas=2, populacao=4, geracoes=1, dobras=3,
                                repeticoes=1)
    assert len(res) == 2 and len(reg.historico("xor")) == 2


def test_orquestrador_usa_o_campeao(tmp_path):
    from quirios_engine.orquestrador import candidatos_padrao
    reg = RegistroCampeoes(tmp_path / "c.json")
    assert "quantico[campeao]" not in [c.nome for c in candidatos_padrao(QuantumConfig(4, "anel", 1),
                                                                         registro=reg, chave="xor")]
    reg.registrar("xor", {"genoma": normalizar(dict(GENOMA_PADRAO, emaranhador="cz"))}, promover=True)
    nomes = [c.nome for c in candidatos_padrao(QuantumConfig(4, "anel", 1), registro=reg, chave="xor")]
    assert "quantico[campeao]" in nomes


def test_restricao_de_quanticidade():
    X, y = _dados()
    livre = _busca().executar(X, y)
    assert "simulavel_classicamente" in livre["melhor"]
    quantica = _busca(min_portas_2q=4).executar(X, y)
    assert quantica["melhor"]["custo"] >= 4 and not quantica["melhor"]["simulavel_classicamente"]


def test_modo_kernel():
    X, y = _dados(n=30)
    res = _busca(modo="kernel", populacao=3, geracoes=1).executar(X, y)
    assert 0 <= res["melhor"]["acuracia"] <= 1 and res["melhor"]["erro"] is None


def test_registro_assinado_com_ml_dsa(tmp_path):
    pytest.importorskip("cryptography.hazmat.primitives.asymmetric.mldsa")
    from quirios_engine.seguranca import Assinador, Verificador, gerar_chaves
    priv, pub = gerar_chaves(tmp_path / "chaves")
    X, y = _dados()
    reg = RegistroCampeoes(tmp_path / "c.json", assinador=Assinador(priv))
    _busca(registro=reg, chave="xor", populacao=3, geracoes=1).executar(X, y)
    assert (tmp_path / "c.json.sig").exists()
    RegistroCampeoes(tmp_path / "c.json", verificador=Verificador(pub))          # íntegro: abre
    dados = json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))
    dados["problemas"]["xor"]["historico"][0]["acuracia"] = 1.0                   # adulterado
    (tmp_path / "c.json").write_text(json.dumps(dados), encoding="utf-8")
    with pytest.raises(ValueError, match="pós-quântica"):
        RegistroCampeoes(tmp_path / "c.json", verificador=Verificador(pub))


def test_cli_evolucao(tmp_path, capsys):
    pytest.importorskip("sklearn")
    from quirios_engine.evolucao import main
    main(["--dataset", "iris", "--n-qubits", "4", "--n-treino", "40", "--populacao", "3",
          "--geracoes", "1", "--registro", str(tmp_path / "c.json")])
    saida = capsys.readouterr().out
    assert "padrão QURIOS" in saida and "rodada 1 (motor quantico)" in saida
