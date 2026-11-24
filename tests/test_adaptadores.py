import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from quirios_engine import adaptadores as ad  # noqa: E402
from quirios_engine import llm_tensorial as lt  # noqa: E402


def _llm_minusculo(seed=0):
    torch.manual_seed(seed)
    cfg = transformers.LlamaConfig(vocab_size=96, hidden_size=48, intermediate_size=96, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                                   tie_word_embeddings=True)
    m = transformers.LlamaForCausalLM(cfg)
    m.eval()
    return m


@pytest.mark.parametrize("tipo", ["lora", "tt"])
def test_adaptador_comeca_igual_ao_original_e_so_ele_treina(tipo):
    m, ref = _llm_minusculo(), _llm_minusculo()
    assert ad.instalar(m, tipo, posto=4, d=2) == 2 * 7
    x = torch.randint(0, 96, (1, 12))
    with torch.inference_mode():
        assert float((m(x).logits - ref(x).logits).abs().max()) < 1e-5     # dW = 0 no começo
    tre = ad.treinaveis(m)
    assert all(p.requires_grad for p in tre)
    assert sum(p.numel() for p in tre) == ad.params_treinaveis(ref, tipo, 4, 2)
    assert sum(p.numel() for p in m.parameters() if p.requires_grad) == sum(p.numel() for p in tre)


def test_tt_tem_menos_parametros_e_o_lora_equivalente_casa_com_ele():
    m = _llm_minusculo()
    tt = ad.params_treinaveis(m, "tt", 4, 2)
    assert tt < ad.params_treinaveis(m, "lora", 8)
    r = ad.posto_lora_equivalente(m, 4, 2)
    assert abs(ad.params_treinaveis(m, "lora", r) - tt) <= ad.params_treinaveis(m, "lora", 1)


@pytest.mark.parametrize("tipo", ["lora", "tt"])
def test_ajuste_aprende_uma_sequencia(tipo):
    m = _llm_minusculo()
    ad.instalar(m, tipo, posto=4, d=2)
    ids = torch.arange(90).repeat(20)                                         # padrão fácil de aprender
    antes = lt.perplexidade(m, ids, janela=32, max_tokens=400)
    t, perdas = ad.ajustar(m, ids, passos=60, lote=4, seq=32, lr=1e-2)
    assert lt.perplexidade(m, ids, janela=32, max_tokens=400) < antes * 0.7
    assert len(perdas) == 60 and t > 0


def test_demo_escolhe_a_taxa_na_validacao(monkeypatch):
    g = torch.Generator().manual_seed(0)
    monkeypatch.setattr(lt, "carregar", lambda nome, dispositivo="cpu": (None, _llm_minusculo()))
    monkeypatch.setattr(lt, "texto_conversas", lambda tok, divisao="train_sft", **k: divisao)
    monkeypatch.setattr(lt, "tokens", lambda tok, texto: torch.arange(90).repeat(30 if texto == "train_sft" else 5))
    res = ad.demo("minusculo", configs=(("tt", 4), ("lora", "mesmos")), passos=20, lote=2, seq=16,
                  lrs=(1e-3, 1e-2), d=2, max_tokens=200)
    assert [l["tipo"] for l in res["linhas"]] == [None, "tt", "lora"]
    for l in res["linhas"][1:]:
        assert len(l["tentativas"]) == 2
        assert l["ppl_validacao"] == min(t["ppl_validacao"] for t in l["tentativas"])
    assert res["linhas"][2]["mesmos_parametros_do_tt"]
    assert "adaptador TT" in ad.texto_demo(res)
    del g


def test_tipo_invalido():
    with pytest.raises(ValueError):
        ad.Adaptador(torch.nn.Linear(4, 4), "magico")
