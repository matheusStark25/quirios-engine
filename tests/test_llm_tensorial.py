import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from quirios_engine import llm_tensorial as lt  # noqa: E402
from quirios_engine.ia_tensorial import CamadaPostoBaixo, CamadaTT, contar_params  # noqa: E402


def _llm_minusculo(seed=0):
    """Llama de 2 blocos, pesos aleatórios: a mesma arquitetura do SmolLM2, sem download."""
    torch.manual_seed(seed)
    cfg = transformers.LlamaConfig(vocab_size=96, hidden_size=48, intermediate_size=96, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                                   tie_word_embeddings=True)
    m = transformers.LlamaForCausalLM(cfg)
    m.eval()
    return m


def test_comprimir_troca_todas_as_camadas_lineares_dos_blocos():
    m = _llm_minusculo()
    antes = contar_params(m)
    x = torch.randint(0, 96, (1, 12))
    assert lt.comprimir_llm(m, "tt", posto=10 ** 6, d=2) == 2 * 7          # posto cheio: mesma função
    assert isinstance(m.model.layers[0].mlp.up_proj, CamadaTT)
    ref = _llm_minusculo()
    assert (m(x).logits - ref(x).logits).abs().max() < 1e-3
    m2 = _llm_minusculo()
    lt.comprimir_llm(m2, "svd", posto=4, d=2)
    assert isinstance(m2.model.layers[1].self_attn.o_proj, CamadaPostoBaixo) and contar_params(m2) < antes
    with pytest.raises(ValueError):
        lt.comprimir_llm(_llm_minusculo(), "magico")


def test_branqueamento_pelas_ativacoes_erra_menos_na_saida():
    m = _llm_minusculo()
    ids = torch.randint(0, 96, (2000,))
    cov = lt.covariancias_de_entrada(m, ids, alvos=lt.ALVOS_MLP, n_seq=8, seq=32)
    assert set(cov) == {f"{i}.{n}" for i in range(2) for n in lt.ALVOS_MLP}
    assert torch.equal(cov["0.gate_proj"], cov["0.up_proj"])                 # mesma entrada
    lin = m.model.layers[0].mlp.down_proj
    # entradas correlacionadas, como as ativações reais: o erro na saída é o que importa
    g = torch.Generator().manual_seed(1)
    X = torch.randn(4000, 96, generator=g) @ torch.randn(96, 96, generator=g) * torch.linspace(0.05, 3, 96)
    cov_x = X.T @ X / len(X)
    alvo = X @ lin.weight.T
    erro = lambda c: float(((X @ c.V.T @ c.U.T) - alvo).pow(2).mean().detach())
    from quirios_engine.ia_tensorial import CamadaPostoBaixo
    assert erro(lt.postobaixo_com_dados(lin, 8, cov_x)) < erro(CamadaPostoBaixo.de_linear(lin, 8))
    # na GPU (Kaggle) a covariância chega em float32 de outro dispositivo; a conta é sempre na CPU em float64
    c = lt.postobaixo_com_dados(lin, 8, cov_x.float())
    assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in c.parameters())
    assert all(v.device.type == "cpu" for v in cov.values())
    m2 = _llm_minusculo()
    assert lt.comprimir_llm(m2, "svd_dados", alvos=lt.ALVOS_MLP, cov=cov, fator=2.0) == 6
    with pytest.raises(ValueError):
        lt.comprimir_llm(m2, "magico")


def test_destilacao_aproxima_o_comprimido_do_original():
    professor = _llm_minusculo()
    ids = torch.randint(0, 96, (4000,))
    janelas, top_p, top_i = lt.alvos_do_professor(professor, ids, passos=40, lote=2, seq=16, k=8)
    assert janelas[0].shape == (2, 16) and top_p[0].shape == (2, 16, 8)
    assert torch.allclose(top_p[0].float().sum(-1), torch.ones(2, 16), atol=1e-2)
    aluno = _llm_minusculo()
    lt.comprimir_llm(aluno, "tt", posto=2, d=2)

    def distancia():
        with torch.inference_mode():
            x = janelas[0]
            return float((torch.softmax(aluno(x).logits, -1) - torch.softmax(professor(x).logits, -1)).abs().sum(-1).mean())

    antes = distancia()
    lt.curar(aluno, janelas, top_p, top_i, lr=3e-3)
    assert distancia() < antes
    assert all(p.requires_grad for p in aluno.parameters())                 # volta ao estado normal


def test_guarda_contra_divergencia_volta_ao_ponto_bom():
    professor = _llm_minusculo()
    ids = torch.randint(0, 96, (4000,))
    janelas, top_p, top_i = lt.alvos_do_professor(professor, ids, passos=30, lote=2, seq=16, k=8)
    aluno = _llm_minusculo()
    lt.comprimir_llm(aluno, "tt", posto=4, d=2)
    info = {}
    lt.curar(aluno, janelas, top_p, top_i, lr=50.0, info=info)               # taxa absurda: diverge
    assert info["reducoes_lr"] >= 1
    assert all(torch.isfinite(p).all() for p in aluno.parameters())
    assert lt.perplexidade(aluno, ids, janela=16, max_tokens=200) < 1e4      # não terminou destruído
    info = {}
    lt.curar(aluno, janelas, top_p, top_i, lr=1e-3, info=info)
    assert info["reducoes_lr"] == 0                                         # taxa normal: nenhum alarme falso


def test_perplexidade_de_um_modelo_que_ja_sabe_a_sequencia():
    m = _llm_minusculo()
    ids = torch.randint(0, 96, (300,))
    ppl = lt.perplexidade(m, ids, janela=32, max_tokens=300)
    assert 30 < ppl < 400                                                    # pesos aleatórios ~ vocabulário (96)
