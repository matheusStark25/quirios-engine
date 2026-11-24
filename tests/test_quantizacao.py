import pytest

torch = pytest.importorskip("torch")

from quirios_engine import quantizacao as qz  # noqa: E402


def _entradas_correlacionadas(n, dim, seed=1):
    """Como as ativações reais: poucos canais grandes, muitos pequenos, correlacionados."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, dim, generator=g) @ torch.randn(dim, dim, generator=g) * torch.linspace(0.05, 3, dim)


def test_grupo_usa_o_maior_divisor_que_cabe():
    assert qz._grupo(896, 128) == 128 and qz._grupo(4864, 128) == 128
    assert qz._grupo(576, 128) == 96                                         # SmolLM2: 576 = 6 x 96
    assert qz._grupo(100, 0) == 100 and qz._grupo(64, 128) == 64


def test_arredondamento_por_grupo_e_fiel_e_o_zero_cabe_na_grade():
    W = torch.randn(32, 256)
    for bits in (8, 4, 3):
        q, e, z = qz.quantizar_grupos(W, bits, 64)
        assert q.dtype == torch.uint8 and int(q.max()) <= 2 ** bits - 1
        assert e.dtype == torch.float16 and e.shape == (32, 4, 1)
        assert int(z.max()) <= 2 ** bits - 1
        erro = (qz.desquantizar(q, e, z) - W).abs().reshape(32, 4, 64)
        assert bool((erro <= e.float() * 1.001).all())                      # nunca mais que um degrau da grade
        assert float(erro.mean()) < float(e.float().mean()) * 0.35          # em média, ~1/4 de degrau
    Wpos = torch.rand(8, 64) + 1                                             # grupo todo positivo
    q, e, z = qz.quantizar_grupos(Wpos, 4, 64)
    assert float((qz.desquantizar(q, e, z) - Wpos).abs().max()) < 0.2
    with pytest.raises(ValueError):
        qz.quantizar_grupos(W, 1, 64)


def test_guiado_pelos_dados_e_a_correcao_erram_menos_na_saida():
    torch.manual_seed(0)
    lin = torch.nn.Linear(128, 64)
    X = _entradas_correlacionadas(4000, 128)
    C = X.T @ X / len(X)
    alvo = lin(X)
    erro = lambda c: float((c(X) - alvo).pow(2).mean().detach())
    rtn = qz.CamadaQuantizada.de_linear(lin, 3, 32)
    dados = qz.CamadaQuantizada.de_linear(lin, 3, 32, cov=C, dados=True)
    corr = qz.CamadaQuantizada.de_linear(lin, 3, 32, cov=C, dados=True, posto_correcao=8)
    assert erro(dados) <= erro(rtn) * 1.0001                                # alfa = 0 está na grade
    assert erro(corr) < erro(dados)
    assert dados.alfa > 0                                                    # canais desiguais: a escala ajuda
    with pytest.raises(ValueError):
        qz.CamadaQuantizada.de_linear(lin, 4, 32, dados=True)               # sem cov


def test_tamanho_em_bits_conta_tudo_o_que_e_guardado():
    lin = torch.nn.Linear(256, 32)
    c = qz.CamadaQuantizada.de_linear(lin, 4, 128)
    grupos = 32 * 2
    assert c.tamanho_bits() == 256 * 32 * 4 + grupos * (16 + 4) + 32 * 16   # pesos + escala/zero + viés
    X = torch.randn(500, 256)
    cc = qz.CamadaQuantizada.de_linear(lin, 4, 128, cov=X.T @ X / 500, dados=True, posto_correcao=3)
    extra = (16 * 256 if cc.canal is not None else 0) + 16 * 3 * (256 + 32)
    assert cc.tamanho_bits() == c.tamanho_bits() + extra
    modelo = torch.nn.Sequential(torch.nn.Linear(8, 256), c)
    assert qz.bits_do_modelo(modelo) == 16 * (8 * 256 + 256) + c.tamanho_bits()


def test_oito_bits_quase_nao_muda_a_camada_e_a_matriz_fica_guardada():
    torch.manual_seed(0)
    lin = torch.nn.Linear(96, 48)
    c = qz.CamadaQuantizada.de_linear(lin, 8, 32)
    x = torch.randn(5, 96)
    with torch.inference_mode():
        assert float((c(x) - lin(x)).abs().max()) < 0.01
        W = c.peso()
        assert c.peso() is W                                                 # inferência: não remonta


def test_ler_metodo():
    assert qz.ler_metodo("int4") == (4, False, False)
    assert qz.ler_metodo("int3+dados+correcao") == (3, True, True)
    assert qz.ler_metodo("int8+dados") == (8, True, False)
    assert qz.ler_metodo("tt") is None and qz.ler_metodo("svd_dados") is None
    for ruim in ("int9", "int4+magia", "intx", "int4+dados+dados"):
        with pytest.raises(ValueError):
            qz.ler_metodo(ruim)
    assert qz.posto_para_bits(896, 4864, 0.5) == 24
