import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from quirios_engine import ia_tensorial as it  # noqa: E402


def test_camada_tt_e_a_matriz_que_representa():
    torch.manual_seed(0)
    c = it.CamadaTT(784, 1024, 8)
    x = torch.randn(5, 784)
    W = torch.as_tensor(c.densa(), dtype=torch.float32)
    assert (c(x) - (x @ W.T + c.vies)).abs().max() < 1e-4
    assert it.contar_params(c) == it.params_camada_tt(784, 1024, 8) + 1024
    assert 0.5 < float(W.std()) / (2 / 784) ** 0.5 < 1.5                 # inicialização tipo He
    assert tuple(c(torch.randn(2, 3, 784)).shape) == (2, 3, 1024)          # lotes com mais eixos


def test_posto_cheio_reproduz_a_camada_original():
    torch.manual_seed(1)
    lin = nn.Linear(64, 96)
    x = torch.randn(4, 64)
    assert (it.CamadaTT.de_linear(lin, 10 ** 6, d=3)(x) - lin(x)).abs().max() < 1e-5
    assert (it.CamadaPostoBaixo.de_linear(lin, 64)(x) - lin(x)).abs().max() < 1e-5


def test_nucleos_equilibrados_mesma_matriz_mesma_escala():
    torch.manual_seed(2)
    lin = nn.Linear(512, 768)
    c = it.CamadaTT.de_linear(lin, 32, d=3)
    rms = [float(G.detach().pow(2).mean().sqrt()) for G in c.nucleos]
    assert max(rms) / min(rms) < 1.0001
    from quirios_engine.redes_tensoriais import fatorar, tt_svd_matriz
    cru = tt_svd_matriz(lin.weight.detach().double().numpy(), fatorar(768, 3), fatorar(512, 3), 32)
    rms_cru = [float((G ** 2).mean() ** 0.5) for G in cru]
    assert max(rms_cru) / min(rms_cru) > 3                                  # o TT-SVD puro é desequilibrado
    sem = it.CamadaTT(512, 768, 32, 3, nucleos=cru)
    sem.vies.data.copy_(c.vies.data)
    x = torch.randn(4, 512)
    assert (c(x) - sem(x)).abs().max() < 1e-5                               # mesma função


def test_comprimir_modelo_troca_so_as_camadas_grandes():
    m = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
    tt = it.comprimir_modelo(m, "tt", 4)
    svd = it.comprimir_modelo(m, "svd", 4)
    assert [type(l).__name__ for l in tt] == ["CamadaTT", "ReLU", "Linear"]
    assert [type(l).__name__ for l in svd] == ["CamadaPostoBaixo", "ReLU", "Linear"]
    assert isinstance(m[0], nn.Linear)                                     # o original não muda
    p_tt, p_svd = it.contar_params(tt[0]), it.contar_params(svd[0])
    assert abs(p_tt - p_svd) / p_tt < 0.5                                  # comparação com os mesmos parâmetros
    with pytest.raises(ValueError):
        it.comprimir_modelo(m, "magico", 4)


def test_treina_e_aprende():
    torch.manual_seed(2)
    X = torch.randn(512, 64)
    y = (X[:, 0] + X[:, 1] > 0).long()
    m = nn.Sequential(it.CamadaTT(64, 64, 4, d=3), nn.ReLU(), nn.Linear(64, 2))
    antes = it.acuracia(m, X, y)
    it.treinar(m, X, y, epocas=15, lote=64, lr=3e-3)
    assert it.acuracia(m, X, y) > max(0.9, antes)


def test_experimento_de_velocidade_pequeno():
    r = it.experimento_velocidade(larguras=(256,), posto=4, lotes=(1, 8))
    tipos = [l["camada"] for l in r["linhas"]]
    assert tipos[0] == "densa" and len(tipos) == 3
    assert all(l["compressao"] >= 1 and l["ms_lote_8"] > 0 for l in r["linhas"])
    assert "L=  256" in it.texto_velocidade(r)
