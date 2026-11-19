import pytest

torch = pytest.importorskip("torch")

from quirios_engine import llm_nativo as ln  # noqa: E402
from quirios_engine.ia_tensorial import CamadaPostoBaixo, CamadaTT, contar_params  # noqa: E402


def test_as_quatro_versoes_e_a_largura_equivalente():
    V = 50
    tt = ln.MiniGPT(V, 64, 2, 4, 32, ln.fabrica("tt", 4, 3))
    pb = ln.MiniGPT(V, 64, 2, 4, 32, ln.fabrica("posto_baixo", 4, 3))
    assert isinstance(tt.blocos[0].fc1, CamadaTT) and isinstance(pb.blocos[1].qkv, CamadaPostoBaixo)
    denso = ln.MiniGPT(V, 64, 2, 4, 32)
    assert contar_params(tt) < contar_params(denso)
    L = ln._largura_equivalente(V, contar_params(tt), 2, 4, 32)
    assert contar_params(ln.MiniGPT(V, L, 2, 4, 32)) <= contar_params(tt) < contar_params(ln.MiniGPT(V, L + 16, 2, 4, 32))


def test_treina_aprende_e_escreve():
    torch.manual_seed(0)
    ids = torch.tensor([1, 2, 3, 4] * 400)                                     # padrão fácil de aprender
    m = ln.MiniGPT(5, 32, 1, 2, 16, ln.fabrica("tt", 4, 2))
    antes = ln.bits_por_caractere(m, ids, max_chars=200)
    ln.treinar(m, ids, passos=120, lote=8, seq=16, lr=5e-3)
    depois = ln.bits_por_caractere(m, ids, max_chars=200)
    assert depois < antes and depois < 1.0
    texto = ln.escrever(m, ["?", "a", "b", "c", "d"], inicio="ab", n=8, temperatura=0.3)
    assert len(texto) == 10 and "abcd" in texto
