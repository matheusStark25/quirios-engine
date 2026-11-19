import numpy as np
import pytest

pytest.importorskip("cryptography.hazmat.primitives.asymmetric.mldsa")

from quirios_engine import QuantumConfig, VQCClassifier  # noqa: E402
from quirios_engine.seguranca import Assinador, Verificador, gerar_chaves  # noqa: E402


@pytest.fixture
def chaves(tmp_path):
    return gerar_chaves(tmp_path / "chaves")


def test_assinar_e_verificar_arquivo(tmp_path, chaves):
    priv, pub = chaves
    arq = tmp_path / "dados.json"
    arq.write_text('{"a": 1}', encoding="utf-8")
    sig = Assinador(priv).assinar_arquivo(arq)
    assert sig.read_text(encoding="ascii").startswith("ML-DSA-65")
    v = Verificador(pub)
    assert v.verificar_arquivo(arq) is True
    arq.write_text('{"a": 2}', encoding="utf-8")                 # adulterado
    assert v.verificar_arquivo(arq) is False
    assert v.verificar_arquivo(tmp_path / "sem_assinatura.json") is False


def test_chave_de_outro_par_nao_verifica(tmp_path, chaves):
    priv, _ = chaves
    _, pub_outro = gerar_chaves(tmp_path / "outras", nome="outro")
    arq = tmp_path / "x.bin"
    arq.write_bytes(b"quirios")
    Assinador(priv).assinar_arquivo(arq)
    assert Verificador(pub_outro).verificar_arquivo(arq) is False


def test_modelo_salvo_assinado(tmp_path, chaves):
    priv, pub = chaves
    r = np.random.default_rng(0)
    X = r.uniform(0, 1, (12, 3)); y = (X[:, 0] > 0.5).astype(float)
    clf = VQCClassifier(QuantumConfig(3, "anel", 1), epocas=1, seed=0).fit(X, y)
    clf.salvar(tmp_path / "m.json", assinador=Assinador(priv))
    ok = VQCClassifier.carregar(tmp_path / "m.json", verificador=Verificador(pub))
    assert np.allclose(ok.predict_proba(X), clf.predict_proba(X))
    txt = (tmp_path / "m.json").read_text(encoding="utf-8").replace('"b": ', '"b": 1+')
    (tmp_path / "m.json").write_text(txt, encoding="utf-8")
    with pytest.raises(ValueError, match="assinatura"):
        VQCClassifier.carregar(tmp_path / "m.json", verificador=Verificador(pub))
