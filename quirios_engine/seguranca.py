"""
Segurança pós-quântica: assinatura digital ML-DSA-65 (NIST FIPS 204, 2024)
para os artefatos do QURIOS -- modelos salvos, registro de campeões.

Computadores quânticos grandes quebram RSA e curvas elípticas (algoritmo de
Shor); ML-DSA é baseado em reticulados e foi padronizado para resistir a
eles. Uma infraestrutura quântica deve assinar o que produz com algo que o
próprio quântico não quebra.

    from quirios_engine.seguranca import gerar_chaves, Assinador, Verificador
    privada, publica = gerar_chaves("chaves/")             # uma vez
    Assinador(privada).assinar_arquivo("modelo.json")      # cria modelo.json.sig
    Verificador(publica).verificar_arquivo("modelo.json")  # True / False

    clf.salvar("modelo.json", assinador=Assinador(privada))
    VQCClassifier.carregar("modelo.json", verificador=Verificador(publica))

Exige cryptography >= 48 (extra [pqc]). Guarde a chave privada fora do
repositório; a pública pode ser distribuída.
"""
import base64
from pathlib import Path

ALGORITMO = "ML-DSA-65"


def _mldsa():
    try:
        from cryptography.hazmat.primitives.asymmetric import mldsa
    except ImportError as e:
        raise ImportError('assinatura pós-quântica exige cryptography>=48: pip install -e ".[pqc]"') from e
    return mldsa


def gerar_chaves(pasta, nome="quirios"):
    """Gera um par ML-DSA-65 em PEM. Devolve (caminho_privada, caminho_publica)."""
    from cryptography.hazmat.primitives import serialization as s
    chave = _mldsa().MLDSA65PrivateKey.generate()
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    priv, pub = pasta / f"{nome}_mldsa65.pem", pasta / f"{nome}_mldsa65.pub.pem"
    priv.write_bytes(chave.private_bytes(s.Encoding.PEM, s.PrivateFormat.PKCS8, s.NoEncryption()))
    pub.write_bytes(chave.public_key().public_bytes(s.Encoding.PEM, s.PublicFormat.SubjectPublicKeyInfo))
    return priv, pub


def _caminho_sig(caminho):
    return Path(str(caminho) + ".sig")


class Assinador:
    def __init__(self, chave_privada):
        from cryptography.hazmat.primitives import serialization as s
        _mldsa()
        self._chave = s.load_pem_private_key(Path(chave_privada).read_bytes(), password=None)
        if type(self._chave).__name__ != "MLDSA65PrivateKey":
            raise ValueError(f"a chave precisa ser {ALGORITMO}")

    def assinar(self, dados: bytes) -> bytes:
        return self._chave.sign(dados)

    def assinar_arquivo(self, caminho):
        """Escreve <arquivo>.sig com a assinatura em base64."""
        sig = self.assinar(Path(caminho).read_bytes())
        p = _caminho_sig(caminho)
        p.write_text(f"{ALGORITMO}\n{base64.b64encode(sig).decode()}\n", encoding="ascii")
        return p


class Verificador:
    def __init__(self, chave_publica):
        from cryptography.hazmat.primitives import serialization as s
        _mldsa()
        self._chave = s.load_pem_public_key(Path(chave_publica).read_bytes())

    def verificar(self, dados: bytes, assinatura: bytes) -> bool:
        from cryptography.exceptions import InvalidSignature
        try:
            self._chave.verify(assinatura, dados)
            return True
        except InvalidSignature:
            return False

    def verificar_arquivo(self, caminho):
        p = _caminho_sig(caminho)
        if not p.exists():
            return False
        alg, b64 = p.read_text(encoding="ascii").split()
        return alg == ALGORITMO and self.verificar(Path(caminho).read_bytes(), base64.b64decode(b64))
