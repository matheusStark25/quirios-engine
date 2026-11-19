"""
Proteção em camadas do QURIOS (defesa em profundidade, como em bancos):
se uma camada falhar, as outras seguram, e tudo fica registrado de um jeito
que não dá para apagar sem ser notado.

    python -m quirios_engine.protecao chave criar --arquivo chaves.json --escopos leitura treino --dias 90
    python -m quirios_engine.protecao chave listar --arquivo chaves.json
    python -m quirios_engine.protecao chave revogar ID --arquivo chaves.json
    python -m quirios_engine.protecao cofre gerar --pasta cofre/
    python -m quirios_engine.protecao auditoria verificar trilha.jsonl --chave-publica chave.pub.pem

1. Identidade (CofreDeChaves): chaves aleatórias de 256 bits, guardadas só
   como hash SHA-256, com escopos (leitura, treino, admin), validade e
   revogação; a comparação é em tempo constante (não vaza pelo tempo).
2. Contenção (LimitadorTaxa, BloqueioForcaBruta): limite de requisições por
   chave e por IP, e bloqueio temporário do IP depois de várias tentativas de
   autenticação erradas.
3. Integridade (TrilhaAuditoria): cada registro leva o hash do anterior (uma
   cadeia) e a cabeça da cadeia é assinada com ML-DSA-65 (pós-quântico):
   alterar, apagar, reordenar ou truncar registros é detectado.
4. Sigilo (Cofre): arquivos cifrados com AES-256-GCM; a chave de cada
   arquivo é embrulhada com ML-KEM-768 (NIST FIPS 203, pós-quântico). Um
   computador quântico que quebre RSA não abre o que o QURIOS guardou hoje.
5. Agentes de IA (Guarda): limites de tamanho e de chamadas, lista de
   ferramentas permitidas e auditoria de cada chamada ao servidor MCP.

O que o código não garante sozinho e é responsabilidade da implantação: TLS
(HTTPS) na frente da API, rede isolada, chaves num gerenciador de segredos ou
HSM, backups e testes de invasão feitos por terceiros. Nenhum sistema é
impossível de invadir; o objetivo é que uma invasão seja cara, contida e
visível.
"""
import argparse
import base64
import datetime
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from collections import deque
from pathlib import Path

ESCOPOS = ("leitura", "treino", "admin")
CHAVE_VALIDA = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
MAX_IDENTIDADES = 10_000          # contenção: identidades (IPs, chaves) acompanhadas ao mesmo tempo


def _agora():
    return datetime.datetime.now(datetime.timezone.utc)


def _gravar_atomico(caminho, texto):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    tmp = caminho.with_suffix(caminho.suffix + ".tmp")
    tmp.write_text(texto, encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)                          # só o dono lê (onde o sistema permite)
    except OSError:                                   # pragma: no cover
        pass
    os.replace(tmp, caminho)


# ── 1. identidade ───────────────────────────────────────────────────────────

class CofreDeChaves:
    """Chaves de API guardadas só como hash. A chave em si aparece uma vez, ao ser criada."""

    def __init__(self, caminho):
        self.caminho = Path(caminho)
        self._trava = threading.Lock()

    def _ler(self):
        if not self.caminho.exists():
            return {"formato": "quirios.chaves", "chaves": []}
        return json.loads(self.caminho.read_text(encoding="utf-8"))

    @staticmethod
    def _hash(chave):
        return hashlib.sha256(chave.encode()).hexdigest()

    def criar(self, escopos=("leitura", "treino"), dias=None, nome=""):
        escopos = list(escopos)
        if not escopos or any(e not in ESCOPOS for e in escopos):
            raise ValueError(f"escopos devem estar em {ESCOPOS}")
        chave = "qur_" + secrets.token_urlsafe(32)                    # 256 bits aleatórios
        ident = secrets.token_hex(6)
        with self._trava:
            d = self._ler()
            d["chaves"].append({"id": ident, "nome": nome, "hash": self._hash(chave), "escopos": escopos,
                                "criada": _agora().isoformat(timespec="seconds"),
                                "expira": (_agora() + datetime.timedelta(days=dias)).isoformat(timespec="seconds")
                                if dias else None, "revogada": False})
            _gravar_atomico(self.caminho, json.dumps(d, indent=1, ensure_ascii=False))
        return chave, ident

    def revogar(self, ident):
        with self._trava:
            d = self._ler()
            achou = False
            for e in d["chaves"]:
                if e["id"] == ident:
                    e["revogada"], achou = True, True
            if not achou:
                raise KeyError(ident)
            _gravar_atomico(self.caminho, json.dumps(d, indent=1, ensure_ascii=False))

    def listar(self):
        return [{k: v for k, v in e.items() if k != "hash"} for e in self._ler()["chaves"]]

    def ativas(self):
        """Quantas chaves valem agora (nem revogadas nem vencidas)."""
        agora = _agora()
        return sum(1 for e in self._ler()["chaves"] if not e.get("revogada")
                   and not (e.get("expira") and datetime.datetime.fromisoformat(e["expira"]) < agora))

    def autenticar(self, chave):
        """A entrada da chave (sem o hash) ou None. Percorre TODAS as chaves e compara em
        tempo constante, para o tempo de resposta não revelar nada."""
        if not chave or len(chave) > 200:
            return None
        h, achada = self._hash(chave), None
        for e in self._ler()["chaves"]:
            if hmac.compare_digest(h, e["hash"]):
                achada = e
        if achada is None or achada["revogada"]:
            return None
        if achada["expira"] and datetime.datetime.fromisoformat(achada["expira"]) < _agora():
            return None
        return {k: v for k, v in achada.items() if k != "hash"}


# ── 2. contenção ────────────────────────────────────────────────────────────

class LimitadorTaxa:
    """Balde de fichas por identidade: `por_minuto` em média, rajadas de até `rajada`.
    A memória é limitada: com IPs trocados sem parar (IPv6), baldes já cheios são
    esquecidos (esquecer um balde cheio não muda nada) e, no limite, os mais antigos."""

    def __init__(self, por_minuto=120, rajada=30):
        self.taxa, self.rajada = por_minuto / 60.0, float(rajada)
        self._baldes, self._trava = {}, threading.Lock()

    def _podar(self, agora):
        self._baldes = {k: (f, t) for k, (f, t) in self._baldes.items() if f + (agora - t) * self.taxa < self.rajada}
        while len(self._baldes) > 0.9 * MAX_IDENTIDADES:     # ainda cheio de baldes ativos: saem os mais antigos
            self._baldes.pop(next(iter(self._baldes)))       # (até 90%, para a poda não rodar a cada pedido)

    def permitir(self, ident):
        """(permitido, segundos até a próxima ficha)."""
        agora = time.monotonic()
        with self._trava:
            if ident not in self._baldes and len(self._baldes) >= MAX_IDENTIDADES:
                self._podar(agora)
            fichas, t = self._baldes.get(ident, (self.rajada, agora))
            fichas = min(self.rajada, fichas + (agora - t) * self.taxa)
            if fichas >= 1:
                self._baldes[ident] = (fichas - 1, agora)
                return True, 0.0
            self._baldes[ident] = (fichas, agora)
            return False, (1 - fichas) / self.taxa


class BloqueioForcaBruta:
    """Bloqueia um IP por `bloqueio_s` depois de `max_falhas` autenticações erradas em `janela_s`."""

    def __init__(self, max_falhas=10, janela_s=300, bloqueio_s=900):
        self.max_falhas, self.janela_s, self.bloqueio_s = max_falhas, janela_s, bloqueio_s
        self._falhas, self._bloqueados, self._trava = {}, {}, threading.Lock()

    def bloqueado(self, ip):
        with self._trava:
            fim = self._bloqueados.get(ip)
            if fim and fim > time.monotonic():
                return fim - time.monotonic()
            self._bloqueados.pop(ip, None)
            return 0.0

    def _podar(self, agora):
        """Memória limitada: falhas fora da janela e bloqueios vencidos são esquecidos;
        no limite, os registros mais antigos saem primeiro."""
        self._falhas = {k: q for k, q in self._falhas.items() if q and agora - q[-1] <= self.janela_s}
        self._bloqueados = {k: fim for k, fim in self._bloqueados.items() if fim > agora}
        for d in (self._falhas, self._bloqueados):
            while len(d) > 0.9 * MAX_IDENTIDADES:
                d.pop(next(iter(d)))

    def falhou(self, ip):
        agora = time.monotonic()
        with self._trava:
            if ip not in self._falhas and (len(self._falhas) >= MAX_IDENTIDADES
                                           or len(self._bloqueados) >= MAX_IDENTIDADES):
                self._podar(agora)
            q = self._falhas.setdefault(ip, deque())
            q.append(agora)
            while q and agora - q[0] > self.janela_s:
                q.popleft()
            if len(q) >= self.max_falhas:
                self._bloqueados[ip] = agora + self.bloqueio_s
                q.clear()

    def sucesso(self, ip):
        with self._trava:
            self._falhas.pop(ip, None)


# ── 3. integridade ──────────────────────────────────────────────────────────

def _hash_registro(r):
    corpo = {k: v for k, v in r.items() if k != "hash"}
    return hashlib.sha256(json.dumps(corpo, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class TrilhaAuditoria:
    """Registro só de acréscimo, encadeado por hash; com assinador, a cabeça da cadeia
    (último número e último hash) é assinada com ML-DSA a cada registro."""

    def __init__(self, caminho, assinador=None):
        from .evolucao import _TravaArquivo
        self.caminho, self.assinador = Path(caminho), assinador
        self.cabeca = Path(str(caminho) + ".cabeca.json")
        self._TravaArquivo = _TravaArquivo

    def _ler_ultimo(self):
        """Último registro lendo só o FIM do arquivo (a trilha cresce para sempre; reler
        tudo a cada requisição deixaria a API mais lenta a cada dia)."""
        seq, h = 0, "0" * 64
        if not self.caminho.exists():
            return seq, h
        with open(self.caminho, "rb") as f:
            fim = f.seek(0, os.SEEK_END)
            bloco, cauda = 4096, b""
            while fim > 0:
                ini = max(0, fim - bloco)
                f.seek(ini)
                cauda = f.read(fim - ini) + cauda
                fim = ini
                linhas = [l for l in cauda.split(b"\n") if l.strip()]
                if len(linhas) >= 2 or (linhas and fim == 0):
                    r = json.loads(linhas[-1])
                    return r["seq"], r["hash"]
        return seq, h

    def registrar(self, evento):
        with self._TravaArquivo(self.caminho):
            seq, anterior = self._ler_ultimo()
            r = dict(evento, seq=seq + 1, ts=_agora().isoformat(timespec="milliseconds"), anterior=anterior)
            r["hash"] = _hash_registro(r)
            self.caminho.parent.mkdir(parents=True, exist_ok=True)
            with open(self.caminho, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(r, ensure_ascii=True) + "\n")    # ASCII puro:   etc. não quebram linha
            if self.assinador is not None:
                msg = f"{r['seq']}:{r['hash']}".encode()
                _gravar_atomico(self.cabeca, json.dumps(
                    {"seq": r["seq"], "hash": r["hash"],
                     "assinatura": base64.b64encode(self.assinador.assinar(msg)).decode()}))
        return r


def verificar_trilha(caminho, verificador=None):
    """{'ok', 'registros', 'problema'}: refaz a cadeia inteira e, com verificador,
    confere a assinatura da cabeça (pega truncamento do fim)."""
    caminho = Path(caminho)
    if not caminho.exists():
        return {"ok": False, "registros": 0, "problema": "trilha inexistente"}
    anterior, n = "0" * 64, 0
    linhas = caminho.read_bytes().split(b"\n")              # só \n separa registros (não  , \x85...)
    if linhas and not linhas[-1].strip():
        linhas.pop()
    for i, linha in enumerate(linhas, 1):
        try:
            r = json.loads(linha)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {"ok": False, "registros": n, "problema": f"linha {i} corrompida"}
        if not isinstance(r, dict):
            return {"ok": False, "registros": n, "problema": f"linha {i} não é um registro"}
        if r.get("seq") != i or r.get("anterior") != anterior or _hash_registro(r) != r.get("hash"):
            return {"ok": False, "registros": n, "problema": f"registro {i} alterado, removido ou fora de ordem"}
        anterior, n = r["hash"], i
    if verificador is not None:
        cab_p = Path(str(caminho) + ".cabeca.json")
        if not cab_p.exists():
            return {"ok": False, "registros": n, "problema": "cabeça assinada ausente"}
        cab = json.loads(cab_p.read_text(encoding="utf-8"))
        if not verificador.verificar(f"{cab['seq']}:{cab['hash']}".encode(), base64.b64decode(cab["assinatura"])):
            return {"ok": False, "registros": n, "problema": "assinatura da cabeça inválida"}
        if cab["seq"] != n or cab["hash"] != anterior:
            return {"ok": False, "registros": n, "problema": f"trilha truncada: a cabeça assinada é o registro "
                                                            f"{cab['seq']}, o arquivo tem {n}"}
    return {"ok": True, "registros": n, "problema": None}


# ── 4. sigilo ───────────────────────────────────────────────────────────────

class Cofre:
    """Envelope híbrido pós-quântico: AES-256-GCM nos dados; a chave AES nasce de um
    encapsulamento ML-KEM-768 para a chave pública do cofre. Só quem tem a privada abre."""

    FORMATO = "quirios.cofre.v1"

    def __init__(self, publica=None, privada=None):
        from cryptography.hazmat.primitives import serialization as s
        self._pub = s.load_pem_public_key(Path(publica).read_bytes()) if publica else None
        self._priv = s.load_pem_private_key(Path(privada).read_bytes(), password=None) if privada else None
        if self._priv is not None and self._pub is None:
            self._pub = self._priv.public_key()

    @staticmethod
    def gerar_chaves(pasta, nome="quirios"):
        from cryptography.hazmat.primitives import serialization as s
        from cryptography.hazmat.primitives.asymmetric import mlkem
        k = mlkem.MLKEM768PrivateKey.generate()
        pasta = Path(pasta)
        pasta.mkdir(parents=True, exist_ok=True)
        priv, pub = pasta / f"{nome}_mlkem768.pem", pasta / f"{nome}_mlkem768.pub.pem"
        priv.write_bytes(k.private_bytes(s.Encoding.PEM, s.PrivateFormat.PKCS8, s.NoEncryption()))
        try:
            os.chmod(priv, 0o600)
        except OSError:                               # pragma: no cover
            pass
        pub.write_bytes(k.public_key().public_bytes(s.Encoding.PEM, s.PublicFormat.SubjectPublicKeyInfo))
        return priv, pub

    def cifrar(self, dados: bytes, contexto=b"") -> bytes:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        if self._pub is None:
            raise ValueError("cofre sem chave pública")
        segredo, ct_kem = self._pub.encapsulate()
        nonce = secrets.token_bytes(12)
        aad = self.FORMATO.encode() + b"|" + contexto
        cifrado = AESGCM(segredo).encrypt(nonce, dados, aad)
        return json.dumps({"formato": self.FORMATO, "kem": "ML-KEM-768", "aead": "AES-256-GCM",
                           "contexto": contexto.decode(errors="replace"),
                           "kem_ct": base64.b64encode(ct_kem).decode(), "nonce": base64.b64encode(nonce).decode(),
                           "dados": base64.b64encode(cifrado).decode()}).encode()

    def decifrar(self, envelope: bytes, contexto=b"") -> bytes:
        """Falha (InvalidTag) se o envelope foi alterado, se o contexto não é o mesmo ou se a chave é outra."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        if self._priv is None:
            raise ValueError("cofre sem chave privada")
        e = json.loads(envelope)
        if e.get("formato") != self.FORMATO:
            raise ValueError("não é um envelope do cofre do QURIOS")
        segredo = self._priv.decapsulate(base64.b64decode(e["kem_ct"]))
        aad = self.FORMATO.encode() + b"|" + contexto
        return AESGCM(segredo).decrypt(base64.b64decode(e["nonce"]), base64.b64decode(e["dados"]), aad)


# ── 5. agentes de IA (servidor MCP) ─────────────────────────────────────────

class Guarda:
    """Protege ferramentas chamadas por agentes de IA: lista do que é permitido,
    orçamento de chamadas por hora, limites de tamanho e auditoria de cada chamada.
    O texto que um agente envia é sempre DADO: nenhuma ferramenta executa código,
    abre caminhos de arquivo ou muda configuração a partir dele."""

    def __init__(self, permitidas=None, chamadas_por_hora=600, max_amostras=5000, max_features=256, trilha=None):
        self.permitidas = set(permitidas) if permitidas else None
        self.limite = LimitadorTaxa(chamadas_por_hora, rajada=max(5, chamadas_por_hora // 20))
        self.max_amostras, self.max_features, self.trilha = max_amostras, max_features, trilha

    @classmethod
    def do_ambiente(cls):
        perm = os.environ.get("QURIOS_MCP_FERRAMENTAS")
        trilha = os.environ.get("QURIOS_AUDITORIA")
        return cls(permitidas=[p.strip() for p in perm.split(",")] if perm else None,
                   chamadas_por_hora=int(os.environ.get("QURIOS_MCP_CHAMADAS_HORA", "600")),
                   trilha=TrilhaAuditoria(trilha) if trilha else None)

    def checar_dados(self, X=None, chave=None):
        if X is not None:
            if not isinstance(X, (list, tuple)) or len(X) > self.max_amostras:
                raise ValueError(f"X deve ter no máximo {self.max_amostras} amostras")
            if any(isinstance(a, (list, tuple)) and len(a) > self.max_features for a in X):
                raise ValueError(f"cada amostra deve ter no máximo {self.max_features} features")
        if chave is not None and not CHAVE_VALIDA.match(str(chave)):
            raise ValueError("chave deve ter de 1 a 64 caracteres entre letras, números e _ . : -")

    def proteger(self, fn):
        import functools

        @functools.wraps(fn)
        def envolvida(*args, **kwargs):
            nome, t0, status = fn.__name__, time.perf_counter(), "ok"
            try:
                if self.permitidas is not None and nome not in self.permitidas:
                    raise PermissionError(f"ferramenta '{nome}' não está liberada neste servidor")
                ok, espera = self.limite.permitir("mcp")
                if not ok:
                    raise PermissionError(f"limite de chamadas atingido; tente em {espera:.0f}s")
                self.checar_dados(kwargs.get("X"), kwargs.get("chave"))
                return fn(*args, **kwargs)
            except Exception as e:
                status = f"erro: {type(e).__name__}"
                raise
            finally:
                if self.trilha is not None:
                    self.trilha.registrar({"origem": "mcp", "ferramenta": nome, "status": status,
                                           "ms": round(1000 * (time.perf_counter() - t0), 1),
                                           "amostras": len(kwargs["X"]) if isinstance(kwargs.get("X"), list) else None})
        return envolvida


# ── linha de comando ────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m quirios_engine.protecao", description="Proteção em camadas do QURIOS.")
    sub = ap.add_subparsers(dest="area", required=True)
    pc = sub.add_parser("chave").add_subparsers(dest="acao", required=True)
    c1 = pc.add_parser("criar")
    c1.add_argument("--arquivo", required=True)
    c1.add_argument("--escopos", nargs="+", default=["leitura", "treino"], choices=ESCOPOS)
    c1.add_argument("--dias", type=int, default=None)
    c1.add_argument("--nome", default="")
    c2 = pc.add_parser("listar")
    c2.add_argument("--arquivo", required=True)
    c3 = pc.add_parser("revogar")
    c3.add_argument("id")
    c3.add_argument("--arquivo", required=True)
    pk = sub.add_parser("cofre").add_subparsers(dest="acao", required=True)
    k1 = pk.add_parser("gerar")
    k1.add_argument("--pasta", required=True)
    ps = sub.add_parser("assinatura").add_subparsers(dest="acao", required=True)
    s1 = ps.add_parser("gerar")
    s1.add_argument("--pasta", required=True)
    pa = sub.add_parser("auditoria").add_subparsers(dest="acao", required=True)
    a1 = pa.add_parser("verificar")
    a1.add_argument("trilha")
    a1.add_argument("--chave-publica", default=None)
    a = ap.parse_args(argv)
    if a.area == "chave":
        cofre = CofreDeChaves(a.arquivo)
        if a.acao == "criar":
            chave, ident = cofre.criar(a.escopos, a.dias, a.nome)
            print(f"chave {ident} criada com escopos {a.escopos}. Guarde-a agora; ela não será mostrada de novo:\n{chave}")
        elif a.acao == "listar":
            print(json.dumps(cofre.listar(), indent=1, ensure_ascii=False))
        else:
            cofre.revogar(a.id)
            print(f"chave {a.id} revogada")
    elif a.area in ("cofre", "assinatura"):
        if a.area == "cofre":
            priv, pub = Cofre.gerar_chaves(a.pasta)
        else:
            from .seguranca import gerar_chaves
            priv, pub = gerar_chaves(a.pasta)
        print(f"chave privada (guarde em segredo): {priv}\nchave pública: {pub}")
    else:
        ver = None
        if a.chave_publica:
            from .seguranca import Verificador
            ver = Verificador(a.chave_publica)
        r = verificar_trilha(a.trilha, ver)
        print(json.dumps(r, ensure_ascii=False))
        if not r["ok"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
