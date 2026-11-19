"""
API HTTP do QURIOS Engine -- a porta de entrada para software clássico.

    pip install -e ".[api]"
    uvicorn quirios_engine.api:app --reload

Documentação interativa em http://127.0.0.1:8000/docs.

Sem chave o servidor NÃO SOBE: defina uma das duas (toda rota, menos /saude e a
página do painel, exige o cabeçalho X-API-Key):
    QURIOS_CHAVES_API          cofre de chaves com escopos (python -m quirios_engine.protecao chave criar)
    QURIOS_API_KEY             uma chave única com acesso total (16 caracteres ou mais)

Variáveis de ambiente opcionais:
    QURIOS_AUDITORIA           trilha de auditoria encadeada (assinada se houver QURIOS_CHAVE_PQC)
    QURIOS_COFRE_PUBLICA/_PRIVADA  chaves ML-KEM-768: modelos persistidos ficam cifrados
    QURIOS_REQ_MINUTO, QURIOS_RAJADA, QURIOS_MAX_FALHAS, QURIOS_BLOQUEIO_S, QURIOS_MAX_CORPO_MB
    QURIOS_DIR_MODELOS         pasta onde modelos VQC treinados são salvos e recarregados
                               ao reiniciar (sem ela, ficam só em memória)
    QURIOS_TRABALHADORES       threads para tarefas assíncronas (padrão 4)
    QURIOS_REGISTRO            registro de campeões da auto-evolução
                               (padrão campeoes_quirios.json)
    QURIOS_CHAVE_PQC           chave privada ML-DSA: assina modelos e registro
    QURIOS_CHAVE_PQC_PUBLICA   chave pública ML-DSA: só carrega o que estiver assinado

Treino pesado: mande "assincrono": true -- a resposta traz o id de uma
tarefa; consulte GET /tarefas/{id} até "concluida". Hardware IBM não é
exposto aqui (tokens ficam fora da API); use BackendQiskit direto.
"""
import hashlib
import hmac
import json
import os
from contextlib import asynccontextmanager
import threading
import time
import uuid
from pathlib import Path
from typing import Literal, Optional

import numpy as np

try:
    from fastapi import Depends, FastAPI, Header, HTTPException, Request
    from fastapi.encoders import jsonable_encoder
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    from pydantic import BaseModel, ConfigDict, Field, model_validator
    from starlette.concurrency import run_in_threadpool
except ImportError as e:                    # pragma: no cover
    raise ImportError('A API exige as dependências extras: pip install -e ".[api]"') from e

from . import __version__
from .backends import BackendExato, BackendMPS, dispositivos_disponiveis
from .core import MAX_QUBITS_VETORIAL, QuantumConfig
from .entropy import entropia_emaranhamento, entropia_medicao, pureza
from .orquestrador import Orquestrador, candidatos_padrao, escolher_backend
from .sistema import RFF, Brutos, ClassificadorQURIOS, Quantico
from .protecao import BloqueioForcaBruta, CofreDeChaves, LimitadorTaxa, TrilhaAuditoria
from .tarefas import Registro, blas_em_uma_thread, em_segundo_plano
from .vqc import VQCClassifier

# Limite de qubits por backend na API (treino síncrono precisa caber numa requisição)
MAX_QUBITS = {"exato": 12, "mps": 64, "auto": 64}


# ── esquemas ────────────────────────────────────────────────────────────────

class _Entrada(BaseModel):
    """Base das entradas: NaN e infinito são recusados (o JSON do Python os aceita)."""
    model_config = ConfigDict(allow_inf_nan=False)


class ConfigCircuito(_Entrada):
    n_qubits: int = Field(8, ge=2, le=max(MAX_QUBITS.values()))
    topologia: Literal["anel", "linear", "todos_pares", "blocos"] = "anel"
    n_camadas: int = Field(2, ge=0, le=8, description="camadas de entrelaçamento (CNOTs)")
    tam_bloco: int = Field(8, ge=2)
    n_conexoes: int = Field(3, ge=1)

    def construir(self):
        return QuantumConfig(self.n_qubits, self.topologia, self.n_camadas,
                             tam_bloco=self.tam_bloco, n_conexoes=self.n_conexoes)


def _checar_X(X, n_features, codificacao, n_qubits):
    if any(len(linha) != len(X[0]) for linha in X):
        raise ValueError("todas as linhas de X devem ter o mesmo número de features")
    if n_features is not None and len(X[0]) != n_features:
        raise ValueError(f"cada linha de X deve ter {n_features} features")
    A = np.asarray(X, float)
    if codificacao in ("angulo", "iqp"):
        if A.size and (A.min() < 0 or A.max() > 1):
            raise ValueError(f"codificação '{codificacao}': valores de X devem estar em [0, 1]")
    elif A.shape[1] + 1 > (1 << n_qubits):            # +1: amplitude constante (ver Ansatz)
        raise ValueError(f"amplitude: {A.shape[1]} features não cabem em {n_qubits} qubits "
                         f"(máximo {(1 << n_qubits) - 1})")


class _ComDados(_Entrada):
    circuito: ConfigCircuito = Field(default_factory=ConfigCircuito)
    backend: Literal["exato", "mps", "auto"] = Field(
        "exato", description="exato = vetor de estado; mps = tensor network (50-100+ qubits); "
                             "auto = o QURIOS escolhe pelo tamanho e entrelaçamento")
    X: list[list[float]] = Field(..., min_length=4)
    y: list[int] = Field(..., min_length=4)
    seed: int = 0
    assincrono: bool = Field(False, description="true: responde na hora com o id de uma tarefa")

    def _validar_dados(self, codificacao):
        if len(self.X) != len(self.y):
            raise ValueError("X e y devem ter o mesmo número de amostras")
        if set(self.y) != {0, 1}:
            raise ValueError("y deve ser binário e conter as duas classes (0 e 1)")
        limite = MAX_QUBITS[self.backend]
        if self.circuito.n_qubits > limite:
            raise ValueError(f"backend '{self.backend}' aceita até {limite} qubits na API")
        _checar_X(self.X, None, codificacao, self.circuito.n_qubits)


class PedidoTreino(_ComDados):
    tipo: Literal["vqc", "sistema"] = Field(
        "vqc", description="vqc = VQCClassifier treinado; "
                           "sistema = ClassificadorQURIOS (Brutos, RFF, Quantico por validação cruzada)")
    epocas: int = Field(20, ge=1, le=500, description="só para tipo=vqc")
    lr: float = Field(0.3, gt=0, description="só para tipo=vqc com parameter_shift")
    camada_pos: bool = Field(True, description="só para tipo=vqc")
    otimizador: Literal["parameter_shift", "spsa"] = "parameter_shift"
    codificacao: Literal["angulo", "iqp", "amplitude"] = "angulo"
    camadas_dados: Optional[int] = Field(None, ge=1, le=16,
                                         description="camadas de re-uploading (None = ceil(d / n_qubits))")
    inicializacao: Literal["padrao", "pequena"] = "padrao"
    treino_por_camadas: bool = False
    rotacoes: list[Literal["rx", "ry", "rz"]] = Field(["ry", "rz"], min_length=1, max_length=4)
    emaranhador: Literal["cnot", "cz", "rzz"] = Field(
        "cnot", description="rzz = porta fracionária treinável (nativa nos processadores IBM Heron)")
    leitura: Literal["z", "z_zz", "pauli", "pauli2"] = "z"
    escala_rff: float = Field(1.0, gt=0, description="só para tipo=sistema")

    @model_validator(mode="after")
    def _validar(self):
        self._validar_dados(self.codificacao)
        return self


class PedidoOrquestrar(_ComDados):
    politica: Literal["evidencia", "melhor_media"] = "evidencia"
    alfa: float = Field(0.05, gt=0, lt=1)
    repeticoes: int = Field(3, ge=1, le=10)
    incluir_vqc: bool = Field(False, description="inclui um VQC com SPSA entre os candidatos")
    incluir_kernel: bool = Field(False, description="inclui o kernel quântico projetado")
    chave_campeao: Optional[str] = Field(None, description="inclui o campeão da auto-evolução deste problema")

    @model_validator(mode="after")
    def _validar(self):
        self._validar_dados("angulo")
        return self


class PedidoEvoluir(_ComDados):
    chave: str = Field(..., min_length=1, max_length=120, description="nome do problema no registro")
    geracoes: int = Field(4, ge=1, le=50)
    populacao: int = Field(8, ge=3, le=60)
    modo: Literal["reservatorio", "kernel"] = "reservatorio"
    motor: Literal["quantico", "classico"] = Field(
        "quantico", description="quantico = a população evolui em superposição num circuito gerador")
    assincrono: bool = Field(True, description="a busca é longa: por padrão vira tarefa")

    @model_validator(mode="after")
    def _validar(self):
        self._validar_dados("angulo")
        if self.circuito.n_qubits > 16:
            raise ValueError("a auto-evolução pela API aceita até 16 qubits")
        return self


class PedidoDadosQuanticos(_Entrada):
    n_spins: int = Field(8, ge=3, le=10)
    n_amostras: int = Field(160, ge=20, le=400)
    shots: list[int] = Field([10, 100], min_length=1, max_length=5)
    h_min: float = Field(0.7, gt=0)
    h_max: float = Field(1.4, gt=0)
    n_treino: int = Field(40, ge=8)
    repeticoes: int = Field(5, ge=1, le=20)
    seed: int = 0
    assincrono: bool = True

    @model_validator(mode="after")
    def _validar(self):
        if self.h_min >= self.h_max:
            raise ValueError("h_min deve ser menor que h_max")
        if self.n_treino >= self.n_amostras - 4:
            raise ValueError("n_treino deve deixar amostras para teste")
        return self


class PedidoPainel(_Entrada):
    dataset: Literal["sintetico", "breast_cancer", "wine", "iris"] = "sintetico"
    n_qubits: int = Field(6, ge=2, le=8)
    geracoes: int = Field(6, ge=1, le=20)
    populacao: int = Field(12, ge=4, le=48)
    emaranhamento: bool = True
    seed: int = 0


class PedidoProjetar(_ComDados):
    geracoes: list[str] = Field(["hoje", "proxima", "dez_vezes", "tolerante_a_falhas"], min_length=1)
    shots: int = Field(2000, ge=100, le=100_000)

    @model_validator(mode="after")
    def _validar(self):
        self._validar_dados("angulo")
        if self.circuito.n_qubits > 10:
            raise ValueError("a projeção de hardware pela API aceita até 10 qubits (simulação com ruído)")
        return self


class PedidoPrever(_Entrada):
    X: list[list[float]] = Field(..., min_length=1)


class PedidoEntropia(_Entrada):
    x: list[float]
    subsistema: list[int] = Field([0], min_length=1,
                                  description="qubits do subsistema A; o resto é traçado fora")


# ── estado do servidor ──────────────────────────────────────────────────────

blas_em_uma_thread()        # requisições, tarefas e sessões do painel rodam em threads paralelas


@asynccontextmanager
async def _ciclo(_app):
    verificar_chaves_configuradas()          # sem chave, o uvicorn para aqui com a mensagem de como criar
    yield


app = FastAPI(
    lifespan=_ciclo,
    title="QURIOS Engine",
    version=__version__,
    description="Infraestrutura entre software clássico e quântico: treino de modelos quânticos "
                "variacionais, orquestração clássico/quântico com evidência estatística e tarefas "
                "assíncronas.",
)


def _finito(o):
    if isinstance(o, float) and not np.isfinite(o):
        return str(o)
    if isinstance(o, dict):
        return {k: _finito(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finito(v) for v in o]
    return o


def _cabecalhos_seguranca(resp, request):
    h = resp.headers
    h.setdefault("X-Content-Type-Options", "nosniff")
    h.setdefault("X-Frame-Options", "DENY")
    h.setdefault("Referrer-Policy", "no-referrer")
    h.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    h.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    if request.url.path == "/painel":
        h.setdefault("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                     "style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; "
                     "frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
    elif request.url.path not in ("/docs", "/redoc"):
        h.setdefault("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'")
        h.setdefault("Cache-Control", "no-store")
    if request.url.scheme == "https":
        h.setdefault("Strict-Transport-Security", "max-age=63072000; includeSubDomains")
    return resp


@app.middleware("http")
async def _blindagem(request: Request, call_next):
    """Antes de qualquer rota: IP bloqueado por força bruta, tamanho do corpo, identidade
    e limite de taxa. Depois: cabeçalhos de segurança e um registro na trilha de auditoria."""
    t0 = time.perf_counter()
    ip = request.client.host if request.client else "?"
    ident = None
    espera = _bloqueio.bloqueado(ip)
    tamanho = request.headers.get("content-length")
    if espera:
        resp = JSONResponse({"detail": "muitas tentativas de autenticação; IP bloqueado temporariamente"},
                            status_code=429, headers={"Retry-After": str(int(espera) + 1)})
    elif request.method in ("POST", "PUT", "PATCH") and (tamanho is None or "transfer-encoding" in request.headers):
        # corpo em pedaços (chunked) não diz o tamanho antes: aceitar furaria o limite abaixo
        resp = JSONResponse({"detail": "envie o corpo com Content-Length"}, status_code=411)
    elif tamanho is not None and not tamanho.isdigit():
        resp = JSONResponse({"detail": "Content-Length inválido"}, status_code=400)
    elif int(tamanho or 0) > MAX_CORPO:
        resp = JSONResponse({"detail": f"requisição maior que {MAX_CORPO // 2 ** 20} MB"}, status_code=413)
    else:
        ident = _identidade(request.headers.get("x-api-key"))
        if ident is None and request.url.path not in ROTAS_PUBLICAS:
            _bloqueio.falhou(ip)
            resp = JSONResponse({"detail": "X-API-Key ausente ou inválida"}, status_code=401)
        else:
            ok, espera = _limite.permitir(ident["id"] if ident else f"ip:{ip}")
            if not ok:
                resp = JSONResponse({"detail": "limite de requisições atingido"}, status_code=429,
                                    headers={"Retry-After": str(int(espera) + 1)})
            else:
                if ident is not None:
                    _bloqueio.sucesso(ip)
                request.state.identidade = ident
                resp = await call_next(request)
    _cabecalhos_seguranca(resp, request)
    trilha = _trilha()
    if trilha is not None:                # gravar e assinar é bloqueante: fora do laço de eventos
        await run_in_threadpool(trilha.registrar, {
            "origem": "api", "ip": ip, "metodo": request.method, "rota": request.url.path,
            "status": resp.status_code, "quem": ident["id"] if ident else None,
            "ms": round(1000 * (time.perf_counter() - t0), 1)})
    return resp


@app.exception_handler(RequestValidationError)
async def _erro_validacao(request, exc):
    # O handler padrão repete a entrada no erro; com NaN ela não vira JSON e a conexão caía sem resposta.
    return JSONResponse(status_code=422, content={"detail": _finito(jsonable_encoder(exc.errors()))})
_modelos = {}
_rejeitados = {}          # modelos persistidos recusados no carregamento (assinatura inválida etc.)
_trava = threading.Lock()
_tarefas = Registro()


# ── proteção em camadas (protecao.py) ───────────────────────────────────────

ROTAS_PUBLICAS = {"/saude", "/painel", "/docs", "/openapi.json", "/redoc"}
_ORDEM_ESCOPO = {"leitura": 0, "treino": 1, "admin": 2}
_bloqueio = BloqueioForcaBruta(int(os.environ.get("QURIOS_MAX_FALHAS", "10")), 300,
                               int(os.environ.get("QURIOS_BLOQUEIO_S", "900")))
_limite = LimitadorTaxa(int(os.environ.get("QURIOS_REQ_MINUTO", "600")), int(os.environ.get("QURIOS_RAJADA", "120")))
MAX_CORPO = int(float(os.environ.get("QURIOS_MAX_CORPO_MB", "16")) * 2 ** 20)


MIN_CHAVE_UNICA = 16
_COMO_CRIAR = ("crie uma chave com `python -m quirios_engine.protecao chave criar --arquivo chaves.json` e "
               "aponte QURIOS_CHAVES_API para o arquivo (ou defina QURIOS_API_KEY com "
               f"{MIN_CHAVE_UNICA} caracteres ou mais)")


def verificar_chaves_configuradas():
    """Sem chave o QURIOS não sobe: chamada na partida do servidor. Recusa também
    chave única curta demais e cofre de chaves sem nenhuma chave valendo."""
    unica, arquivo = os.environ.get("QURIOS_API_KEY"), os.environ.get("QURIOS_CHAVES_API")
    if not unica and not arquivo:
        raise RuntimeError(f"sem chave o QURIOS não sobe: {_COMO_CRIAR}")
    if unica and len(unica) < MIN_CHAVE_UNICA:
        raise RuntimeError(f"QURIOS_API_KEY curta demais (mínimo {MIN_CHAVE_UNICA} caracteres): o QURIOS não sobe")
    if arquivo:
        try:
            ativas = CofreDeChaves(arquivo).ativas()
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise RuntimeError(f"arquivo de chaves ilegível ({arquivo}): o QURIOS não sobe") from e
        if not ativas and not unica:
            raise RuntimeError(f"nenhuma chave valendo em {arquivo} (revogadas, vencidas ou arquivo "
                               f"inexistente): o QURIOS não sobe; {_COMO_CRIAR}")


def _modo_chaves():
    if os.environ.get("QURIOS_CHAVES_API"):
        return "cofre de chaves"
    return "chave única" if os.environ.get("QURIOS_API_KEY") else "nenhuma: o servidor recusa tudo"


def _identidade(chave):
    """Quem fez a requisição: {'id', 'escopos'}, ou None se a chave não vale.
    Sem chaves configuradas ninguém entra (o servidor nem sobe; isto é a segunda trava)."""
    unica, arquivo = os.environ.get("QURIOS_API_KEY"), os.environ.get("QURIOS_CHAVES_API")
    if not unica and not arquivo:
        return None
    if unica and chave and hmac.compare_digest(hashlib.sha256(chave.encode()).digest(),
                                               hashlib.sha256(unica.encode()).digest()):
        return {"id": "chave-unica", "escopos": list(_ORDEM_ESCOPO)}
    if arquivo:
        return CofreDeChaves(arquivo).autenticar(chave)
    return None


_trilhas = {}


def _trilha():
    p = os.environ.get("QURIOS_AUDITORIA")
    if not p:
        return None
    if p not in _trilhas:
        _trilhas[p] = TrilhaAuditoria(p, assinador=_assinador())
    return _trilhas[p]


def _exigir(escopo):
    def dependencia(request: Request):
        ident = getattr(request.state, "identidade", None)
        if ident is None:
            raise HTTPException(401, "X-API-Key ausente ou inválida")
        if max((_ORDEM_ESCOPO.get(e, -1) for e in ident.get("escopos") or ()), default=-1) < _ORDEM_ESCOPO[escopo]:
            raise HTTPException(403, f"esta chave não tem o escopo '{escopo}'")
    return [Depends(dependencia)]


leitura, treino, admin = _exigir("leitura"), _exigir("treino"), _exigir("admin")
protegido = treino


def _dir_modelos():
    d = os.environ.get("QURIOS_DIR_MODELOS")
    return Path(d) if d else None


def _assinador():
    p = os.environ.get("QURIOS_CHAVE_PQC")
    if not p:
        return None
    from .seguranca import Assinador
    return Assinador(p)


def _verificador():
    p = os.environ.get("QURIOS_CHAVE_PQC_PUBLICA")
    if not p:
        return None
    from .seguranca import Verificador
    return Verificador(p)


def _registro_campeoes():
    from .evolucao import RegistroCampeoes
    return RegistroCampeoes(os.environ.get("QURIOS_REGISTRO", "campeoes_quirios.json"),
                            assinador=_assinador(), verificador=_verificador())


def _resumo(mid, reg):
    r = {"id": mid, "tipo": reg["tipo"], "circuito": reg["circuito"], "backend": reg["backend"],
         "n_features": reg["n_features"], "n_treino": reg["n_treino"],
         "acuracia_treino": reg["acuracia_treino"]}
    m = reg["modelo"]
    if reg["tipo"] == "vqc":
        r["historico_loss"] = [float(v) for v in m.historico_loss]
        r["circuitos_executados"] = int(m.circuitos_executados)
        r["ansatz"] = m.ansatz.para_dict()
        r["otimizador"] = reg.get("otimizador")
    elif reg["tipo"] == "sistema":
        r["relatorio_cv"] = {k: {"media": v[0], "desvio": v[1]} for k, v in m.relatorio_cv.items()}
        r["escolhido"] = m.escolhido.nome
    else:
        r["relatorio"] = m.relatorio
    return r


def _cofre():
    pub, priv = os.environ.get("QURIOS_COFRE_PUBLICA"), os.environ.get("QURIOS_COFRE_PRIVADA")
    if not pub and not priv:
        return None
    from .protecao import Cofre
    return Cofre(pub, priv)


def _registrar(reg):
    mid = uuid.uuid4().hex[:12]
    with _trava:
        _modelos[mid] = reg
    d = _dir_modelos()
    if d is not None and reg["tipo"] == "vqc":
        d.mkdir(parents=True, exist_ok=True)
        cofre, assinador = _cofre(), _assinador()
        if cofre is not None:                         # cifrado em memória: o texto aberto nunca vai ao disco
            p = d / f"{mid}.json.cofre"
            p.write_bytes(cofre.cifrar(reg["modelo"].para_json().encode(), contexto=mid.encode()))
            if assinador is not None:                 # cifra-depois-assina: adulteração é recusada antes de abrir
                assinador.assinar_arquivo(p)
        else:
            reg["modelo"].salvar(d / f"{mid}.json", assinador=assinador)
        meta = {k: reg[k] for k in ("tipo", "circuito", "backend", "n_features", "n_treino",
                                    "acuracia_treino", "otimizador", "codificacao")}
        (d / f"{mid}.meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return mid


def carregar_persistidos(pasta=None):
    """Recarrega modelos VQC salvos em QURIOS_DIR_MODELOS (chamado no start)."""
    d = Path(pasta) if pasta else _dir_modelos()
    if d is None or not d.exists():
        return 0
    n = 0
    verificador, cofre = _verificador(), _cofre()
    for meta_p in d.glob("*.meta.json"):
        mid = meta_p.name[:-len(".meta.json")]
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            cifrado = d / f"{mid}.json.cofre"
            if cifrado.exists():
                if cofre is None:
                    raise PermissionError("modelo cifrado e o servidor não tem a chave do cofre")
                if verificador is not None and not verificador.verificar_arquivo(cifrado):
                    raise ValueError("assinatura pós-quântica inválida ou ausente")
                texto = cofre.decifrar(cifrado.read_bytes(), contexto=mid.encode()).decode()
                modelo = VQCClassifier.carregar(None, conteudo=texto)
            else:
                if cofre is not None:
                    raise PermissionError("modelo em texto aberto num servidor que exige cifra")
                modelo = VQCClassifier.carregar(d / f"{mid}.json", verificador=verificador)
        except Exception as e:        # adulterado/corrompido: recusa este, o servidor sobe
            _rejeitados[mid] = f"{type(e).__name__}: {e}"
            continue
        with _trava:
            _modelos[mid] = dict(meta, modelo=modelo)
        n += 1
    return n


carregar_persistidos()


def _buscar(mid):
    with _trava:
        reg = _modelos.get(mid)
    if reg is None:
        raise HTTPException(404, f"modelo '{mid}' não encontrado")
    return reg


def _backend(nome, cfg):
    if nome == "exato":
        return BackendExato()
    if nome == "mps":
        return BackendMPS()
    return escolher_backend(cfg)[0]


# ── execução (compartilhada entre síncrono e assíncrono) ────────────────────

def _treinar(pedido):
    cfg = pedido.circuito.construir()
    backend = _backend(pedido.backend, cfg)
    X, y = np.asarray(pedido.X, float), np.asarray(pedido.y, float)
    if pedido.tipo == "vqc":
        modelo = VQCClassifier(cfg, lr=pedido.lr, epocas=pedido.epocas, camada_pos=pedido.camada_pos,
                               seed=pedido.seed, backend=backend, otimizador=pedido.otimizador,
                               n_camadas=pedido.camadas_dados, codificacao=pedido.codificacao,
                               inicializacao=pedido.inicializacao,
                               treino_por_camadas=pedido.treino_por_camadas,
                               rotacoes=tuple(pedido.rotacoes), emaranhador=pedido.emaranhador,
                               leitura=pedido.leitura).fit(X, y)
    else:
        modelo = ClassificadorQURIOS([Brutos(), RFF(escala=pedido.escala_rff, seed=pedido.seed),
                                      Quantico(cfg, backend, seed=pedido.seed)], seed=pedido.seed).fit(X, y)
    reg = {"tipo": pedido.tipo, "modelo": modelo, "circuito": pedido.circuito.model_dump(),
           "backend": backend.nome, "n_features": X.shape[1], "n_treino": len(X),
           "acuracia_treino": float(np.mean(modelo.predict(X) == y)),
           "otimizador": pedido.otimizador if pedido.tipo == "vqc" else None,
           "codificacao": pedido.codificacao if pedido.tipo == "vqc" else "angulo"}
    mid = _registrar(reg)
    return _resumo(mid, reg)


def _orquestrar(pedido):
    cfg = pedido.circuito.construir()
    backend = _backend(pedido.backend, cfg)
    X, y = np.asarray(pedido.X, float), np.asarray(pedido.y, float)
    reg_c = _registro_campeoes() if pedido.chave_campeao else None
    orq = Orquestrador(candidatos_padrao(cfg, backend, incluir_vqc=pedido.incluir_vqc, seed=pedido.seed,
                                         incluir_kernel=pedido.incluir_kernel, registro=reg_c,
                                         chave=pedido.chave_campeao),
                       politica=pedido.politica, alfa=pedido.alfa, repeticoes=pedido.repeticoes,
                       seed=pedido.seed).fit(X, y)
    reg = {"tipo": "orquestrador", "modelo": orq, "circuito": pedido.circuito.model_dump(),
           "backend": backend.nome, "n_features": X.shape[1], "n_treino": len(X),
           "acuracia_treino": float(np.mean(orq.predict(X) == y)), "codificacao": "angulo"}
    mid = _registrar(reg)
    return _resumo(mid, reg)


def _evoluir(pedido):
    from .evolucao import BuscaEvolutiva
    from .evolucao_quantica import EvolucaoQuantica
    X, y = np.asarray(pedido.X, float), np.asarray(pedido.y, float)
    motor = EvolucaoQuantica if pedido.motor == "quantico" else BuscaEvolutiva
    res = motor(pedido.circuito.n_qubits, populacao=pedido.populacao, geracoes=pedido.geracoes,
                modo=pedido.modo, seed=pedido.seed, registro=_registro_campeoes(),
                chave=pedido.chave).executar(X, y)
    m, p = res["melhor"], res["padrao"]
    extra = {}
    if pedido.motor == "quantico":
        extra = {"n_qubits_gerador": res["n_qubits_gerador"], "genomas_possiveis": res["genomas_possiveis"],
                 "entropia_inicial_bits": res["entropia_inicial_bits"],
                 "entropia_final_bits": res["entropia_final_bits"],
                 "ligacoes_aprendidas": [{"genes": list(l["genes"]), "informacao_mutua": l["informacao_mutua"]}
                                         for l in res["ligacoes_aprendidas"]]}
    return {"chave": pedido.chave, "motor": pedido.motor, "promovido": res["promovido"], "motivo": res["motivo"],
            **extra,
            "melhor": {"descricao": m["descricao"], "genoma": {k: list(v) if isinstance(v, tuple) else v
                                                               for k, v in m["genoma"].items()},
                       "acuracia_cv": m["acuracia"], "portas_2q": m["custo"]},
            "padrao_quirios": {"acuracia_cv": p["acuracia"], "portas_2q": p["custo"]},
            "genomas_avaliados": res["avaliados"], "historico": res["historico"], "tempo_s": res["tempo_s"]}


def _dados_quanticos(pedido):
    from .dados_quanticos import comparar, fases_ising
    saida = []
    for s in pedido.shots:
        d = fases_ising(pedido.n_spins, pedido.n_amostras, h_min=pedido.h_min, h_max=pedido.h_max,
                        shots=s, seed=pedido.seed)
        saida.append(comparar(d, pedido.n_treino, pedido.repeticoes, pedido.seed))
    return {"resultados": saida}


def _projetar(pedido):
    from .ansatz import Ansatz
    from .futuro import projetar_hardware
    cfg = pedido.circuito.construir()
    X, y = np.asarray(pedido.X, float), np.asarray(pedido.y, float)
    ans = Ansatz(cfg, n_features=X.shape[1])
    return {"linhas": projetar_hardware(X, y, ans, ans.params_iniciais(pedido.seed),
                                        geracoes=pedido.geracoes, shots=pedido.shots, seed=pedido.seed)}


def _executar(funcao, pedido, origem):
    if pedido.assincrono:
        t = _tarefas.adicionar(em_segundo_plano(funcao, pedido, origem=origem))
        return JSONResponse(status_code=202, content={"tarefa": t.resumo()})
    try:
        return funcao(pedido)
    except ValueError as e:
        raise HTTPException(422, str(e))


# ── rotas ───────────────────────────────────────────────────────────────────

@app.get("/saude")
def saude():
    r = {"status": "ok", "versao": __version__,
         "protecao": {"chaves": _modo_chaves(), "auditoria": bool(os.environ.get("QURIOS_AUDITORIA")),
                      "cofre_pos_quantico": _cofre() is not None, "assinatura_pos_quantica": bool(
                          os.environ.get("QURIOS_CHAVE_PQC") or os.environ.get("QURIOS_CHAVE_PQC_PUBLICA"))}}
    if _rejeitados:
        r["modelos_rejeitados"] = sorted(_rejeitados)     # só os ids: o motivo fica no log do servidor
    return r


@app.get("/backends", dependencies=leitura)
def backends():
    """Backends que esta instância consegue executar."""
    out = [dict(BackendExato().capacidades(), max_qubits_api=MAX_QUBITS["exato"])]
    try:
        import qiskit_aer  # noqa: F401
        out.append(dict(BackendMPS().capacidades(), max_qubits_api=MAX_QUBITS["mps"]))
    except ImportError:
        pass
    return {"backends": out, "dispositivos": list(dispositivos_disponiveis())}


@app.post("/modelos", status_code=201, dependencies=treino)
def treinar(pedido: PedidoTreino):
    return _executar(_treinar, pedido, "treino")


@app.post("/orquestrar", status_code=201, dependencies=treino)
def orquestrar(pedido: PedidoOrquestrar):
    """Compara clássico e quântico na validação cruzada e registra o escolhido como modelo."""
    return _executar(_orquestrar, pedido, "orquestrador")


@app.post("/evoluir", status_code=201, dependencies=treino)
def evoluir(pedido: PedidoEvoluir):
    """Auto-evolução: busca um circuito melhor e só promove o campeão do problema
    com evidência estatística. Por padrão roda como tarefa (202 + id)."""
    return _executar(_evoluir, pedido, "evolucao")


@app.get("/campeoes", dependencies=leitura)
def campeoes():
    """Campeões registrados pela auto-evolução, por problema."""
    reg = _registro_campeoes()
    out = {}
    for ch in reg.chaves():
        c = reg.campeao(ch)
        if c:
            out[ch] = {k: c.get(k) for k in ("descricao", "genoma", "acuracia", "custo", "data", "motivo",
                                             "motor", "versao_quirios")}
            out[ch]["tentativas"] = len(reg.historico(ch))
    return out


@app.post("/dados-quanticos", status_code=201, dependencies=treino)
def dados_quanticos(pedido: PedidoDadosQuanticos):
    """Benchmark com dados quânticos (fases de Ising): clássicos veem medidas com
    poucos shots, quânticos veem o estado. Por padrão roda como tarefa (202 + id)."""
    return _executar(_dados_quanticos, pedido, "dados-quanticos")


@app.post("/projetar", dependencies=treino)
def projetar(pedido: PedidoProjetar):
    """Desempenho do circuito em gerações de hardware (hoje -> tolerante a falhas)."""
    return _executar(_projetar, pedido, "projecao")


from .painel import HTML as _HTML_PAINEL, GerenciadorSessoes  # noqa: E402

_painel = GerenciadorSessoes()


@app.get("/painel", response_class=HTMLResponse)
def painel():
    """Página do painel ao vivo (a página em si não contém dados; as rotas de dados exigem a chave)."""
    return _HTML_PAINEL


@app.post("/painel/sessoes", status_code=201, dependencies=treino)
def painel_iniciar(pedido: PedidoPainel):
    """Inicia uma evolução quântica ao vivo; os eventos saem em /painel/sessoes/{id}/eventos."""
    try:
        s = _painel.criar(**pedido.model_dump())
    except RuntimeError as e:
        raise HTTPException(429, str(e))
    return {"id": s.id}


@app.get("/painel/sessoes/{sid}/eventos", dependencies=leitura)
def painel_eventos(sid: str):
    s = _painel.obter(sid)
    if s is None:
        raise HTTPException(404, f"sessão '{sid}' não encontrada")
    return StreamingResponse(s.fluxo(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/painel/sessoes/{sid}", dependencies=leitura)
def painel_estado(sid: str):
    s = _painel.obter(sid)
    if s is None:
        raise HTTPException(404, f"sessão '{sid}' não encontrada")
    return {"id": s.id, "config": s.config, "terminou": s.terminou, "eventos": list(s.eventos)}


@app.get("/tarefas", dependencies=leitura)
def listar_tarefas():
    return [t.resumo() for t in _tarefas.listar()]


@app.get("/tarefas/{tid}", dependencies=leitura)
def consultar_tarefa(tid: str):
    t = _tarefas.obter(tid)
    if t is None:
        raise HTTPException(404, f"tarefa '{tid}' não encontrada")
    r = t.resumo()
    if r["status"] == "concluida":
        r["resultado"] = t.resultado()
    return r


@app.get("/modelos", dependencies=leitura)
def listar():
    with _trava:
        itens = list(_modelos.items())
    return [_resumo(mid, reg) for mid, reg in itens]


@app.get("/modelos/{mid}", dependencies=leitura)
def detalhar(mid: str):
    return _resumo(mid, _buscar(mid))


@app.delete("/modelos/{mid}", status_code=204, dependencies=admin)
def remover(mid: str):
    with _trava:
        if _modelos.pop(mid, None) is None:
            raise HTTPException(404, f"modelo '{mid}' não encontrado")
    d = _dir_modelos()
    if d is not None:                     # texto aberto, cifrado, assinaturas e metadados: nada fica no disco
        for nome in (".json", ".json.sig", ".json.cofre", ".json.cofre.sig", ".meta.json"):
            (d / f"{mid}{nome}").unlink(missing_ok=True)


@app.post("/modelos/{mid}/prever", dependencies=leitura)
def prever(mid: str, pedido: PedidoPrever):
    reg = _buscar(mid)
    try:
        _checar_X(pedido.X, reg["n_features"], reg.get("codificacao", "angulo"), reg["circuito"]["n_qubits"])
    except ValueError as e:
        raise HTTPException(422, str(e))
    proba = reg["modelo"].predict_proba(np.asarray(pedido.X, float))
    return {"predicoes": (proba > 0.5).astype(int).tolist(),
            "probabilidades": [float(p) for p in proba]}


@app.post("/modelos/{mid}/entropia", dependencies=leitura)
def entropia(mid: str, pedido: PedidoEntropia):
    """Entrelaçamento do estado que o modelo VQC treinado produz para a amostra x."""
    reg = _buscar(mid)
    if reg["tipo"] != "vqc":
        raise HTTPException(422, "entropia só está disponível para modelos tipo=vqc")
    n = reg["circuito"]["n_qubits"]
    if n > MAX_QUBITS_VETORIAL:
        raise HTTPException(422, f"entropia exige o vetor de estado completo (até {MAX_QUBITS_VETORIAL} qubits)")
    sub = sorted(set(pedido.subsistema))
    if any(q < 0 or q >= n for q in sub):
        raise HTTPException(422, f"subsistema deve conter qubits entre 0 e {n - 1}")
    try:
        _checar_X([pedido.x], reg["n_features"], reg.get("codificacao", "angulo"), n)
    except ValueError as e:
        raise HTTPException(422, str(e))
    st = reg["modelo"].estado_quantico(pedido.x)
    return {"subsistema": sub,
            "entropia_emaranhamento": entropia_emaranhamento(st, n, sub),
            "entropia_medicao": entropia_medicao(st),
            "pureza": pureza(st)}
