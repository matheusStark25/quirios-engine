"""
Tarefas assíncronas: enviar agora, consultar depois.

Toda execução pesada do QURIOS (features em backend, treino, orquestração)
pode virar uma Tarefa. Isso importa principalmente no hardware IBM, onde a
fila leva de minutos a horas: o programa envia o job, guarda o id e segue.

    tarefa = backend.submeter(X, ansatz, params)
    tarefa.status()        # "na_fila" | "executando" | "concluida" | "erro" | "cancelada"
    tarefa.resultado()     # bloqueia até terminar (timeout opcional)

Tarefas locais rodam num pool de threads (QURIOS_TRABALHADORES, padrão 4).
Tarefas Qiskit embrulham o job do próprio Qiskit/IBM Runtime; um job IBM
pode ser recuperado em outro processo com BackendQiskit.recuperar(job_id).
"""
import datetime
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

ESTADOS = ("na_fila", "executando", "concluida", "erro", "cancelada")
FINAIS = ("concluida", "erro", "cancelada")

_executor = None
_trava_executor = threading.Lock()


def blas_em_uma_thread():
    """O OpenBLAS do numpy 2.x no Windows derruba o processo (access violation)
    quando várias threads entram ao mesmo tempo nas rotinas multithread dele
    (eigh, vdot de vetores complexos grandes) -- achado pelo teste de estresse
    da API. Quem já paraleliza por tarefa usa o BLAS em 1 thread por processo."""
    os.environ["OPENBLAS_NUM_THREADS"] = "1"   # bibliotecas BLAS carregadas depois (o scipy importa a sua sob demanda)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:                        # pragma: no cover
        return False
    threadpool_limits(1, user_api="blas")
    return True


def executor():
    global _executor
    with _trava_executor:
        if _executor is None:
            blas_em_uma_thread()
            n = int(os.environ.get("QURIOS_TRABALHADORES", "4"))
            _executor = ThreadPoolExecutor(max_workers=max(1, n), thread_name_prefix="quirios")
        return _executor


def _agora():
    return datetime.datetime.now().isoformat(timespec="seconds")


class Tarefa:
    """Handle de uma execução assíncrona. Subclasses implementam _status,
    _obter e cancelar."""

    def __init__(self, origem, id=None):
        self.id = id or uuid.uuid4().hex[:12]
        self.origem = origem
        self.criada_em = _agora()

    def status(self):
        return self._status()

    def pronta(self):
        return self.status() in FINAIS

    def resultado(self, timeout=None):
        """Espera terminar e devolve o resultado. Relança o erro da execução,
        ou TimeoutError se passar de timeout segundos."""
        limite = None if timeout is None else time.monotonic() + timeout
        while not self.pronta():
            if limite is not None and time.monotonic() >= limite:
                raise TimeoutError(f"tarefa {self.id} não terminou em {timeout}s")
            time.sleep(0.05 if limite is None else min(0.05, max(0.0, limite - time.monotonic())))
        return self._obter()

    def erro(self):
        return None

    def resumo(self):
        return {"id": self.id, "origem": self.origem, "status": self.status(),
                "criada_em": self.criada_em, "erro": self.erro()}

    def __repr__(self):
        return f"Tarefa(id='{self.id}', origem='{self.origem}', status='{self.status()}')"


class TarefaLocal(Tarefa):
    """Função executada no pool de threads do QURIOS."""

    def __init__(self, origem, futuro, id=None):
        super().__init__(origem, id)
        self._futuro = futuro

    def _status(self):
        f = self._futuro
        if f.cancelled():
            return "cancelada"
        if f.done():
            return "erro" if f.exception() is not None else "concluida"
        return "executando" if f.running() else "na_fila"

    def _obter(self):
        return self._futuro.result()

    def erro(self):
        f = self._futuro
        if f.done() and not f.cancelled() and f.exception() is not None:
            e = f.exception()
            return f"{type(e).__name__}: {e}"
        return None

    def cancelar(self):
        """Só cancela o que ainda está na fila (threads em execução não param)."""
        return self._futuro.cancel()


def em_segundo_plano(funcao, *args, origem="local", **kwargs):
    """Executa funcao(*args, **kwargs) no pool e devolve a Tarefa."""
    return TarefaLocal(origem, executor().submit(funcao, *args, **kwargs))


_STATUS_QISKIT = {
    "INITIALIZING": "na_fila", "QUEUED": "na_fila", "VALIDATING": "na_fila",
    "RUNNING": "executando", "DONE": "concluida", "ERROR": "erro", "CANCELLED": "cancelada",
}


class TarefaQiskit(Tarefa):
    """Job de primitivo Qiskit (Aer ou IBM Runtime) + pós-processamento do resultado."""

    def __init__(self, origem, job, posprocessar):
        jid = job.job_id() if hasattr(job, "job_id") else None
        super().__init__(origem, id=jid)
        self.job = job
        self._posprocessar = posprocessar
        self._valor = None
        self._trava = threading.Lock()

    def _status(self):
        s = self.job.status()
        nome = getattr(s, "name", str(s)).upper()
        return _STATUS_QISKIT.get(nome, "executando")

    def _obter(self):
        with self._trava:
            if self._valor is None:
                self._valor = self._posprocessar(self.job.result())
            return self._valor

    def erro(self):
        if self.status() != "erro":
            return None
        msg = getattr(self.job, "error_message", None)
        return msg() if callable(msg) else "job terminou com erro"

    def cancelar(self):
        try:
            self.job.cancel()
            return True
        except Exception:
            return False


class Registro:
    """Tarefas por id, thread-safe (usado pela API)."""

    def __init__(self):
        self._tarefas = {}
        self._trava = threading.Lock()

    def adicionar(self, tarefa):
        with self._trava:
            self._tarefas[tarefa.id] = tarefa
        return tarefa

    def obter(self, id):
        with self._trava:
            return self._tarefas.get(id)

    def listar(self):
        with self._trava:
            return list(self._tarefas.values())

    def remover(self, id):
        with self._trava:
            return self._tarefas.pop(id, None)
