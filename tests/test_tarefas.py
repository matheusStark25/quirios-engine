import threading

import pytest

from quirios_engine.tarefas import Registro, em_segundo_plano


def test_tarefa_concluida():
    t = em_segundo_plano(lambda a, b: a + b, 2, 3, origem="soma")
    assert t.resultado(timeout=10) == 5
    assert t.status() == "concluida" and t.pronta() and t.erro() is None
    assert t.resumo()["origem"] == "soma"


def test_tarefa_com_erro_relanca():
    def falha():
        raise ValueError("deu ruim")
    t = em_segundo_plano(falha)
    with pytest.raises(ValueError, match="deu ruim"):
        t.resultado(timeout=10)
    assert t.status() == "erro"
    assert "deu ruim" in t.erro()


def test_timeout():
    liberar = threading.Event()
    t = em_segundo_plano(liberar.wait)
    with pytest.raises(TimeoutError):
        t.resultado(timeout=0.2)
    assert t.status() in ("executando", "na_fila")
    liberar.set()
    assert t.resultado(timeout=10) is True


def test_registro():
    reg = Registro()
    t = reg.adicionar(em_segundo_plano(lambda: 1))
    assert reg.obter(t.id) is t and t in reg.listar()
    assert reg.remover(t.id) is t and reg.obter(t.id) is None
