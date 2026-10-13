"""Estado efimero de UN turno, que sobrevive a las copias de contexto.

LangChain ejecuta las tools dentro de `asyncio.gather`, y cada task arranca con
una COPIA del contexto. Consecuencia: un `ContextVar.set()` hecho ADENTRO de una
tool se descarta cuando la task termina — no lo ve el padre ni la tool hermana.
Mutar un objeto compartido SI se ve, porque es el mismo objeto en todas las
copias.

Esa diferencia explica por que el registro de URLs del guardrail funcionaba
(muta un set con `.update()`) y la guarda de idempotencia de `derivhumano` no
(reasignaba el contextvar con `.set()`): la guarda era inerte y el bot podia
mandar dos mails por la misma clienta.

Por eso aca el contextvar guarda un dict que se crea UNA vez al empezar el turno
y despues solo se muta. Vive en su propio archivo, separado de main.py, para que
se pueda testear: main.py no es importable sin las dependencias del servicio.
"""
from contextvars import ContextVar

_turn: ContextVar = ContextVar("turn_state", default=None)


def reset_turn():
    """Arranca un turno nuevo. Llamar UNA vez por request, antes del agente."""
    estado = {}
    _turn.set(estado)
    return estado


def _estado():
    """El dict del turno. Si nadie llamo a reset_turn, devuelve uno descartable
    en vez de romper: peor que perder la idempotencia es tirar el turno."""
    estado = _turn.get()
    if estado is None:
        estado = {}
        _turn.set(estado)
    return estado


def get_once(clave):
    """Lo que dejo una ejecucion previa de esta accion en el turno, o None."""
    return _estado().get(clave)


def mark_once(clave, valor):
    """Marca la accion como hecha y devuelve el valor.

    Hay que llamarla ANTES del primer `await` de la accion: si no, dos tools en
    paralelo pasan las dos el chequeo, porque la segunda arranca mientras la
    primera esta suspendida.
    """
    _estado()[clave] = valor
    return valor


def clear_once(clave):
    """Borra la marca. Para cuando la accion fallo y tiene que poder reintentarse:
    sin esto, el segundo intento devolveria 'exito' sin haber hecho nada.
    """
    _estado().pop(clave, None)
