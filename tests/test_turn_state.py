"""Tests del estado por turno, corriendo bajo asyncio.gather REAL.

El test que acompanaba al primer intento de idempotencia solo revisaba el texto
de un aviso, asi que no detecto que la guarda era inerte: un `ContextVar.set()`
dentro de una tool se pierde al terminar la task. Estos tests reproducen la
forma en que LangChain ejecuta las tools, que es donde el bug vivia.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "orchestrator_service"))

from contextvars import ContextVar  # noqa: E402

from turn_state import clear_once, get_once, mark_once, reset_turn  # noqa: E402

HANDOFF = "handoff"


def test_la_marca_sobrevive_a_gather():
    # El bug original: con .set() el padre veia None y la guarda no servia.
    async def tool():
        await asyncio.sleep(0)
        mark_once(HANDOFF, "ok")

    async def main():
        reset_turn()
        await asyncio.gather(tool())
        return get_once(HANDOFF)

    assert asyncio.run(main()) == "ok"


def test_un_contextvar_reasignado_NO_sobrevive():
    # Deja constancia de POR QUE hace falta el dict: esta es la forma que fallaba.
    cv = ContextVar("demo", default=None)

    async def tool():
        await asyncio.sleep(0)
        cv.set("ok")

    async def main():
        cv.set(None)
        await asyncio.gather(tool())
        return cv.get()

    assert asyncio.run(main()) is None


def test_dos_tools_en_paralelo_solo_una_ejecuta():
    # El caso real: el modelo emite dos tool_calls de derivhumano en el mismo paso.
    mails = []

    async def derivhumano(n):
        if get_once(HANDOFF):
            return "ya derivado"
        mark_once(HANDOFF, "en curso")   # ANTES del await: si no, pasan las dos
        await asyncio.sleep(0.01)        # el SMTP
        mails.append(n)
        return mark_once(HANDOFF, "exito")

    async def main():
        reset_turn()
        return await asyncio.gather(derivhumano(1), derivhumano(2))

    resultados = asyncio.run(main())
    assert len(mails) == 1, f"se mandaron {len(mails)} mails"
    assert "ya derivado" in resultados


def test_dos_pasos_sucesivos_solo_uno_ejecuta():
    # El otro caso real: el agente llama la tool en dos pasos del mismo turno.
    mails = []

    async def derivhumano():
        if get_once(HANDOFF):
            return "ya derivado"
        mark_once(HANDOFF, "en curso")
        await asyncio.sleep(0)
        mails.append(1)
        return mark_once(HANDOFF, "exito")

    async def main():
        reset_turn()
        await asyncio.gather(derivhumano())
        await asyncio.gather(derivhumano())

    asyncio.run(main())
    assert len(mails) == 1


def test_el_reintento_antifuga_no_duplica_el_mail():
    # El reintento del commit 1b09ced corre el agente ENTERO de nuevo, dentro del
    # mismo request. La marca tiene que seguir viva entre las dos corridas.
    mails = []

    async def derivhumano():
        if get_once(HANDOFF):
            return "ya derivado"
        mark_once(HANDOFF, "en curso")
        await asyncio.sleep(0)
        mails.append(1)
        return mark_once(HANDOFF, "exito")

    async def corrida_del_agente():
        await asyncio.gather(derivhumano())

    async def main():
        reset_turn()
        await corrida_del_agente()   # primera pasada: se fuga el JSON
        await corrida_del_agente()   # reintento
    asyncio.run(main())
    assert len(mails) == 1


def test_si_el_mail_falla_se_puede_reintentar():
    # Sin clear_once, el segundo intento devolveria "exito" sin mandar nada.
    intentos = []

    async def derivhumano(falla):
        if get_once(HANDOFF):
            return "ya derivado"
        mark_once(HANDOFF, "en curso")
        await asyncio.sleep(0)
        intentos.append(1)
        if falla:
            clear_once(HANDOFF)
            return "error de SMTP"
        return mark_once(HANDOFF, "exito")

    async def main():
        reset_turn()
        primero = await derivhumano(falla=True)
        segundo = await derivhumano(falla=False)
        return primero, segundo

    primero, segundo = asyncio.run(main())
    assert primero == "error de SMTP"
    assert segundo == "exito"
    assert len(intentos) == 2


def test_turnos_distintos_no_se_contaminan():
    async def main():
        reset_turn()
        mark_once(HANDOFF, "exito")
        primero = get_once(HANDOFF)
        reset_turn()
        return primero, get_once(HANDOFF)

    primero, segundo = asyncio.run(main())
    assert primero == "exito"
    assert segundo is None


def test_sin_reset_no_explota():
    # Peor que perder la idempotencia es tirar el turno entero.
    assert get_once("cualquiera") is None
    assert mark_once("cualquiera", "x") == "x"
    clear_once("cualquiera")


def test_si_revienta_antes_del_envio_la_marca_no_queda_pegada():
    # La marca se pone dentro del try. Si algo falla entre la marca y el envio,
    # el except la limpia y la derivacion se puede reintentar en el mismo turno.
    intentos = []

    async def derivhumano(revienta):
        if get_once(HANDOFF):
            return "ya derivado"
        try:
            mark_once(HANDOFF, "en curso")
            intentos.append(1)
            if revienta:
                raise RuntimeError("fallo armando el mail")
            await asyncio.sleep(0)
            return mark_once(HANDOFF, "exito")
        except Exception:
            clear_once(HANDOFF)
            return "error"

    async def main():
        reset_turn()
        return await derivhumano(True), await derivhumano(False)

    primero, segundo = asyncio.run(main())
    assert primero == "error"
    assert segundo == "exito", "la marca quedo pegada y bloqueo el reintento"
    assert len(intentos) == 2
