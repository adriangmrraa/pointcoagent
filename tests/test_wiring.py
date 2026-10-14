"""Verifica el CABLEADO en main.py, no solo los modulos que usa.

Motivo: `test_turn_state.py` prueba el mecanismo de idempotencia con un
derivhumano de mentira, asi que no detectaria que main.py dejara de cablearlo
-que es exactamente la forma del bug que sobrevivio mes y medio: el modulo
andaba, el uso no-.

Como main.py no es importable sin las dependencias del servicio, se parsea con
`ast` y se comparan posiciones en el archivo.
"""
import ast
import sys
from pathlib import Path

import pytest

MAIN = Path(__file__).resolve().parent.parent / "orchestrator_service" / "main.py"
ARBOL = ast.parse(MAIN.read_text(encoding="utf-8"))


def _funcion(nombre):
    for n in ast.walk(ARBOL):
        if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef)) and n.name == nombre:
            return n
    pytest.fail(f"no existe la funcion {nombre} en main.py")


def _lineas_llamando(nodo, funcion):
    return [n.lineno for n in ast.walk(nodo)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == funcion]


def _lineas_metodo(nodo, metodo):
    return [n.lineno for n in ast.walk(nodo)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == metodo]


def test_chat_resetea_el_turno_antes_de_correr_el_agente():
    chat = _funcion("chat_endpoint")
    resets = _lineas_llamando(chat, "reset_turn")
    invokes = _lineas_metodo(chat, "ainvoke")
    assert resets, "chat_endpoint no llama reset_turn: la idempotencia queda sin estado"
    assert invokes, "chat_endpoint no llama ainvoke"
    assert min(resets) < min(invokes), (
        f"reset_turn (linea {min(resets)}) tiene que ir ANTES del primer "
        f"ainvoke (linea {min(invokes)})")


def test_derivhumano_marca_antes_del_primer_await():
    deriv = _funcion("derivhumano")
    marcas = _lineas_llamando(deriv, "mark_once")
    awaits = [n.lineno for n in ast.walk(deriv) if isinstance(n, ast.Await)]
    assert marcas, "derivhumano no llama mark_once: la guarda no marca nada"
    assert awaits, "derivhumano no tiene awaits"
    assert min(marcas) < min(awaits), (
        f"mark_once (linea {min(marcas)}) tiene que ir antes del primer await "
        f"(linea {min(awaits)}): si no, dos tools en paralelo mandan dos mails")


def test_derivhumano_consulta_la_guarda_antes_de_marcar():
    deriv = _funcion("derivhumano")
    assert min(_lineas_llamando(deriv, "get_once")) < min(_lineas_llamando(deriv, "mark_once"))


def test_derivhumano_limpia_la_marca_solo_en_el_manejador_de_error():
    # No alcanza con que clear_once exista en algun lado: tiene que estar DENTRO
    # del except. Si alguien lo moviera al camino feliz, borraria la marca
    # despues de mandar el mail y un segundo llamado mandaria otro.
    deriv = _funcion("derivhumano")
    en_except = []
    for n in ast.walk(deriv):
        if isinstance(n, ast.ExceptHandler):
            en_except += _lineas_llamando(n, "clear_once")
    todos = _lineas_llamando(deriv, "clear_once")
    assert en_except, ("clear_once tiene que estar en el except: sin eso, un "
                       "fallo de SMTP deja la marca puesta y el equipo nunca se "
                       "entera de esa clienta")
    assert todos == en_except, (
        f"hay clear_once fuera del except (lineas {sorted(set(todos) - set(en_except))}): "
        "borrar la marca en el camino feliz habilita un segundo mail")


def test_el_contextvar_inerte_no_volvio():
    # La forma vieja (ContextVar reasignado dentro de la tool) era decorativa.
    assert "current_turn_handoff_result" not in MAIN.read_text(encoding="utf-8")


def test_el_prompt_no_tiene_llaves_sueltas():
    # sys_template es un f-string: una llave de mas no rompe la sintaxis pero
    # corrompe el prompt en silencio.
    nodo = next((n for n in ast.walk(ARBOL) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "sys_template" for t in n.targets)), None)
    assert nodo is not None, "no se encontro sys_template"
    ctx = {"SHIPPING_PARTNERS": "x", "current_time": "x", "customer_name": "x",
           "store_address": "x", "store_catalog": "x", "store_description": "x",
           "store_name": "x", "store_website": "x"}
    prompt = eval(compile(ast.Expression(nodo.value), "<s>", "eval"), {}, ctx)
    import re
    sueltas = set(re.findall(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}", prompt))
    # format_instructions es preexistente: SystemMessage no es un template, asi
    # que el .partial() nunca lo reemplaza. Documentado, no se toca.
    assert sueltas <= {"{format_instructions}"}, f"placeholders inesperados: {sueltas}"


def test_el_prompt_solo_rutea_bolsos_por_categoria():
    """Guarda contra reintroducir el ruteo por categoria sin evals.

    Se intento (d27a8fb, 19385f0) y se revirtio: medido con el modelo real, la
    instruccion enfatica de "Accesorios para el pie" arrastraba todo lo que
    suena a accesorio de puntas, y el bot dejaba de encontrar el Endurecedor
    SHOE GLUE (4/4) y la Bolsa de Malla MESH BAGS (4/4), que hoy SI encuentra.
    Dos intentos de acotarla arreglaron un caso y rompieron otro.

    Si alguien vuelve a instruirlo, que sea con un set de evals de ruteo
    adelante. Este test existe para que ese cambio sea deliberado y no se
    cuele: borralo en el mismo commit que suma las evals.
    """
    texto = MAIN.read_text(encoding="utf-8")
    assert 'search_by_store_category("Bolsos")' in texto, (
        "el ruteo de bolsos/mochilas es el unico probado en produccion")
    for sin_evals in ('search_by_store_category("Accesorios para el pie")',
                      'search_by_store_category("Punteras")'):
        assert sin_evals not in texto, (
            f"el prompt volvio a instruir {sin_evals} sin evals de ruteo: "
            "eso rompe endurecedor y bolsa de red, que hoy funcionan")
