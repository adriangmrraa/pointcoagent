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


def test_derivhumano_limpia_la_marca_si_falla():
    deriv = _funcion("derivhumano")
    assert _lineas_llamando(deriv, "clear_once"), (
        "sin clear_once, un fallo de SMTP deja la marca puesta y el equipo "
        "nunca se entera de esa clienta")


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


def test_las_tres_categorias_obligatorias_estan_en_el_prompt():
    # Punteras y Accesorios para el pie no comparten NI UN producto: si el
    # prompt deja de nombrarlas, las consultas de punteras se van a la
    # categoria equivocada (medido con el modelo real: 3 de 6 corridas).
    texto = MAIN.read_text(encoding="utf-8")
    for categoria in ('search_by_store_category("Bolsos")',
                      'search_by_store_category("Accesorios para el pie")',
                      'search_by_store_category("Punteras")'):
        assert categoria in texto, f"el prompt ya no instruye {categoria}"
