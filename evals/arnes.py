"""Arnés compartido de los bancos de prueba: habla con el modelo real.

Los bancos (`banco_*.py`, `eval_combinaciones.py`) miden cosas distintas pero
todos necesitan lo mismo: el system prompt de verdad, las herramientas de
verdad, y poder comparar la versión que está en producción contra la que
estamos por subir.

Dos decisiones que importan:

1. **Las herramientas se leen del `main.py` por AST**, no se copian a mano. Si
   alguien cambia un docstring de una tool, los bancos lo miden al toque. Un
   banco con herramientas inventadas mide otra cosa y da falsa tranquilidad.

2. **El prompt también se saca del `main.py`**, de los dos: el de `HEAD` (lo
   que está corriendo) y el del working tree (lo que vamos a subir). Sin el
   "antes" no hay forma de saber si un cambio mejora o empeora.

No escribe nada fuera de `evals/data/` (ignorado por git). Necesita
OPENROUTER_API_KEY en el .env local; nunca la imprime.
"""
import ast
import io
import json
import re
import subprocess
import sys
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "orchestrator_service"))

ENV = {}
for _l in io.open(RAIZ / ".env", encoding="utf-8", errors="ignore"):
    _l = _l.strip()
    if _l and not _l.startswith("#") and "=" in _l:
        _k, _, _v = _l.partition("=")
        ENV[_k.strip()] = _v.strip().strip('"').strip("'")

KEY = ENV.get("OPENROUTER_API_KEY")
MODELO = ENV.get("EVAL_MODEL", "openai/gpt-4.1-mini")
URL = "https://openrouter.ai/api/v1/chat/completions"
DATOS = Path(__file__).resolve().parent / "data"


def exigir_key():
    if not KEY:
        sys.exit("Falta OPENROUTER_API_KEY en el .env local.")


# --- el main.py de produccion y el del working tree -------------------------

def main_de_head():
    """Copia del main.py que está corriendo, para comparar contra."""
    DATOS.mkdir(exist_ok=True)
    destino = DATOS / "_main_head.py"
    r = subprocess.run(["git", "show", "HEAD:orchestrator_service/main.py"],
                       cwd=RAIZ, capture_output=True, text=True, encoding="utf-8")
    if not r.stdout:
        sys.exit("No pude leer el main.py de HEAD: " + (r.stderr or "")[:200])
    io.open(destino, "w", encoding="utf-8").write(r.stdout)
    return destino


MAIN_ACTUAL = RAIZ / "orchestrator_service" / "main.py"

# Valores de relleno del f-string. Son los del tenant real salvo las claves.
RELLENO = {
    "store_name": "Pointe Coach",
    "store_description": "tienda de artículos de danza clásica y contemporánea",
    "store_website": "https://pointecoach.shop",
    "store_address": "Paraná, Entre Ríos",
    "store_catalog": "",
    "current_time": "2026-11-19 17:00 (Argentina)",
    "customer_name": "",
    "SHIPPING_PARTNERS": "Andreani, Correo Argentino, Vía Cargo",
}


def plantilla(ruta_main=None):
    """El system prompt renderizado, sacado del f-string `sys_template`."""
    ruta_main = ruta_main or MAIN_ACTUAL
    src = io.open(ruta_main, encoding="utf-8").read()
    m = re.search(r'sys_template = f"""(.*?)"""', src, re.S)
    if not m:
        sys.exit("No encontré sys_template en " + str(ruta_main))
    cuerpo = m.group(1)
    # las expresiones anidadas {f"..." if x else ""} se resuelven a vacío
    cuerpo = re.sub(r'\{f"[^"]*"[^}]*\}', "", cuerpo)
    cuerpo = cuerpo.replace("{{", "\x01").replace("}}", "\x02")
    for k, v in RELLENO.items():
        cuerpo = cuerpo.replace("{" + k + "}", v)
    sobran = re.findall(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}", cuerpo)
    if sobran:
        sys.exit("El prompt espera variables que no sé rellenar: %s" % set(sobran))
    return cuerpo.replace("\x01", "{").replace("\x02", "}")


# --- las herramientas, leidas del codigo ------------------------------------

def herramientas(ruta_main=None):
    """Las tools del agente tal como el modelo las ve, sacadas del @tool."""
    ruta_main = ruta_main or MAIN_ACTUAL
    arbol = ast.parse(io.open(ruta_main, encoding="utf-8").read())
    fuera = []
    for n in ast.walk(arbol):
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not any(getattr(d, "id", "") == "tool" for d in n.decorator_list):
            continue
        props = {a.arg: {"type": "string"} for a in n.args.args}
        fuera.append({"type": "function", "function": {
            "name": n.name,
            "description": (ast.get_docstring(n) or n.name).strip(),
            "parameters": {"type": "object", "properties": props,
                           "required": list(props)},
        }})
    if not fuera:
        sys.exit("No encontré herramientas @tool en " + str(ruta_main))
    return fuera


# --- hablar con el modelo ---------------------------------------------------

def _post(payload):
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + KEY,
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return {"_error": "HTTP %s: %s" % (e.code, e.read().decode()[:200])}
    except Exception as e:                                  # red, timeout
        return {"_error": type(e).__name__ + ": " + str(e)[:150]}


def que_tool_llama(system, pregunta, tools=None, historial=None):
    """Qué herramienta elige el modelo. Devuelve (nombre, argumentos) o (None, texto).

    Manda `response_format: json_object` porque producción lo manda
    (`model_kwargs` en main.py). Sin eso el banco mide un modelo que no es el
    que atiende a las clientas.
    """
    msgs = [{"role": "system", "content": system}]
    msgs += historial or []
    msgs.append({"role": "user", "content": pregunta})
    resp = _post({"model": MODELO, "messages": msgs,
                  "tools": tools or herramientas(), "temperature": 0,
                  "max_tokens": 500,
                  "response_format": {"type": "json_object"}})
    if "_error" in resp:
        return None, resp["_error"]
    msg = (resp.get("choices") or [{}])[0].get("message", {})
    llamadas = msg.get("tool_calls") or []
    if not llamadas:
        return None, (msg.get("content") or "")[:300]
    nombres = []
    for c in llamadas:
        f = c.get("function", {})
        try:
            args = json.loads(f.get("arguments") or "{}")
        except Exception:
            args = {"_crudo": f.get("arguments")}
        nombres.append((f.get("name"), args))
    return nombres[0][0], nombres[0][1] if len(nombres) == 1 else nombres


def responder(system, pregunta, salida_tool, nombre_tool="search_specific_products",
              args_tool=None, tools=None, cola=None):
    """Qué le contesta a la clienta si la tool devolvió `salida_tool`.

    Se le arma el turno ya con la herramienta ejecutada, porque lo que se quiere
    medir es la REDACCIÓN, no el ruteo (eso lo mide banco_ruteo.py).
    """
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": pregunta},
        {"role": "assistant", "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": nombre_tool,
                         "arguments": json.dumps(args_tool or {"q": "Leotardo"})}}]},
        {"role": "tool", "tool_call_id": "call_1",
         "content": json.dumps(salida_tool, ensure_ascii=False)},
    ]
    # `cola` sirve para reproducir el reintento de un guardia: la respuesta que
    # salió mal, más el aviso del sistema.
    msgs += cola or []
    resp = _post({"model": MODELO, "messages": msgs, "tools": tools or herramientas(),
                  "temperature": 0, "max_tokens": 2000,
                  "response_format": {"type": "json_object"}})
    if "_error" in resp:
        return "ERROR " + resp["_error"]
    bruto = ((resp.get("choices") or [{}])[0].get("message", {}).get("content") or "")
    try:
        d = json.loads(bruto)
        textos = [m.get("text", "") for m in d.get("messages", [])
                  if isinstance(m, dict)]
        unido = " ".join(t for t in textos if t)
        return unido or bruto
    except Exception:
        return bruto


# --- lo que ve el modelo de un producto -------------------------------------

def simplificado(p, agrupado=True):
    """Replica simplify_product() de main.py. agrupado=False = la version vieja."""
    from catalog import (available_variant_values, variant_availability_text,
                         SIN_VARIANTES)
    vs = p.get("variants") or []
    v0 = vs[0] if vs else {}
    detalle = (variant_availability_text(p) if agrupado
               else ", ".join(available_variant_values(vs)))
    salida = {
        "id": p.get("id", 1),
        "name": p.get("name", {}).get("es", "Sin nombre"),
        "price": v0.get("price", "0"),
        "promotional_price": v0.get("promotional_price"),
        "description": (p.get("description") or {}).get("es", ""),
        "url": p.get("canonical_url", "https://pointecoach.shop/productos/x"),
        "imageUrl": "https://pointecoach.shop/img/x.jpg",
    }
    # igual que simplify_product: si no hay opciones, un valor explícito
    salida["variants"] = detalle or SIN_VARIANTES
    return salida


def producto(nombre, attrs, combos, precio="73000", desc="", url=None, pid=1):
    """Un producto con la forma que devuelve Tienda Nube.

    combos: lista de (stock, [valor por cada attr]).
    """
    return {
        "id": pid,
        "name": {"es": nombre},
        "attributes": [{"es": a} for a in attrs],
        "variants": [{"stock": s, "price": precio,
                      "values": [{"es": x} for x in vals]} for s, vals in combos],
        "description": {"es": desc},
        "canonical_url": url or "https://pointecoach.shop/productos/x",
    }


# --- herramientas para juzgar texto ----------------------------------------

def pelado(s):
    """Sin acentos y en minúscula: el modelo escribe "Bordó" y "Bordo"."""
    return "".join(c for c in unicodedata.normalize("NFD", s or "")
                   if unicodedata.category(c) != "Mn").lower()


_CORTES = re.compile(r"(?<=[.!?])\s+|\n+|\s+\|\s+")
# La ficha del producto no es una afirmación del bot: "Variantes: Talle M:
# Negro" es la lista honesta, aunque nombre el talle y el color pedidos.
_FICHA = re.compile(r"(?i)^\s*(variantes|precio)\s*:|^https?://|^\s*$")


def frases(texto):
    return [f for f in _CORTES.split(texto or "") if f.strip()]


def prosa(texto):
    """Solo lo que el bot afirma con sus palabras, sin la ficha del producto."""
    return [f for f in frases(texto) if not _FICHA.search(f)]


NEGACION = re.compile(
    r"(?i)("
    r"no (la |lo |me |nos |te )?(tengo|tenemos|queda|quedan|hay|est[aá]n?|disponible"
    r"|viene[n]?|son|es|manejamos|trabajamos|por talle)"
    r"|pero no\b"
    r"|no (en|ese|esa|est[eé]|esos|esas|para ese)\b"
    r"|no (est[aá]|estan|están) disponible"
    r"|sin stock|se agot|no me queda"
    r"|solo (hay|tengo|tenemos|est[aá] disponible)\b"
    r")")


def comparar(nombre_banco, casos, corridas, medir, columnas):
    """Corre `medir(caso, brazo)` y arma la tabla. medir devuelve (categoria, detalle).

    brazos: lista de (etiqueta, system). `columnas` son las categorías a contar,
    la PRIMERA es la que no debería pasar nunca.
    """
    print("=" * 78)
    print(nombre_banco)
    print("=" * 78)
    puntaje = {}
    detalle = {}
    for caso in casos:
        print()
        print(caso["id"] + "   ->   " + caso["pregunta"])
        for etiqueta, system in caso["_brazos"]:
            puntaje.setdefault(etiqueta, {c: 0 for c in columnas})
            marcas = []
            for _ in range(corridas):
                cat, texto = medir(caso, system)
                if cat in puntaje[etiqueta]:
                    puntaje[etiqueta][cat] += 1
                marcas.append(cat)
                print("  %-18s %-9s %s" % (etiqueta, cat, " ".join(texto.split())[:150]))
            detalle[(caso["id"], etiqueta)] = marcas
    print()
    print("=" * 78)
    print("%-20s %s" % ("", "  ".join("%-9s" % c for c in columnas)))
    for etiqueta in puntaje:
        print("  %-18s %s" % (etiqueta, "  ".join(
            "%-9d" % puntaje[etiqueta][c] for c in columnas)))
    return puntaje, detalle
