"""¿El bot promete combinaciones de talle y color que no existen?

Caso real 2026-11-19: la Amanda SO DANCA tiene talle M en Negro y en Bordo,
pero NO en Rosa té. El bot veía la lista aplanada "P, Negro, Rosa té, M, Bordo,
G" y no tenía forma de distinguirlo: si una clienta pedía "M en rosa" podía
decir que sí. Prometer algo que no existe es peor que no encontrarlo.

Mide TRES variantes contra el modelo real, para saber qué arregla qué:

  A: prompt viejo + variantes aplanadas   (lo que hay en producción hoy)
  B: prompt viejo + variantes agrupadas   (solo el arreglo del dato)
  C: prompt nuevo + variantes agrupadas   (dato + instrucción)

Los casos de control importan tanto como los casos del bug: si el bot empieza a
negar cosas que SÍ tiene, el arreglo es peor que el problema.

Necesita OPENROUTER_API_KEY en el .env local. No escribe nada, no toca la base.

Uso:
    python evals/eval_combinaciones.py
    python evals/eval_combinaciones.py --corridas 3
"""
import argparse
import io
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "orchestrator_service"))
from catalog import available_variant_values, variant_availability_text  # noqa: E402

ENV = {}
for linea in io.open(RAIZ / ".env", encoding="utf-8", errors="ignore"):
    linea = linea.strip()
    if linea and not linea.startswith("#") and "=" in linea:
        k, _, v = linea.partition("=")
        ENV[k.strip()] = v.strip().strip('"').strip("'")

KEY = ENV.get("OPENROUTER_API_KEY")
if not KEY:
    sys.exit("Falta OPENROUTER_API_KEY en el .env local.")

MODELO = "openai/gpt-4.1-mini"
URL = "https://openrouter.ai/api/v1/chat/completions"

TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_specific_products",
        "description": "Busca productos por nombre o categoría.",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}},
                       "required": ["q"]},
    },
}]


# --- productos reales del catálogo (variantes tal como las trae Tienda Nube) --

def prod(nombre, attrs, combos, precio="73000", url="https://pointecoach.shop/productos/x"):
    return {
        "name": {"es": nombre},
        "attributes": [{"es": a} for a in attrs],
        "variants": [{"stock": s, "price": precio,
                      "values": [{"es": x} for x in vals]} for s, vals in combos],
        "canonical_url": url,
        "description": {"es": "Malla de danza."},
    }


AMANDA = prod("Amanda SO DANCA | Malla con encaje superpuesto en el busto",
              ["Talle", "Color"], [
                  (1, ["P", "Negro"]), (0, ["P", "Bordo"]), (1, ["P", "Rosa té"]),
                  (2, ["M", "Negro"]), (2, ["M", "Bordo"]), (0, ["M", "Rosa té"]),
                  (1, ["G", "Negro"]), (0, ["G", "Bordo"]), (0, ["G", "Rosa té"]),
              ])

POLLERIN = prod("Pollerín SO DANCA | Pollera de crepé para danza",
                ["Color", "Talle"], [
                    (1, ["Blanco", "P"]), (1, ["Negro", "P"]), (1, ["Rosa", "P"]),
                    (1, ["Lila", "P"]), (1, ["Negro", "M"]), (1, ["Rosa", "M"]),
                    (1, ["Negro", "G"]),
                ], precio="45000")

AF23 = prod("Zapatillas de Media punta AF23 ADULTO", ["Talle", "Color"], [
    (1, ["34", "Rosa"]), (1, ["35", "Rosa"]), (1, ["37", "Rosa"]),
    (1, ["38", "Rosa"]), (1, ["39", "Beige"]), (1, ["39", "Rosa"]),
    (1, ["39", "Blanco"]),
], precio="52000")

LIFEKNIT = prod("Medias LifeKnit Sox II", ["Color", "Talle"], [
    (1, ["Beige", "S"]), (1, ["Negro", "S"]), (1, ["Beige", "M"]),
    (1, ["Negro", "M"]), (1, ["Beige", "L"]), (1, ["Negro", "L"]),
], precio="38000")


def simplificado(p, agrupado):
    """Lo que ve el modelo. agrupado=False reproduce la lista aplanada de hoy."""
    v = p["variants"][0]
    return {
        "id": 1,
        "name": p["name"]["es"],
        "price": v["price"],
        "promotional_price": None,
        "description": p["description"]["es"],
        "variants": (variant_availability_text(p) if agrupado
                     else ", ".join(available_variant_values(p["variants"]))),
        "url": p["canonical_url"],
        "imageUrl": "https://img/x.jpg",
    }


# --- los casos ---------------------------------------------------------------
# "prohibido": si alguna de estas frases aparece afirmada, el bot prometió mal.
# "esperado": lo que SÍ tiene que nombrar.

CASOS = [
    {
        "id": "amanda-M-rosa (EL BUG)",
        "producto": AMANDA,
        "pregunta": "Hola! La malla Amanda la tenes en talle M en rosa?",
        "pide": ("M", "Rosa té"),
        "existe": False,
        "debe_nombrar": ["negro", "bordo"],
    },
    {
        "id": "pollerin-M-lila",
        "producto": POLLERIN,
        "pregunta": "tenes el pollerin lila en M?",
        "pide": ("M", "Lila"),
        "existe": False,
        "debe_nombrar": ["negro", "rosa"],
    },
    {
        "id": "af23-34-blanco",
        "producto": AF23,
        "pregunta": "las media punta AF23 las tenes en 34 blancas?",
        "pide": ("34", "Blanco"),
        "existe": False,
        "debe_nombrar": ["rosa"],
    },
    {
        "id": "CONTROL amanda-P-rosa (SI existe)",
        "producto": AMANDA,
        "pregunta": "la malla Amanda en talle P la tenes en rosa te?",
        "pide": ("P", "Rosa té"),
        "existe": True,
        "debe_nombrar": ["rosa"],
    },
    {
        "id": "CONTROL lifeknit-M-negro (SI existe)",
        "producto": LIFEKNIT,
        "pregunta": "las medias LifeKnit en M negras tenes?",
        "pide": ("M", "Negro"),
        "existe": True,
        "debe_nombrar": ["negro"],
    },
    {
        "id": "CONTROL amanda-M (sin color)",
        "producto": AMANDA,
        "pregunta": "tenes la malla Amanda en M?",
        "pide": ("M", "Negro"),
        "existe": True,
        "debe_nombrar": ["negro", "bordo"],
    },
]

# Las formas en que el modelo realmente niega. La primera version de este
# regex solo buscaba "no tengo/no tenemos" y marcaba como ERROR respuestas
# correctas como "...en negro y bordo, pero no en rosa". Medir mal es peor que
# no medir: parecia que el arreglo no funcionaba.
NEGACION = re.compile(
    r"(?i)("
    r"no (la |lo |me |nos |te )?(tengo|tenemos|queda|quedan|hay|est[aá]n?|disponible)"
    r"|pero no\b"
    r"|no (en|ese|esa|est[eé]|esos|esas|para ese)\b"
    r"|no (est[aá]|estan|están) disponible"
    r"|sin stock|se agot|no me queda"
    r"|solo (hay|tengo|tenemos|est[aá] disponible)\b"
    r")")


def plantilla(ruta_main):
    """Saca el sys_template del main.py indicado y lo rellena."""
    src = io.open(ruta_main, encoding="utf-8").read()
    cuerpo = re.search(r'sys_template = f"""(.*?)"""', src, re.S).group(1)
    # las expresiones anidadas {f"..." if x else ""} se resuelven a vacío
    cuerpo = re.sub(r'\{f"[^"]*"[^}]*\}', "", cuerpo)
    cuerpo = cuerpo.replace("{{", "\x01").replace("}}", "\x02")
    for k, v in {
        "store_name": "Pointe Coach", "store_description": "tienda de danza",
        "store_website": "https://pointecoach.shop", "store_address": "Paraná",
        "store_catalog": "", "current_time": "2026-11-19 17:00",
        "customer_name": "", "SHIPPING_PARTNERS": "Andreani, Correo Argentino",
    }.items():
        cuerpo = cuerpo.replace("{" + k + "}", v)
    return cuerpo.replace("\x01", "{").replace("\x02", "}")


def pedir(system, caso, agrupado):
    salida = simplificado(caso["producto"], agrupado)
    mensajes = [
        {"role": "system", "content": system},
        {"role": "user", "content": caso["pregunta"]},
        {"role": "assistant", "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "search_specific_products",
                         "arguments": json.dumps({"q": "Leotardo"})}}]},
        {"role": "tool", "tool_call_id": "call_1",
         "content": json.dumps([salida], ensure_ascii=False)},
    ]
    cuerpo = json.dumps({
        "model": MODELO, "messages": mensajes, "tools": TOOLS,
        "temperature": 0, "max_tokens": 2000,
        "response_format": {"type": "json_object"},
    }).encode()
    req = urllib.request.Request(URL, data=cuerpo, headers={
        "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            resp = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return "ERROR HTTP %s: %s" % (e.code, e.read().decode()[:150]), salida["variants"]
    msg = (resp.get("choices") or [{}])[0].get("message", {})
    bruto = msg.get("content") or ""
    try:
        d = json.loads(bruto)
        textos = [m.get("text", "") for m in d.get("messages", []) if isinstance(m, dict)]
        return " ".join(t for t in textos if t), salida["variants"]
    except Exception:
        return bruto, salida["variants"]


def _pelado(s):
    """Sin acentos y en minuscula: el modelo escribe "Bordo" y "Bordo"."""
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", s or "")
                   if unicodedata.category(c) != "Mn").lower()


def _frases(texto):
    cortes = re.compile(r"(?<=[.!?])\s+" r"|\n+" r"|\s+\|\s+")
    return [f for f in cortes.split(texto) if f.strip()]


# La ficha del producto NO es una promesa: "Variantes: Talle M: Negro, Bordo"
# es la lista honesta, aunque nombre el talle y el color que se pidieron. Lo que
# se juzga es lo que el bot AFIRMA con sus palabras.
_FICHA = re.compile(r"(?i)^\s*(variantes|precio)\s*:|^https?://|^\s*$")


def _prosa(texto):
    return [f for f in _frases(texto) if not _FICHA.search(f)]


def _nombra_talle(frase, talle):
    return re.search(r"(?<![a-z0-9])" + re.escape(_pelado(talle)) + r"(?![a-z0-9])",
                     _pelado(frase)) is not None


def juzgar(caso, texto):
    """Clasifica la respuesta en una de tres cosas, no en bien/mal.

    Perseguir un "bien o mal" llevo a tres versiones de este juez que median
    cualquier cosa: marcaba como error respuestas correctas porque el bot
    escribia "Bordo" con acento, o porque negaba algo de OTRO talle, o porque
    decia "no en rosa" en vez de "no en rosa te". Lo que de verdad hay que
    separar es:

      PROMETE  -> afirma la combinacion pedida y no existe. Esto cuesta plata
                  y enoja a la clienta: es el bug.
      CONTESTA -> responde la pregunta con la verdad.
      ESQUIVA  -> no promete nada falso, pero tampoco contesta: pega la ficha y
                  se va. No es peligroso, es mala atencion.
    """
    if texto.startswith("ERROR HTTP"):
        return None, texto[:60]
    talle, color = caso["pide"]
    # El bot abrevia: "Rosa te" lo escribe "rosa". Se busca la primera palabra.
    raiz = _pelado(color).split()[0]
    relevantes = [f for f in _prosa(texto) if raiz in _pelado(f)]

    afirma = [f for f in relevantes
              if not NEGACION.search(f) and _nombra_talle(f, talle)]
    niega = [f for f in relevantes if NEGACION.search(f)]

    if not caso["existe"]:
        if afirma:
            return "PROMETE", afirma[0][:95]
        if not niega:
            return "ESQUIVA", "no dijo que esa combinación no está"
        faltan = [w for w in caso["debe_nombrar"]
                  if _pelado(w).split()[0] not in _pelado(texto)]
        if faltan:
            return "CONTESTA", "sin ofrecer alternativa (%s)" % ", ".join(faltan)
        return "CONTESTA", "y ofreció alternativa"

    # Casos de control: tiene que poder decir que SI.
    malas = [f for f in niega if _nombra_talle(f, talle)]
    if malas:
        return "PROMETE", "negó lo que SÍ tiene: " + malas[0][:70]
    if not relevantes and _pelado(color) not in _pelado(texto):
        return "ESQUIVA", "no nombró el color pedido"
    return "CONTESTA", "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corridas", type=int, default=2)
    args = ap.parse_args()

    import subprocess
    cache = Path(__file__).resolve().parent / "data"
    cache.mkdir(exist_ok=True)
    viejo = cache / "_main_head.py"
    io.open(viejo, "w", encoding="utf-8").write(
        subprocess.run(["git", "show", "HEAD:orchestrator_service/main.py"],
                       cwd=RAIZ, capture_output=True, text=True,
                       encoding="utf-8").stdout)

    prompt_viejo = plantilla(viejo)
    prompt_nuevo = plantilla(RAIZ / "orchestrator_service" / "main.py")
    print("prompt viejo %d chars · prompt nuevo %d chars\n" % (
        len(prompt_viejo), len(prompt_nuevo)))

    arms = [("A viejo+aplanado", prompt_viejo, False),
            ("B viejo+agrupado", prompt_viejo, True),
            ("C nuevo+agrupado", prompt_nuevo, True)]

    categorias = ("PROMETE", "CONTESTA", "ESQUIVA")
    puntaje = {a[0]: {c: 0 for c in categorias} for a in arms}
    for caso in CASOS:
        print("=" * 78)
        print(caso["id"] + "   ->   " + caso["pregunta"])
        for etiqueta, system, agrupado in arms:
            for n in range(args.corridas):
                texto, vistas = pedir(system, caso, agrupado)
                cat, motivo = juzgar(caso, texto)
                if n == 0:
                    print("  [%s]  ve: %s" % (etiqueta, vistas))
                if cat in puntaje[etiqueta]:
                    puntaje[etiqueta][cat] += 1
                plano = " ".join(texto.split())
                print("      %-9s %-40s %s" % (cat, motivo, plano[:170]))
        print()

    print("=" * 78)
    print("%-20s %-9s %-9s %-9s" % ("", "PROMETE", "CONTESTA", "ESQUIVA"))
    print("%-20s %s" % ("", "(el bug)"))
    for etiqueta, _, _ in arms:
        p = puntaje[etiqueta]
        print("  %-18s %-9d %-9d %-9d" % (
            etiqueta, p["PROMETE"], p["CONTESTA"], p["ESQUIVA"]))


if __name__ == "__main__":
    main()
