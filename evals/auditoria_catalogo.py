"""Audita el catálogo: ¿el bot encuentra cada producto como lo pide una clienta?

El problema que motiva esto (caso real 2026-11-19): una clienta pidió "el
leotardo Amanda" y el bot dijo que no lo tenía. El producto se llama "Amanda SO
DANCA | Malla con encaje superpuesto en el busto" — sin la palabra "leotardo".
Tienda Nube exige que TODOS los términos estén en el nombre, así que la búsqueda
dio cero.

Este script recorre el catálogo completo y, para cada producto, prueba cómo lo
pediría una clienta de verdad:

  - por su nombre de modelo solo ("Amanda")
  - por categoría + modelo ("leotardo Amanda", "malla Amanda")
  - por categoría sola ("leotardo")

Y marca los que el bot NO encontraría. Es de solo lectura.

Uso:
    python evals/auditoria_catalogo.py
    python evals/auditoria_catalogo.py --detalle
"""
import argparse
import io
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "orchestrator_service"))
from catalog import (available_variant_values, is_product_available,  # noqa: E402
                     query_terms, rank_widened_results, sin_acentos)

# Cómo nombra la clienta cada familia. Se usa para armar las consultas de prueba.
CATEGORIA_A_PALABRA = {
    "leotardos": ["leotardo", "malla", "body"],
    "zapatillas de punta": ["puntas", "zapatillas de punta"],
    "zapatillas de media punta": ["media punta", "zapatillas de media punta"],
    "medias": ["medias", "cancan"],
    "bolsos": ["bolso", "mochila"],
    "punteras": ["punteras"],
    "metatarsianas": ["metatarsianas"],
    "indumentaria": ["pollera", "pollerin", "malla", "leotardo"],
    "nuevos ingresos": [],
}
# Palabras del nombre que NO identifican el modelo.
GENERICAS = {
    "so", "danca", "danza", "de", "del", "la", "el", "los", "las", "para", "con",
    "sin", "y", "o", "un", "una", "en", "al", "capezio", "grishko", "sansha",
    "bunheads", "pointe", "coach", "americano", "americana", "ballet", "clasica",
    "clasico", "zapatillas", "zapatilla", "media", "medias", "punta", "puntas",
    "leotardo", "malla", "maillot", "bolso", "bolsa", "mochila", "mochilas",
    "puntera", "punteras", "metatarsiana", "metatarsianas", "cintas", "cinta",
    "elasticos", "elastico", "protector", "protectores", "separador",
    "separadores", "pollera", "pollerin", "falda", "infantil", "adulto", "tela",
    "silicona", "gel", "nude", "negro", "rosa", "blanco", "beige", "ii", "iii",
}


def cargar_env():
    env = {}
    for linea in io.open(RAIZ / ".env", encoding="utf-8", errors="replace"):
        linea = linea.strip()
        if "=" in linea and not linea.startswith("#"):
            k, v = linea.split("=", 1)
            env[k.strip()] = v.strip()
    return env


ENV = cargar_env()
SID, TOK = ENV["TIENDANUBE_STORE_ID"], ENV["TIENDANUBE_ACCESS_TOKEN"]
H = {"Authentication": f"bearer {TOK}", "User-Agent": "auditoria-catalogo/1.0"}
_cache = {}


def api(path, params):
    clave = path + json.dumps(params, sort_keys=True)
    if clave in _cache:
        return _cache[clave]
    url = f"https://api.tiendanube.com/v1/{SID}{path}?" + urllib.parse.urlencode(params, encoding="utf-8")
    with urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=40) as r:
        _cache[clave] = json.loads(r.read().decode("utf-8"))
    return _cache[clave]


def buscar(q):
    """Los ids que el bot veria con esa consulta (exacta + ampliada)."""
    crudo = api("/products", {"q": q, "per_page": 20})
    vivos = [p for p in crudo if is_product_available(p)]
    if vivos:
        return {p["id"] for p in vivos}
    terms = query_terms(q)[:3]
    if len(terms) < 2:
        return set()
    parciales = []
    for t in terms:
        parciales.append([{"id": p["id"], "name": (p.get("name", {}).get("es") or "")}
                          for p in api("/products", {"q": t, "per_page": 30})
                          if is_product_available(p)])
    parciales.sort(key=len)          # selectividad, igual que en produccion
    vistos, union = set(), []
    for pr in parciales:
        for p in pr:
            if p["id"] not in vistos:
                vistos.add(p["id"]); union.append(p)
    return {p["id"] for p in rank_widened_results(q, union)}


def modelo_de(nombre):
    """Las palabras del nombre que identifican el modelo (Amanda, Hanami, SD34...)."""
    limpio = re.split(r"[|/,]", nombre)[0]
    fuera = []
    for palabra in re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÑñ0-9]+", limpio):
        if sin_acentos(palabra) in GENERICAS or len(palabra) < 3:
            continue
        fuera.append(palabra)
    return fuera


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detalle", action="store_true")
    args = ap.parse_args()

    cats = {c["id"]: ((c.get("name") or {}).get("es") or "?") for c in api("/categories", {"per_page": 100})}

    print("Trayendo el catálogo completo...")
    productos, pagina = [], 1
    while True:
        lote = api("/products", {"per_page": 200, "page": pagina})
        if not lote:
            break
        productos += lote
        if len(lote) < 200:
            break
        pagina += 1

    vivos = [p for p in productos if is_product_available(p)]
    ocultos = [p for p in productos if not is_product_available(p)]
    print(f"  {len(productos)} productos en total")
    print(f"  {len(vivos)} visibles para el bot")
    print(f"  {len(ocultos)} invisibles (sin stock o despublicados)\n")

    invisibles, en_riesgo, ok = [], [], 0
    for n, p in enumerate(vivos, 1):
        if n % 10 == 0:
            print(f"  ...{n}/{len(vivos)}")
        nombre = (p.get("name", {}).get("es") or "?")
        pid = p["id"]
        modelo = modelo_de(nombre)
        propias = [cats.get(c["id"], "").lower() for c in (p.get("categories") or [])]
        palabras = []
        for c in propias:
            palabras += CATEGORIA_A_PALABRA.get(c, [])
        palabras = list(dict.fromkeys(palabras))[:2]

        consultas = []
        if modelo:
            consultas.append(" ".join(modelo[:2]))
            for w in palabras:
                consultas.append(f"{w} {modelo[0]}")
        else:
            consultas += palabras

        if not consultas:
            continue
        fallan = [q for q in consultas if pid not in buscar(q)]
        if len(fallan) == len(consultas):
            invisibles.append((nombre, consultas))
        elif fallan:
            en_riesgo.append((nombre, fallan))
        else:
            ok += 1

    print("\n" + "=" * 74)
    print(f"  se encuentran con todas las formas probadas:  {ok}")
    print(f"  fallan en ALGUNA forma de pedirlos:           {len(en_riesgo)}")
    print(f"  NO se encuentran de ninguna forma:            {len(invisibles)}   <-- venta perdida segura")
    print("=" * 74)

    if invisibles:
        print("\nNO SE ENCUENTRAN DE NINGUNA FORMA:\n")
        for nombre, qs in invisibles:
            print(f"  {nombre}")
            print(f"     se probo: {qs}\n")
    if en_riesgo:
        print("\nFALLAN EN ALGUNA FORMA (la clienta tiene que acertar la palabra):\n")
        for nombre, qs in (en_riesgo if args.detalle else en_riesgo[:15]):
            print(f"  {nombre[:62]}")
            print(f"     no lo encuentra con: {qs}")
        if not args.detalle and len(en_riesgo) > 15:
            print(f"\n  ...y {len(en_riesgo) - 15} mas (correr con --detalle)")

    if ocultos:
        print(f"\n\nINVISIBLES POR STOCK O DESPUBLICADOS ({len(ocultos)}):")
        for p in ocultos[:20]:
            nombre = (p.get("name", {}).get("es") or "?")
            print(f"  {'despublicado' if not p.get('published') else 'sin stock   '}  {nombre[:58]}")
        if len(ocultos) > 20:
            print(f"  ...y {len(ocultos) - 20} mas")


if __name__ == "__main__":
    main()
