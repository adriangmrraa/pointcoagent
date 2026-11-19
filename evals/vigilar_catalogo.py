"""Vigila el catálogo: avisa qué cambió y si el bot encuentra lo nuevo.

La dueña carga los productos; nosotros nos enteramos cuando una clienta se
queja. Esto da vuelta esa secuencia: compara el catálogo contra la última vez
que se corrió y avisa qué cambió, probando además si el bot puede encontrar lo
que se sumó.

Caso que lo motiva (2026-11-19): se cargó "LEDA SO DANCA" sin que nadie avisara.
Una clienta pidió "la malla leda", el bot dijo que no la tenía, y lo
descubrimos leyendo conversaciones una semana después.

Qué avisa:
  - productos NUEVOS, y si el bot los encuentra como los pediría una clienta
  - productos que se quedaron SIN STOCK (dejan de verse)
  - productos que VOLVIERON a tener stock
  - cambios de nombre o de precio

Es de solo lectura sobre Tienda Nube. El único archivo que escribe es su propia
foto del catálogo, fuera del repo.

Uso:
    python evals/vigilar_catalogo.py
"""
import io
import json
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "orchestrator_service"))
from catalog import (is_product_available, query_terms,  # noqa: E402
                     rank_widened_results, sin_acentos)

FOTO = Path.home() / "OneDrive" / "Documentos" / "pointcoagent-informes" / "catalogo-foto.json"

GENERICAS = {
    "so", "danca", "danza", "de", "del", "la", "el", "los", "las", "para", "con",
    "sin", "capezio", "grishko", "sansha", "bunheads", "pointe", "coach",
    "americano", "americana", "ballet", "zapatillas", "zapatilla", "media",
    "medias", "punta", "puntas", "leotardo", "malla", "maillot", "bolso",
    "bolsa", "mochila", "mochilas", "puntera", "punteras", "metatarsiana",
    "metatarsianas", "cintas", "cinta", "elasticos", "elastico", "protector",
    "protectores", "separador", "separadores", "pollera", "pollerin", "falda",
    "infantil", "adulto", "tela", "silicona", "gel", "negro", "rosa", "blanco",
}


def env():
    d = {}
    for linea in io.open(RAIZ / ".env", encoding="utf-8", errors="replace"):
        linea = linea.strip()
        if "=" in linea and not linea.startswith("#"):
            k, v = linea.split("=", 1)
            d[k.strip()] = v.strip()
    return d


E = env()
H = {"Authentication": f"bearer {E['TIENDANUBE_ACCESS_TOKEN']}", "User-Agent": "vigilar-catalogo/1.0"}
_c = {}


def api(params):
    k = json.dumps(params, sort_keys=True)
    if k in _c:
        return _c[k]
    url = (f"https://api.tiendanube.com/v1/{E['TIENDANUBE_STORE_ID']}/products?"
           + urllib.parse.urlencode(params, encoding="utf-8"))
    with urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=40) as r:
        _c[k] = json.loads(r.read().decode("utf-8"))
    return _c[k]


def catalogo():
    todos, pagina = [], 1
    while True:
        lote = api({"per_page": 200, "page": pagina})
        if not lote:
            break
        todos += lote
        if len(lote) < 200:
            break
        pagina += 1
    return todos


def resumen(p):
    var = p.get("variants") or []
    return {
        "nombre": (p.get("name", {}).get("es") or "?"),
        "precio": (var[0].get("price") if var else None),
        "visible": is_product_available(p),
        "stock": sum((v.get("stock") or 0) for v in var if v.get("stock") is not None),
    }


def modelo_de(nombre):
    limpio = re.split(r"[|/,]", nombre)[0]
    return [w for w in re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÑñ0-9]+", limpio)
            if sin_acentos(w) not in GENERICAS and len(w) > 2]


def encuentra(pid, q):
    vivos = [p for p in api({"q": q, "per_page": 20}) if is_product_available(p)]
    if vivos:
        return pid in {p["id"] for p in vivos}
    t = query_terms(q)[:3]
    if len(t) < 2:
        return False
    parciales = [[{"id": p["id"], "name": (p.get("name", {}).get("es") or "")}
                  for p in api({"q": x, "per_page": 30}) if is_product_available(p)] for x in t]
    parciales.sort(key=len)
    vistos, union = set(), []
    for pr in parciales:
        for p in pr:
            if p["id"] not in vistos:
                vistos.add(p["id"]); union.append(p)
    return pid in {p["id"] for p in rank_widened_results(q, union)}


def main():
    print(f"Mirando el catálogo... ({datetime.now():%d/%m/%Y %H:%M})")
    ahora = {str(p["id"]): resumen(p) for p in catalogo()}
    visibles = sum(1 for v in ahora.values() if v["visible"])
    print(f"  {len(ahora)} productos, {visibles} visibles para el bot\n")

    if not FOTO.exists():
        FOTO.parent.mkdir(parents=True, exist_ok=True)
        io.open(FOTO, "w", encoding="utf-8").write(json.dumps(ahora, ensure_ascii=False, indent=1))
        print("Primera corrida: guardé la foto del catálogo.")
        print("La próxima vez te voy a decir qué cambió.")
        return

    antes = json.loads(io.open(FOTO, encoding="utf-8").read())
    nuevos = [(k, v) for k, v in ahora.items() if k not in antes]
    borrados = [(k, v) for k, v in antes.items() if k not in ahora]
    agotados, repuestos, renombrados, reprecio = [], [], [], []
    for k, v in ahora.items():
        a = antes.get(k)
        if not a:
            continue
        if a["visible"] and not v["visible"]:
            agotados.append(v)
        if not a["visible"] and v["visible"]:
            repuestos.append(v)
        if a["nombre"] != v["nombre"]:
            renombrados.append((a["nombre"], v["nombre"]))
        if a["precio"] != v["precio"]:
            reprecio.append((v["nombre"], a["precio"], v["precio"]))

    hubo = False
    if nuevos:
        hubo = True
        print(f"PRODUCTOS NUEVOS ({len(nuevos)}) — ¿el bot los encuentra?\n")
        for pid, v in nuevos:
            modelo = modelo_de(v["nombre"])
            consultas = ([modelo[0]] if modelo else []) + ([" ".join(modelo[:2])] if len(modelo) > 1 else [])
            if not consultas:
                consultas = [v["nombre"].split()[0]]
            fallan = [q for q in consultas if not encuentra(int(pid), q)]
            estado = "NO LO ENCUENTRA" if len(fallan) == len(consultas) else ("parcial" if fallan else "OK")
            print(f"  [{estado}]  {v['nombre'][:58]}")
            print(f"     stock {v['stock']}  ·  ${v['precio']}  ·  probado con {consultas}")
            if fallan:
                print(f"     FALLA con: {fallan}")
            print()
    if agotados:
        hubo = True
        print(f"SE QUEDARON SIN STOCK ({len(agotados)}) — el bot deja de mostrarlos:")
        for v in agotados:
            print(f"  {v['nombre'][:62]}")
        print()
    if repuestos:
        hubo = True
        print(f"VOLVIERON A TENER STOCK ({len(repuestos)}):")
        for v in repuestos:
            print(f"  {v['nombre'][:62]}  (stock {v['stock']})")
        print()
    if renombrados:
        hubo = True
        print(f"CAMBIARON DE NOMBRE ({len(renombrados)}) — ojo, puede romper búsquedas:")
        for a, d in renombrados:
            print(f"  {a[:40]}  ->  {d[:40]}")
        print()
    if reprecio:
        hubo = True
        print(f"CAMBIARON DE PRECIO ({len(reprecio)}):")
        for n, a, d in reprecio:
            print(f"  {n[:48]}  ${a} -> ${d}")
        print()
    if borrados:
        hubo = True
        print(f"YA NO ESTÁN EN EL CATÁLOGO ({len(borrados)}):")
        for _, v in borrados:
            print(f"  {v['nombre'][:62]}")
        print()
    if not hubo:
        print("Sin cambios desde la última vez.")

    io.open(FOTO, "w", encoding="utf-8").write(json.dumps(ahora, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
