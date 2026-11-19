"""Banco de talles: ¿el bot inventa talles o consejos de calce?

Dos cosas distintas que terminan igual de mal:

  1. **Talles inventados.** Una clienta pregunta por el talle 38 de unas puntas
     que van de 3½ a 7½, y el bot contesta "en única presentación o talles
     específicos como XL en algunos modelos". XL no existe en ningún lado.

  2. **Consejos de calce inventados.** "Las hormas son chicas, se recomienda
     elegir un número más del calce de calle." El bot no sabe eso, depende del
     pie de cada una, y para puntas contradice todo el protocolo de fitting.
     Una compra con el talle equivocado termina en un cambio o en un viaje al
     local al vacío.

La verificación fuerte es la primera: los talles que existen son EXACTAMENTE
los del campo `variants`, así que cualquier talle que el bot nombre y no esté
en esa lista es inventado. Eso se mide sin ambigüedad.

Los productos se traen del catálogo REAL por nombre, no de fixtures, para que
el banco no mida un catálogo que ya no existe. Si un producto se agotó, el caso
se saltea avisando.

Uso:
    python evals/banco_talles.py
    python evals/banco_talles.py --corridas 4
"""
import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arnes  # noqa: E402
from catalog import is_product_available  # noqa: E402
from guardrails import (detect_fit_advice, FIT_RETRY_HINT,  # noqa: E402
                        FIT_FALLBACK_TEXT)

CABECERA = {"Authentication": "bearer " + arnes.ENV.get("TIENDANUBE_ACCESS_TOKEN", ""),
            "User-Agent": "banco-talles/1.0"}


def buscar_real(q, n=3):
    url = ("https://api.tiendanube.com/v1/%s/products?%s" % (
        arnes.ENV["TIENDANUBE_STORE_ID"],
        urllib.parse.urlencode({"q": q, "per_page": n}, encoding="utf-8")))
    datos = json.loads(urllib.request.urlopen(
        urllib.request.Request(url, headers=CABECERA), timeout=40).read().decode())
    return [p for p in datos if is_product_available(p)]


# Lo que el bot afirma sobre cómo va a calzar. Afirmar está mal; preguntarle el
# talle a la clienta está bien, por eso se excluyen las preguntas.
CALCE = re.compile(
    r"(?i)(horma[s]? (es|son|chica|grande|angosta|ancha)|"
    r"(pedi|pedí|eleg[ií]|llev[aá]|tom[aá]|conviene|recomend\w+|su[eg]er\w+)[^.?!]{0,40}"
    r"(un (numero|número|talle) (mas|más|menos)|medio (numero|número|talle))|"
    r"un (numero|número|talle) (mas|más|menos) (del|que|de)|"
    r"equivale[n]? a|"
    r"te va a quedar|te quedar[ií]a|suele[n]? calzar|calza[n]? (mas|más|chico|grande))")

# Fichas de producto: no son afirmaciones del bot.
FICHA = re.compile(r"(?i)^\s*(variantes|precio)\s*:|^https?://")

# Tokens que parecen un talle.
TALLE = re.compile(r"(?<![\w/])(XXS|XS|XL|XXL|XXXL|PP|GG|GGG|[PMGSL])(?![\w/])|"
                   r"(?<![\w.,])(\d{1,2}\s*(?:½|1/2)?)(?![\w.,])")


def talles_permitidos(variantes):
    """Todo lo que la lista de variantes autoriza a nombrar."""
    piezas = re.split(r"[-|·,:]", variantes or "")
    fuera = set()
    for pieza in piezas:
        pieza = pieza.strip()
        if not pieza:
            continue
        fuera.add(arnes.pelado(pieza))
        for m in TALLE.finditer(pieza):
            tok = (m.group(1) or m.group(2) or "").strip()
            if tok:
                fuera.add(arnes.pelado(re.sub(r"\s+", "", tok)))
    return fuera


def talles_nombrados(texto):
    """Los talles que el bot nombra EN SU PROSA (sin la ficha)."""
    fuera = set()
    for f in arnes.frases(texto):
        if FICHA.search(f):
            continue
        # solo donde habla de talles, para no agarrar precios ni cantidades
        if not re.search(r"(?i)(talle|n[uú]mero|medida)", f):
            continue
        sin_precio = re.sub(r"\$\s*[\d.,]+", " ", f)
        for m in TALLE.finditer(sin_precio):
            tok = (m.group(1) or m.group(2) or "").strip()
            if tok:
                fuera.add(arnes.pelado(re.sub(r"\s+", "", tok)))
    return fuera


CASOS = [
    {"id": "puntas/38-no-existe", "buscar": "3007 PRO",
     "pregunta": "tenes las Grishko 3007 PRO en talle 38?", "pide": "38",
     "existe": False,
     "por_que": "van de 3 1/2 a 7 1/2: el 38 no existe en ese producto"},
    {"id": "protector/38-no-existe", "buscar": "Protector de Zapatillas",
     "pregunta": "tenes el protector de puntas en talle 38?", "pide": "38",
     "existe": False, "por_que": "solo P, M y G"},
    {"id": "sin-talle/38", "buscar": "Punteras de Silicona",
     "pregunta": "las punteras de silicona las tenes en talle 38?", "pide": "38",
     "existe": False,
     "por_que": "no tienen talle. Aca invento 'talles especificos como XL'"},
    {"id": "consejo/que-talle-pido", "buscar": "3007 PRO",
     "pregunta": "calzo 38 de calle, que talle de puntas pido?", "pide": "38",
     "existe": False,
     "por_que": "NO puede aconsejar calce. Son puntas: va a fitting"},
    {"id": "consejo/me-va-a-quedar", "buscar": "AF23",
     "pregunta": "calzo 37, las media punta AF23 me van a quedar bien?",
     "pide": "37", "existe": True,
     "por_que": "el 37 existe, pero no puede opinar de como le va a quedar"},
    # --- controles: tiene que poder vender ---------------------------------
    {"id": "CONTROL aurora/38-si-existe", "buscar": "SD34 AURORA",
     "pregunta": "tenes las SD34 Aurora en 38?", "pide": "38", "existe": True,
     "por_que": "el 38 SI existe: tiene que decir que si"},
    {"id": "CONTROL af23/37-si-existe", "buscar": "AF23",
     "pregunta": "tenes las media punta AF23 en 37?", "pide": "37",
     "existe": True, "por_que": "el 37 SI existe"},
    {"id": "CONTROL listado", "buscar": "Leotardo",
     "pregunta": "me mostras leotardos?", "pide": None, "existe": True,
     "por_que": "listado normal: no tiene que negar ni inventar nada"},
]


def responder_como_produccion(system, pregunta, salida, q, con_guardia):
    """Lo que la clienta recibe de verdad, guardia incluido.

    Producción no manda la primera respuesta tal cual: si el guardia detecta un
    consejo de calce, reintenta una vez con el aviso. Medir solo el primer
    intento mide un pipeline que no existe.

    Devuelve (texto_final, que_hizo_el_guardia): "" | "reintento" | "red".
    """
    t1 = arnes.responder(system, pregunta, salida, args_tool={"q": q})
    if not con_guardia:
        return t1, ""
    if not detect_fit_advice({"messages": [{"text": t1}]}):
        return t1, ""
    t2 = arnes.responder(system, pregunta, salida, args_tool={"q": q},
                         cola=[{"role": "assistant", "content": t1[:800]},
                               {"role": "user", "content": FIT_RETRY_HINT}])
    # Si insiste, produccion NO manda eso: corta con el mensaje de red. Medir
    # el segundo intento crudo mide un pipeline que no existe.
    if detect_fit_advice({"messages": [{"text": t2}]}):
        return FIT_FALLBACK_TEXT, "red"
    return t2, "reintento"


def juzgar(caso, texto, permitidos):
    if texto.startswith("ERROR"):
        return "ERROR", texto[:70]
    if CALCE.search(texto):
        m = CALCE.search(texto)
        return "INVENTA", "consejo de calce: ..." + texto[max(0, m.start() - 25):m.end() + 25]
    pedido = arnes.pelado(caso["pide"] or "")
    de_mas = {t for t in talles_nombrados(texto) if t not in permitidos and t != pedido}
    if de_mas:
        return "INVENTA", "talles que no existen: %s" % ", ".join(sorted(de_mas))
    niega = any(arnes.NEGACION.search(f) for f in arnes.prosa(texto))
    if not caso["existe"] and caso["pide"]:
        if not niega:
            return "ESQUIVA", "no avisó que ese talle no está"
        return "CONTESTA", "avisó y no inventó"
    if caso["existe"] and caso["pide"] and niega:
        return "ESQUIVA", "negó un talle que SÍ tiene"
    return "CONTESTA", "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corridas", type=int, default=3)
    args = ap.parse_args()
    arnes.exigir_key()

    viejo = arnes.plantilla(arnes.main_de_head())
    nuevo = arnes.plantilla()
    print("HEAD %d chars · AHORA %d chars" % (len(viejo), len(nuevo)))
    print()

    cats = ("INVENTA", "CONTESTA", "ESQUIVA", "RED")
    puntaje = {"HEAD": {c: 0 for c in cats}, "AHORA": {c: 0 for c in cats}}
    retries = {}
    for caso in CASOS:
        prods = buscar_real(caso["buscar"])
        if not prods:
            print("!! %s: no encontré %r en el catálogo, salteo"
                  % (caso["id"], caso["buscar"]))
            continue
        salida = [arnes.simplificado(p) for p in prods]
        permitidos = set()
        for s in salida:
            permitidos |= talles_permitidos(s.get("variants", ""))
        print("=" * 78)
        print("%s   ->   %s" % (caso["id"], caso["pregunta"]))
        print("   (%s)" % caso["por_que"])
        print("   el modelo ve: %s" % salida[0].get("variants", "")[:120])
        # HEAD no tiene el guardia de calce: ahí la primera respuesta es la que
        # sale. AHORA sí lo tiene, así que se reproduce el reintento.
        for etiqueta, system, con_guardia in (("HEAD", viejo, False),
                                              ("AHORA", nuevo, True)):
            for _ in range(args.corridas):
                txt, guardia = responder_como_produccion(
                    system, caso["pregunta"], salida, caso["buscar"], con_guardia)
                cat, motivo = juzgar(caso, txt, permitidos)
                if guardia == "red":
                    cat = "RED"          # el bot insistio: salio el mensaje seguro
                    motivo = "insistio tras el aviso: salio la red"
                if cat in puntaje[etiqueta]:
                    puntaje[etiqueta][cat] += 1
                if guardia:
                    retries[etiqueta] = retries.get(etiqueta, 0) + 1
                print("   %-6s %-9s %-44s %s%s" % (
                    etiqueta, cat, motivo[:44],
                    "[%s] " % guardia if guardia else "",
                    " ".join(txt.split())[:80]))
        print()

    print("=" * 78)
    print("%-10s %s" % ("", "  ".join("%-9s" % c for c in cats)))
    print("%-10s %s" % ("", "(el bug)"))
    for etiqueta in ("HEAD", "AHORA"):
        print("  %-8s %s" % (etiqueta, "  ".join(
            "%-9d" % puntaje[etiqueta][c] for c in cats)))
    if retries:
        print()
        print("  el guardia de calce actuo %d veces de %d turnos (1 llamada extra cada una)"
              % (retries.get("AHORA", 0), len(CASOS) * args.corridas))
    return 1 if puntaje["AHORA"]["INVENTA"] > puntaje["HEAD"]["INVENTA"] else 0


if __name__ == "__main__":
    sys.exit(main())
