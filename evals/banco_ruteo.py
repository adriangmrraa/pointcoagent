"""Banco de ruteo: ¿el bot elige la herramienta correcta?

Este banco existe por una razón concreta. Tres veces se intentó mejorar el
ruteo desde el prompt (commits d27a8fb y 19385f0, y un tercer intento) y las
tres se revirtió, porque arreglaba un caso y rompía otro que nadie estaba
mirando: el Endurecedor SHOE GLUE y la Bolsa de Malla MESH BAGS dejaban de
aparecer. Sin una medición de antes y después, cada cambio de prompt es una
apuesta.

Mide el prompt de HEAD (lo que está corriendo) contra el del working tree (lo
que vamos a subir), con las herramientas y el prompt leídos del main.py de
verdad.

Los casos de CONTROL importan más que los casos nuevos: son los que avisan que
rompimos algo que funcionaba.

Uso:
    python evals/banco_ruteo.py
    python evals/banco_ruteo.py --corridas 3
    python evals/banco_ruteo.py --solo dedos      # filtra por id
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arnes  # noqa: E402

BUSCAR = {"search_specific_products", "search_by_category", "search_by_store_category"}

# espera: conjunto de tools aceptables, o None = tiene que NO llamar a ninguna
CASOS = [
    # --- lo que esta roto hoy -------------------------------------------------
    {"id": "dedos/dedales", "q": "tenes dedales?", "espera": BUSCAR,
     "por_que": "'dedales' esta en el diccionario de sinonimos documentado. "
                "Caso real 2026-10-13: el bot dijo 'no tenemos separadores de "
                "dedos' teniendo 6 productos con stock."},
    {"id": "dedos/spacemakers", "q": "tenes spacemakers?", "espera": BUSCAR,
     "por_que": "'spacemakers' YA figura en el diccionario del prompt y aun asi "
                "disparaba la pregunta de desambiguacion."},
    {"id": "dedos/descriptivo", "q": "necesito algo para separar los dedos del pie",
     "espera": BUSCAR, "por_que": "pedido descriptivo, sin la palabra exacta"},
    {"id": "endurecedor", "q": "tenes endurecedor para las puntas?", "espera": BUSCAR,
     "por_que": "REGRESION HISTORICA: el SHOE GLUE dejo de encontrarse dos veces"},

    # --- controles: desambiguacion que SI corresponde -----------------------
    {"id": "ambiguo/protectores", "q": "tenes protectores?", "espera": None,
     "por_que": "ambiguo de verdad: puede ser de punta o de dedos. Preguntar esta bien."},
    {"id": "ambiguo/almohadillas", "q": "tenes almohadillas?", "espera": None,
     "por_que": "ambiguo: para puntas o para dedos"},
    {"id": "ambiguo/prot-punta", "q": "tenes protectores de punta?", "espera": None,
     "por_que": "ambiguo sin calificador"},

    # --- controles: terminos que NO son ambiguos ----------------------------
    {"id": "no-ambiguo/punteras", "q": "tenes punteras de gel?", "espera": BUSCAR,
     "por_que": "'punteras de gel' va directo a PUNTERAS, no se pregunta"},
    {"id": "no-ambiguo/separadores", "q": "tenes separadores de dedos?", "espera": BUSCAR,
     "por_que": "va directo a SEPARADORES DE DEDOS"},
    {"id": "no-ambiguo/toppers", "q": "tenes toppers de puntas?", "espera": BUSCAR,
     "por_que": "va directo a PROTECTORES DE PUNTAS"},

    # --- controles: el resto del ruteo que ya funciona ----------------------
    {"id": "bolsos", "q": "me mostras bolsos?", "espera": {"search_by_store_category"},
     "por_que": "OBLIGATORIO por categoria: las mochilas no tienen 'bolso' en el nombre"},
    {"id": "mochilas", "q": "tenes mochilas de danza?", "espera": BUSCAR,
     "por_que": "mochila es sinonimo de BOLSOS"},
    {"id": "malla", "q": "tenes mallas?", "espera": BUSCAR,
     "por_que": "'malla' es sinonimo de LEOTARDOS"},
    {"id": "bolsa-red", "q": "tenes bolsa de red para las zapatillas?", "espera": BUSCAR,
     "por_que": "REGRESION HISTORICA: la Bolsa de Malla MESH BAGS"},
    {"id": "medias/cancan", "q": "tenes cancanes?", "espera": BUSCAR,
     "por_que": "'cancan' es sinonimo de MEDIAS"},
    {"id": "vago", "q": "hola, que tienen?", "espera": {"browse_general_storefront"},
     "por_que": "consulta vaga: catalogo general"},
    {"id": "promos", "q": "tienen alguna promo?", "espera": {"cupones_list"},
     "por_que": "promociones vigentes"},
    {"id": "pedido", "q": "queria saber como viene mi pedido 12345",
     "espera": {"orders"}, "por_que": "consulta de pedido"},

    # --- controles: derivacion obligatoria ----------------------------------
    {"id": "deriv/local", "q": "puedo pasar por el local a retirar?",
     "espera": {"derivhumano"},
     "por_que": "Regla 10: toda intencion presencial deriva. Una clienta fue al "
                "local tras una confirmacion del bot."},
    {"id": "deriv/fitting", "q": "necesito turno para probarme puntas",
     "espera": {"derivhumano"}, "por_que": "fitting deriva"},
    {"id": "deriv/cambio", "q": "me quedo chica, puedo cambiarla?",
     "espera": {"derivhumano"}, "por_que": "Regla 11: cambios y devoluciones derivan"},
    {"id": "deriv/descuento", "q": "me podes hacer un descuento si llevo dos?",
     "espera": {"derivhumano"}, "por_que": "Regla 11: negociar precio deriva"},

    # --- control: fuera de catalogo (Regla 8) -------------------------------
    {"id": "fuera-rubro", "q": "tenes zapatillas de tap?", "espera": None,
     "por_que": "Regla 8: fuera del rubro, NO buscar, informar y ofrecer alternativa"},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corridas", type=int, default=2)
    ap.add_argument("--solo", default="")
    args = ap.parse_args()
    arnes.exigir_key()

    casos = [c for c in CASOS if args.solo in c["id"]]
    viejo = arnes.plantilla(arnes.main_de_head())
    nuevo = arnes.plantilla()
    tools_viejas = arnes.herramientas(arnes.DATOS / "_main_head.py")
    tools_nuevas = arnes.herramientas()
    print("HEAD  : prompt %d chars, %d tools" % (len(viejo), len(tools_viejas)))
    print("AHORA : prompt %d chars, %d tools" % (len(nuevo), len(tools_nuevas)))
    if len(casos) < len(CASOS):
        print("(filtrado: %d de %d casos)" % (len(casos), len(CASOS)))
    print()

    brazos = [("HEAD", viejo, tools_viejas), ("AHORA", nuevo, tools_nuevas)]
    puntos = {b[0]: [0, 0] for b in brazos}
    rotos = []

    for caso in casos:
        esperado = caso["espera"]
        linea = {}
        for etiqueta, system, tools in brazos:
            marcas = []
            for _ in range(args.corridas):
                nombre, extra = arnes.que_tool_llama(system, caso["q"], tools)
                if esperado is None:
                    bien = nombre is None
                else:
                    bien = nombre in esperado
                puntos[etiqueta][0] += 1 if bien else 0
                puntos[etiqueta][1] += 1
                marcas.append((nombre or "sin-tool") if not bien
                              else ("ok:" + (nombre or "sin-tool")))
            linea[etiqueta] = marcas
        falla_ahora = sum(1 for m in linea["AHORA"] if not m.startswith("ok:"))
        falla_head = sum(1 for m in linea["HEAD"] if not m.startswith("ok:"))
        estado = "   "
        if falla_ahora > falla_head:
            estado = "!!!"            # lo rompimos
            rotos.append(caso["id"])
        elif falla_ahora < falla_head:
            estado = "+++"            # lo arreglamos
        quiere = "sin-tool" if esperado is None else "/".join(sorted(esperado))
        print("%s %-22s espera %-46s" % (estado, caso["id"], quiere[:46]))
        for etiqueta in ("HEAD", "AHORA"):
            print("      %-6s %s" % (etiqueta, "  ".join(linea[etiqueta])))

    print()
    print("=" * 78)
    for etiqueta, _, _ in brazos:
        b, t = puntos[etiqueta]
        print("  %-6s %d/%d ruteos correctos" % (etiqueta, b, t))
    if rotos:
        print()
        print("  ROMPIMOS: " + ", ".join(rotos))
        return 1
    print()
    print("  no rompimos ningun caso")
    return 0


if __name__ == "__main__":
    sys.exit(main())
