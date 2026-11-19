"""Tests del filtro de disponibilidad de catálogo (orchestrator_service/catalog.py).

Son funciones puras: corren sin Redis, sin Postgres y sin API keys.
    python -m pytest tests/test_catalog.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "orchestrator_service"))

from catalog import (variant_has_stock, is_product_available, available_variant_values,
                     pick_category_id, variant_availability_text, SIN_VARIANTES)


def v(stock, values=None):
    return {"stock": stock, "values": values or []}


# --- variant_has_stock ---

def test_stock_positivo_disponible():
    assert variant_has_stock(v(5)) is True

def test_stock_cero_agotado():
    assert variant_has_stock(v(0)) is False

def test_stock_none_es_infinito():
    # Tienda Nube: stock null = sin control de stock = disponible
    assert variant_has_stock(v(None)) is True

def test_stock_negativo_o_basura():
    assert variant_has_stock(v(-1)) is False
    assert variant_has_stock(v("abc")) is False
    assert variant_has_stock("no-es-dict") is False


# --- is_product_available ---

def test_producto_con_una_variante_en_stock():
    p = {"published": True, "variants": [v(0), v(3)]}
    assert is_product_available(p) is True

def test_producto_todo_agotado_se_oculta():
    p = {"published": True, "variants": [v(0), v(0)]}
    assert is_product_available(p) is False

def test_producto_despublicado_se_oculta_aunque_tenga_stock():
    p = {"published": False, "variants": [v(10)]}
    assert is_product_available(p) is False

def test_producto_stock_infinito():
    p = {"published": True, "variants": [v(None)]}
    assert is_product_available(p) is True

def test_producto_sin_variantes_no_se_filtra():
    # Sin dato de stock preferimos mostrar antes que ocultar por error
    assert is_product_available({"published": True, "variants": []}) is True
    assert is_product_available({"published": True}) is True

def test_entrada_invalida():
    assert is_product_available(None) is False
    assert is_product_available("texto") is False


# --- available_variant_values ---

def test_variantes_agotadas_no_aparecen():
    variants = [
        v(0, [{"es": "Rojo"}]),
        v(4, [{"es": "Negro"}]),
        v(None, [{"es": "Rosa"}]),
    ]
    assert available_variant_values(variants) == ["Negro", "Rosa"]

def test_valores_duplicados_se_unifican_preservando_orden():
    variants = [
        v(2, [{"es": "36"}, {"es": "Negro"}]),
        v(1, [{"es": "37"}, {"es": "Negro"}]),
    ]
    assert available_variant_values(variants) == ["36", "Negro", "37"]

def test_fallback_ingles():
    assert available_variant_values([v(1, [{"en": "Black"}])]) == ["Black"]

def test_lista_invalida():
    assert available_variant_values(None) == []
    assert available_variant_values("x") == []


# --- pick_category_id (búsqueda por categoría real) ---

CATS = [
    {"id": 34839644, "name": {"es": "Bolsos"}},
    {"id": 35012479, "name": {"es": "Accesorios"}},
    {"id": 35012485, "name": {"es": "Accesorios para el pie"}},
]

def test_categoria_match_exacto():
    assert pick_category_id(CATS, "Bolsos") == 34839644
    assert pick_category_id(CATS, "bolsos") == 34839644  # case-insensitive

def test_categoria_match_parcial():
    # "bolso" (singular) encuentra "Bolsos"
    assert pick_category_id(CATS, "bolso") == 34839644

def test_categoria_prioriza_exacto_sobre_parcial():
    # "Accesorios" exacto no debe caer en "Accesorios para el pie"
    assert pick_category_id(CATS, "Accesorios") == 35012479

def test_categoria_inexistente_devuelve_none():
    assert pick_category_id(CATS, "Zapatillas") is None

def test_categoria_entradas_invalidas():
    assert pick_category_id(CATS, "") is None
    assert pick_category_id(CATS, None) is None
    assert pick_category_id(None, "Bolsos") is None


if __name__ == "__main__":
    fallos = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  OK   {name}")
            except AssertionError as e:
                fallos += 1
                print(f"  FAIL {name}: {e}")
    print(f"\n{'TODO OK' if fallos == 0 else str(fallos) + ' FALLOS'}")
    sys.exit(1 if fallos else 0)


# --- Busqueda ampliada (caso real 2026-07-23: "elasticos Grishko") ----------

from catalog import sin_acentos, query_terms, name_match_score, rank_widened_results

# Nombres REALES traidos de la API de Tienda Nube el 2026-08-18 con q="Grishko"
# (19 resultados; el que la clienta queria estaba en el puesto 13).
GRISHKO_REAL = [
    "Zapatillas de Puntas Grishko 3007 PRO",
    "Zapatillas de Puntas Grishko 3007 PRO FLEX",
    "Zapatillas de Puntas Grishko DREAM POINTE",
    "Zapatillas de Puntas Grishko STREAM POINTE",
    "Zapatillas de Puntas Grishko STAR POINTE",
    "Punteras Moleskin Grishko",
    "Punteras de tela Grishko",
    "Punteras de Silicona Grishko",
    "Calcetines SOCKS Grishko",
    "Zapatillas Mediapuntas Dream Stretch Grishko",
    "Bolso Grishko",
    "Leotardo para danza ELLA Grishko",
    "Cintas Elastizadas Grishko",
    "Cuerito para plataforma de Zapatillas de Puntas tamaño PEQUEÑO",
    "Cuerito para Plataforma Zapatillas de Puntas tamaño GRANDE",
    "Leotardo para danza EFFIE Grishko",
    "Leotardo para danza SIA Grishko",
    "Leotardo para danza SIBA Grishko",
    "Protector de Puntas Grishko",
]
# Los 2 que devuelve q="Elasticos" (ninguno es Grishko).
ELASTICOS_REAL = ["Elásticos Bolt Bunheads Capezio",
                  "Elásticos ACANALADOS PARA ZAPATILLAS DE PUNTAS"]


def _prods(nombres):
    return [{"id": i, "name": n} for i, n in enumerate(nombres)]


def test_el_caso_real_queda_primero():
    # Union de ambas busquedas por termino suelto, en el orden en que las
    # devuelve la API. Sin re-ranking, "Cintas Elastizadas Grishko" queda 13ro.
    union = _prods(ELASTICOS_REAL + GRISHKO_REAL)
    top = rank_widened_results("Elásticos Grishko", union)
    assert top[0]["name"] == "Cintas Elastizadas Grishko"


def test_el_prefijo_puentea_elasticos_con_elastizadas():
    terminos = query_terms("Elásticos Grishko")
    assert name_match_score("Cintas Elastizadas Grishko", terminos) == 2
    assert name_match_score("Zapatillas de Puntas Grishko 3007 PRO", terminos) == 1
    assert name_match_score("Elásticos Bolt Bunheads Capezio", terminos) == 1


def test_descarta_lo_que_no_matchea_ningun_termino():
    union = _prods(["Cintas Elastizadas Grishko", "Leotardo Maria SO DANCA"])
    top = rank_widened_results("Elásticos Grishko", union)
    assert [p["name"] for p in top] == ["Cintas Elastizadas Grishko"]


def test_no_reordena_arbitrariamente_los_empates():
    # Las 3 matchean 'punteras' + 'grishko': empatan, y se conserva el orden en
    # que las devolvio la API en vez de reordenarlas de forma arbitraria.
    union = _prods(["Punteras Moleskin Grishko", "Punteras de tela Grishko",
                    "Punteras de Silicona Grishko"])
    top = rank_widened_results("punteras gel Grishko", union)
    assert [p["name"] for p in top] == ["Punteras Moleskin Grishko",
                                        "Punteras de tela Grishko",
                                        "Punteras de Silicona Grishko"]


def test_la_marca_sola_no_alcanza_para_entrar():
    # Pidio elasticos: un bolso Grishko matchea la marca pero no es lo que pidio.
    union = _prods(["Bolso Grishko", "Protector de Puntas Grishko"])
    assert rank_widened_results("Elásticos Grishko", union) == []


def test_si_la_consulta_es_solo_marca_no_se_exige_tipo():
    # "algo de Grishko": no hay termino de tipo con que filtrar, se deja pasar.
    union = _prods(["Bolso Grishko", "Punteras Moleskin Grishko"])
    top = [p["name"] for p in rank_widened_results("Grishko Capezio", union)]
    assert top == ["Bolso Grishko", "Punteras Moleskin Grishko"]


def test_respeta_el_limite():
    assert len(rank_widened_results("Grishko puntas", _prods(GRISHKO_REAL), limite=5)) == 5


def test_normalizacion_de_terminos():
    assert sin_acentos("Elásticos") == "elasticos"
    # Las stopwords no cuentan como termino (si contaran, "de"/"para" matchearian todo).
    assert query_terms("cintas de saten para puntas") == ["cintas", "saten", "puntas"]
    assert query_terms("") == []


def test_una_sola_palabra_no_se_amplia():
    # Con un termino la ampliacion no puede aportar nada: main.py corta antes.
    assert len(query_terms("Grishko")) == 1


def test_entradas_raras_no_explotan():
    assert rank_widened_results("Elásticos Grishko", []) == []
    assert rank_widened_results("", _prods(GRISHKO_REAL)) == []
    assert rank_widened_results("Elásticos Grishko", None) == []
    assert rank_widened_results("Elásticos Grishko", [None, "basura", {"name": None}]) == []


def test_no_mezcla_punteras_cuando_piden_leotardo():
    # Regresion real detectada en la validacion contra la API: con ranking por
    # corte entraban punteras y spacers Capezio junto a los leotardos, porque
    # matchean 'capezio'. Solo debe quedar el tier maximo.
    union = _prods([
        "Leotardo para danza ELLA Grishko",
        "Maillot Heloisa So Danca | Leotardo para Danza con Encaje",
        "Punteras de Tela Cozy Toes Capezio Americano",
        "Super Spacers Bunheads Capezio",
    ])
    top = [p["name"] for p in rank_widened_results("leotardo negro Capezio", union)]
    assert all("Puntera" not in n and "Spacers" not in n for n in top), top
    assert len(top) == 2


def test_solo_sobrevive_el_que_matchea_mas_terminos():
    union = _prods(["Cintas Elastizadas Grishko",       # 2 terminos
                    "Elásticos Bolt Bunheads Capezio",  # 1
                    "Bolso Grishko"])                   # 1
    top = [p["name"] for p in rank_widened_results("Elásticos Grishko", union)]
    assert top == ["Cintas Elastizadas Grishko"]


# --- Resolucion de categoria (caso Luisina, 2026-10-13) --------------------

# Nombres REALES de las categorias de la tienda, en el orden en que los
# devuelve la API (las genericas vienen ANTES que las especificas: ese orden
# era justamente el que rompia el "contiene" original).
CATEGORIAS_REALES = [
    {"id": 39583323, "name": {"es": "Nuevos Ingresos"}},
    {"id": 34838994, "name": {"es": "Zapatillas de punta"}},
    {"id": 35012375, "name": {"es": "Zapatillas de media punta"}},
    {"id": 34839607, "name": {"es": "Medias"}},
    {"id": 35828958, "name": {"es": "Metatarsianas"}},
    {"id": 35012479, "name": {"es": "Accesorios"}},
    {"id": 35012482, "name": {"es": "Elásticos"}},
    {"id": 35012483, "name": {"es": "Cintas de satén y elastizadas"}},
    {"id": 35012485, "name": {"es": "Accesorios para el pie"}},
    {"id": 35012486, "name": {"es": "Punteras"}},
    {"id": 35012487, "name": {"es": "Protectores de puntas"}},
    {"id": 34839644, "name": {"es": "Bolsos"}},
]


def test_el_nombre_exacto_gana_siempre():
    assert pick_category_id(CATEGORIAS_REALES, "Accesorios para el pie") == 35012485
    assert pick_category_id(CATEGORIAS_REALES, "Accesorios") == 35012479
    assert pick_category_id(CATEGORIAS_REALES, "Bolsos") == 34839644
    assert pick_category_id(CATEGORIAS_REALES, "Punteras") == 35012486


def test_variante_del_nombre_cae_en_la_especifica_no_en_la_generica():
    # Regresion real: "Accesorios del pie" devolvia "Accesorios" (26 productos)
    # porque la generica aparece antes en la lista. Debe ganar la especifica.
    assert pick_category_id(CATEGORIAS_REALES, "Accesorios del pie") == 35012485
    assert pick_category_id(CATEGORIAS_REALES, "accesorios para el pie") == 35012485


def test_no_confunde_punteras_con_protectores_de_puntas():
    # Son categorias distintas y el diccionario del prompt las separa a proposito.
    assert pick_category_id(CATEGORIAS_REALES, "Punteras") == 35012486
    assert pick_category_id(CATEGORIAS_REALES, "Protectores de puntas") == 35012487


def test_categoria_inexistente_devuelve_none():
    # Sin match, la tool cae a busqueda de texto; nunca debe inventar una categoria.
    assert pick_category_id(CATEGORIAS_REALES, "Zapatillas de tap") is None
    assert pick_category_id(CATEGORIAS_REALES, "") is None
    assert pick_category_id(CATEGORIAS_REALES, None) is None
    assert pick_category_id([], "Bolsos") is None
    assert pick_category_id(None, "Bolsos") is None


def test_entradas_rotas_no_explotan():
    sucias = [None, "basura", {"id": 1}, {"name": None}, {"name": {"es": None}}]
    assert pick_category_id(sucias, "Bolsos") is None


# --- Alias de categoria (camino lateral del commit d27a8fb) ----------------

from catalog import canonical_category_name


def test_las_formas_de_pedir_dedos_van_a_la_categoria_real():
    # Sin esto, "Separadores de dedos" no resuelve (no es categoria real), cae a
    # busqueda de texto, da 0, y el bot repite el "no tenemos" del caso Luisina.
    for pedido in ["Separadores de dedos", "separadores", "Dedales", "dedal",
                   "Protectores de dedos", "toe spacers", "spacemakers"]:
        assert canonical_category_name(pedido) == "Accesorios para el pie", pedido


def test_no_toca_las_categorias_que_ya_resuelven():
    for pedido in ["Bolsos", "Punteras", "Protectores de puntas", "Medias",
                   "Accesorios para el pie", "Metatarsianas", "Leotardos"]:
        assert canonical_category_name(pedido) == pedido, pedido


def test_el_alias_no_se_traga_cosas_de_otra_categoria():
    # "puntas" y "protectores de puntas" NO son accesorios para el dedo.
    assert canonical_category_name("Zapatillas de punta") == "Zapatillas de punta"
    assert canonical_category_name("Protectores de puntas") == "Protectores de puntas"
    assert canonical_category_name("Zapatillas de tap") == "Zapatillas de tap"


def test_entradas_vacias_pasan_sin_romper():
    assert canonical_category_name("") == ""
    assert canonical_category_name(None) is None
    assert canonical_category_name("   ") == "   "


def test_el_alias_llega_hasta_el_id_correcto():
    # De punta a punta: nombre coloquial -> categoria real -> id.
    assert pick_category_id(CATEGORIAS_REALES,
                            canonical_category_name("Separadores de dedos")) == 35012485
    assert pick_category_id(CATEGORIAS_REALES,
                            canonical_category_name("Dedales")) == 35012485


def test_el_alias_no_le_gana_a_una_categoria_que_si_resuelve():
    # El alias se aplica solo como RESCATE. Si se aplicara antes, estas tres
    # terminarian en "Accesorios para el pie" en vez de su categoria propia.
    assert pick_category_id(CATEGORIAS_REALES, "Punteras") == 35012486
    assert canonical_category_name("Punteras para dedos") == "Accesorios para el pie"
    # pero "Punteras para dedos" SI resuelve sola, asi que el rescate no corre:
    assert pick_category_id(CATEGORIAS_REALES, "Punteras para dedos") == 35012486


def test_todas_las_formas_de_pedir_accesorios_de_dedo():
    # Las 6 formas que una clienta usa de verdad. Todas tienen que llegar a la
    # categoria: por texto dan 0 porque los productos se llaman en ingles.
    for pedido in ["dedales", "dedal", "separadores de dedos", "separadores",
                   "protectores de dedos", "dederas", "almohadillas para dedos",
                   "toe spacers", "spacemakers", "gel para dedos"]:
        assert canonical_category_name(pedido) == "Accesorios para el pie", pedido


def test_lo_que_el_texto_SI_encuentra_no_se_toca():
    # Estos se resuelven por busqueda de texto y el rescate no debe activarse:
    # son justo los que se rompian cuando esto se intento desde el prompt.
    for pedido in ["Endurecedor de puntas", "Bolsa de red", "Bolsa de malla",
                   "Punteras", "punteras de gel", "protectores de puntas",
                   "Medias", "Leotardos", "Bolsos", "Cintas", "Metatarsianas"]:
        assert canonical_category_name(pedido) == pedido, pedido


# --- Sinonimo de busqueda (caso Carolina, 2026-11-09) ----------------------

from catalog import search_term_alias


def test_las_formas_de_pedir_una_pollera_llegan_al_pollerin():
    # "pollerita" y "falda" dan CERO en Tienda Nube; el producto se llama
    # "Pollerin SO DANCA". Con vision, el modelo lee la foto y busca "Pollerita"
    # o "Falda de danza": sin esto seguiria contestando "no tenemos".
    for pedido in ["pollerita", "Pollerita", "falda", "faldita", "Falda de danza",
                   "pollera de gasa"]:
        assert search_term_alias(pedido) == "pollerin", pedido


def test_no_se_busca_a_si_mismo():
    # Si la consulta ya dice "pollerin", el rescate no tiene nada que aportar.
    assert search_term_alias("pollerin") is None
    assert search_term_alias("pollerin negro") is None


def test_el_sinonimo_no_se_traga_otras_consultas():
    for pedido in ["dedales", "punteras", "zapatillas de tap", "medias", "leotardo",
                   "bolsos", "endurecedor", "bolsa de red", ""]:
        assert search_term_alias(pedido) is None, pedido
    assert search_term_alias(None) is None


# --- Orden de la union por selectividad (caso Amanda, 2026-11-19) ----------

LEOTARDOS_REALES = [
    "Leotardo para danza ELLA Grishko",
    "Leotardo para danza EFFIE Grishko",
    "Leotardo para danza SIA Grishko",
    "Leotardo para danza SIBA Grishko",
    "Maillot Heloisa So Danca | Leotardo para Danza con Encaje",
    "Lucila SO DANCA | Leotardo con tirantes finos y encaje en espalda",
    "Laila SO DANCA | Leotardo con escote de tul y accesorios elaborados",
]
AMANDA = "Amanda SO DANCA | Malla con encaje superpuesto en el busto"


def test_el_producto_pedido_por_nombre_no_queda_afuera_del_corte():
    """Caso real: "el leotardo Amanda en talle M lo tenes?"

    El producto se llama "Amanda SO DANCA | Malla...", sin la palabra
    "leotardo", asi que la busqueda exacta da 0. En la ampliada, "leotardo"
    trae 7 productos y "amanda" trae 1: con la union en el orden de la
    consulta la Amanda quedaba 8va y el corte de 6 la dejaba afuera. El bot
    contesto "no tenemos el leotardo Amanda" con 2 unidades en M Negro y 2 en
    M Bordo, y ofrecio un Grishko de $86.000 en vez del de $73.000.

    La union se ordena por selectividad: primero los resultados del termino que
    matcheo MENOS productos, que es el que identifica lo que pidieron.
    """
    por_leotardo = _prods(LEOTARDOS_REALES)          # 7 productos, generico
    por_amanda = _prods([AMANDA])                     # 1 producto, especifico
    for p in por_amanda:
        p["id"] += 100                                # ids distintos

    # Como venia: el generico primero.
    union_vieja = por_leotardo + por_amanda
    top_viejo = [p["name"] for p in rank_widened_results("leotardo Amanda", union_vieja)]
    assert AMANDA not in top_viejo, "este test ya no reproduce el bug original"

    # Ordenado por selectividad: el especifico primero.
    union_nueva = por_amanda + por_leotardo
    top_nuevo = [p["name"] for p in rank_widened_results("leotardo Amanda", union_nueva)]
    assert top_nuevo[0] == AMANDA, f"la Amanda tiene que quedar primera, quedo: {top_nuevo[:2]}"


def test_el_orden_de_la_union_no_le_gana_al_puntaje():
    # La selectividad solo desempata. Un producto que matchea MAS terminos
    # sigue yendo primero, aunque venga de la lista mas larga.
    union = _prods(["Cintas Elastizadas Grishko"]) + _prods(["Elásticos Bolt Bunheads Capezio"])
    union[1]["id"] += 100
    top = [p["name"] for p in rank_widened_results("Elásticos Grishko", union)]
    assert top == ["Cintas Elastizadas Grishko"], top


# --- variant_availability_text: las COMBINACIONES de variantes -------------
# Caso real 2026-11-19: la Amanda tiene talle M en Negro y Bordo, pero NO en
# Rosa té. El bot veía "P, Negro, Rosa té, M, Bordo, G" y podía prometer
# un M en rosa que no existe.

def _prod(attrs, combos):
    """combos: lista de (stock, [valores...]) en el orden de attrs."""
    return {
        "attributes": [{"es": a} for a in attrs],
        "variants": [{"stock": s, "values": [{"es": x} for x in vals]}
                     for s, vals in combos],
    }


AMANDA_VARIANTES = _prod(["Talle", "Color"], [
    (1, ["P", "Negro"]), (0, ["P", "Bordo"]), (1, ["P", "Rosa té"]),
    (2, ["M", "Negro"]), (2, ["M", "Bordo"]), (0, ["M", "Rosa té"]),
    (1, ["G", "Negro"]), (0, ["G", "Bordo"]), (0, ["G", "Rosa té"]),
])


def test_amanda_agrupa_talle_con_sus_colores_reales():
    assert variant_availability_text(AMANDA_VARIANTES) == (
        "Talle P: Negro, Rosa té - Talle M: Negro, Bordo - Talle G: Negro")


def test_amanda_no_ofrece_la_combinacion_agotada():
    # Lo único que importa de verdad: M y "Rosa té" no pueden quedar juntos.
    texto = variant_availability_text(AMANDA_VARIANTES)
    tramo_m = texto.split("Talle M:")[1].split(" - ")[0]
    assert "Rosa té" not in tramo_m, tramo_m


def test_una_sola_dimension_no_cambia_nada():
    # Regresión: los productos de una dimensión tienen que seguir igual que antes.
    p = _prod(["Talle"], [(1, ["P"]), (0, ["M"]), (3, ["G"])])
    assert variant_availability_text(p) == "P, G"
    assert variant_availability_text(p) == ", ".join(
        available_variant_values(p["variants"]))


def test_agrupa_por_talle_aunque_venga_segundo_el_orden_de_atributos():
    # La LEDA es Color/Talle y la Amanda Talle/Color: hay que mirar el nombre.
    p = _prod(["Color", "Talle"], [(1, ["Bordó", "M"]), (1, ["Negro", "G"])])
    assert variant_availability_text(p) == "Talle M: Bordó - Talle G: Negro"


def test_colapsa_cuando_todos_los_talles_traen_lo_mismo():
    p = _prod(["Talle", "Color"], [
        (1, ["P", "Negro"]), (1, ["M", "Negro"]), (1, ["G", "Negro"])])
    assert variant_availability_text(p) == "Negro | Talle: P, M, G"


def test_no_colapsa_si_un_talle_tiene_menos_opciones():
    p = _prod(["Talle", "Color"], [
        (1, ["P", "Negro"]), (1, ["P", "Lila"]), (1, ["M", "Negro"])])
    assert variant_availability_text(p) == "Talle P: Negro, Lila - Talle M: Negro"


def test_demasiadas_combinaciones_avisa_que_no_todas_existen():
    # Las puntas llegan a 42 combinaciones: listarlas es ilegible.
    combos = [(1, [str(t), a, d]) for t in range(30, 40) for a in ("C", "D") for d in ("N",)]
    p = _prod(["Talle", "Anchos", "Durezas"], combos)
    texto = variant_availability_text(p)
    assert texto.endswith("(no todas las combinaciones tienen stock)")
    assert "Talle: 30, 31" in texto and "Anchos: C, D" in texto


def test_texto_agrupado_muy_largo_cae_al_formato_por_dimension():
    # Las Medias de Patón reales: 10 combinaciones de valores largos, y el talle
    # 3 en otro color, así que no se puede colapsar. Agrupado daba 334 chars.
    combos = [(1, [str(t), "Beige Claro" if t == 3 else "Beige Mediano", u])
              for t in range(0, 5) for u in ("(1 unidad)", "(6 unidades)")]
    p = _prod(["Talle", "Color", "Cantidad"], combos)
    texto = variant_availability_text(p)
    assert len(texto) < 200, "%d chars: %s" % (len(texto), texto)
    assert "no todas las combinaciones" in texto
    # y los dos colores siguen estando, no se perdio informacion
    assert "Beige Claro" in texto and "Beige Mediano" in texto


def test_sin_stock_no_dice_nada():
    p = _prod(["Talle", "Color"], [(0, ["P", "Negro"]), (0, ["M", "Negro"])])
    assert variant_availability_text(p) == ""


def test_sin_atributos_igual_agrupa():
    p = {"variants": [{"stock": 1, "values": [{"es": "M"}, {"es": "Negro"}]},
                      {"stock": 1, "values": [{"es": "M"}, {"es": "Bordo"}]}]}
    assert variant_availability_text(p) == "M: Negro, Bordo"


def test_entrada_rota_no_explota():
    for basura in (None, "texto", 42, {}, {"variants": None}, {"variants": "x"},
                   {"variants": [None, 7]}, {"variants": [{"values": None}]}):
        assert variant_availability_text(basura) == ""


def test_stock_infinito_cuenta_como_disponible():
    # stock None = sin control de stock en Tienda Nube: se puede ofrecer.
    p = _prod(["Talle", "Color"], [(None, ["P", "Negro"]), (0, ["M", "Negro"])])
    assert variant_availability_text(p) == "Talle P: Negro"


# --- productos sin opciones de talle ni color -------------------------------
# 18 de los 66 visibles: punteras, gel para dedos, endurecedor, cintas,
# cueritos. Medido con el modelo real: con el campo vacío mete la descripción
# dentro de "Variantes:", y sin el campo INVENTA variantes leyendo la
# descripción ("Variantes: Gel y tela"). Por eso hay un valor explícito.

def test_producto_sin_opciones_no_devuelve_texto_vacio():
    p = {"variants": [{"stock": 3, "values": []}]}
    assert variant_availability_text(p) == ""
    # el valor explícito lo pone simplify_product, no esta función
    assert SIN_VARIANTES and not SIN_VARIANTES.isspace()


def test_simplify_product_siempre_manda_el_campo_variants():
    # Es el contrato con el modelo: el campo va SIEMPRE y nunca vacío.
    import ast
    import io as _io
    from pathlib import Path
    src = _io.open(Path(__file__).resolve().parent.parent / "orchestrator_service" /
                   "main.py", encoding="utf-8").read()
    arbol = ast.parse(src)
    fn = next(n for n in ast.walk(arbol)
              if isinstance(n, ast.FunctionDef) and n.name == "simplify_product")
    cuerpo = ast.dump(fn)
    assert "SIN_VARIANTES" in cuerpo, "simplify_product ya no usa el valor por defecto"
    # y que no haya quedado una asignación condicional que pueda omitir la clave
    assert '"variants"' not in src.split("def simplify_product")[1].split("return salida")[0] \
        or "or SIN_VARIANTES" in src.split("def simplify_product")[1].split("return salida")[0]
