"""Tests del filtro de disponibilidad de catálogo (orchestrator_service/catalog.py).

Son funciones puras: corren sin Redis, sin Postgres y sin API keys.
    python -m pytest tests/test_catalog.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "orchestrator_service"))

from catalog import variant_has_stock, is_product_available, available_variant_values, pick_category_id


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
