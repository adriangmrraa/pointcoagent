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
