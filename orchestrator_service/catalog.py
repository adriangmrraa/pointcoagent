"""Lógica pura de disponibilidad de catálogo (sin dependencias externas).

Regla de negocio (2026-07): un producto sin stock u oculto NO se le pasa al
modelo — si el modelo nunca lo ve, nunca lo puede ofrecer. El filtrado se hace
acá, en código, y no en el prompt, porque es una regla que no puede fallar.

Convenciones de la API de Tienda Nube:
- variants[].stock == None  -> stock infinito (sin control de stock): disponible.
- variants[].stock == 0     -> agotado.
- published == False        -> producto oculto en la tienda: no mostrar.
"""


def variant_has_stock(variant) -> bool:
    if not isinstance(variant, dict):
        return False
    stock = variant.get("stock")
    if stock is None:
        return True
    try:
        return int(stock) > 0
    except (TypeError, ValueError):
        return False


def is_product_available(product) -> bool:
    """True si el producto puede ofrecerse: publicado y con al menos una variante con stock."""
    if not isinstance(product, dict):
        return False
    if product.get("published") is False:
        return False
    variants = product.get("variants") or []
    if not isinstance(variants, list) or not variants:
        # Sin variantes no hay dato de stock: no filtrar (mejor mostrar que ocultar por error).
        return True
    return any(variant_has_stock(v) for v in variants)


def available_variant_values(variants) -> list:
    """Valores de variantes (talles/colores) SOLO de variantes con stock,
    preservando el orden de aparición. Así el bot no ofrece un color agotado."""
    seen = set()
    ordered = []
    if not isinstance(variants, list):
        return ordered
    for v in variants:
        if not isinstance(v, dict) or not variant_has_stock(v):
            continue
        for val in v.get("values") or []:
            if isinstance(val, dict):
                s = val.get("es") or val.get("en")
                if s and s not in seen:
                    seen.add(s)
                    ordered.append(s)
    return ordered
