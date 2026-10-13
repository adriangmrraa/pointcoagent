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


# Formas de pedir una categoria que la busqueda de texto NO puede resolver, y a
# que categoria REAL corresponden. Existe porque los productos para el dedo estan
# nombrados en ingles (Spacemakers, Jelly Toes, Pinky Pads, Super Spacers): si el
# modelo manda "Separadores de dedos", que no es una categoria real, hoy se cae a
# busqueda de texto, da 0, y el bot vuelve a decir "no tenemos" teniendo stock.
_ALIAS_CATEGORIA = (
    (("separador", "dedal", "dedo", "spacer", "spacemaker", "toe"), "Accesorios para el pie"),
)


def canonical_category_name(name):
    """Lleva una forma coloquial de nombrar una categoria al nombre REAL.

    Si no reconoce nada, devuelve el nombre tal cual: nunca inventa categorias.
    """
    if not name:
        return name
    terminos = query_terms(name)
    if not terminos:
        return name
    for prefijos, real in _ALIAS_CATEGORIA:
        if any(t.startswith(prefijos) for t in terminos):
            return real
    return name


def pick_category_id(categories, name):
    """Resuelve el id de una categoría de Tienda Nube por su nombre (es).

    Busca coincidencia exacta primero; si no, coincidencia parcial (contiene).
    Devuelve el id (int) o None. Útil para buscar por categoría real en vez de
    por texto (ej: 'Bolsos' trae bolsos Y mochilas, que por nombre no matchean).
    """
    if not name or not isinstance(categories, list):
        return None
    target = str(name).strip().lower()
    if not target:
        return None
    # 1. Coincidencia exacta
    for c in categories:
        if not isinstance(c, dict):
            continue
        cn = ((c.get("name") or {}).get("es") or "").strip().lower()
        if cn and cn == target:
            return c.get("id")
    # 2. Por solapamiento de palabras: gana la categoria que comparte MAS
    #    terminos con lo pedido.
    #
    #    Antes esto era un "contiene" que devolvia la PRIMERA coincidencia de la
    #    lista, y la lista trae las genericas antes que las especificas: pedir
    #    "Accesorios del pie" caia en "Accesorios" (26 productos) en vez de en
    #    "Accesorios para el pie" (6, los que de verdad buscaban). Comparar por
    #    palabras hace que gane la especifica sin depender del orden.
    pedidos = set(query_terms(target))
    mejor_id, mejor_score, mejor_largo = None, 0, 0
    for c in categories:
        if not isinstance(c, dict):
            continue
        cn = ((c.get("name") or {}).get("es") or "").strip().lower()
        if not cn:
            continue
        propios = set(query_terms(cn))
        score = len(pedidos & propios)
        # La categoria tiene que cubrir TODOS los terminos pedidos, no apenas
        # uno. Con solapamiento parcial, "Zapatillas de tap" caia en "Zapatillas
        # de punta" por compartir "zapatillas" -> ofrecerle puntas de $230.000 a
        # alguien que pidio zapatillas de tap. Mejor devolver None y que la tool
        # caiga a busqueda de texto.
        if score == 0 or score < len(pedidos):
            continue
        # Mas terminos en comun gana. A igualdad, el nombre mas corto: es el
        # match menos diluido (ej. "Punteras" sobre "Protectores de puntas").
        if score > mejor_score or (score == mejor_score and len(cn) < mejor_largo):
            mejor_id, mejor_score, mejor_largo = c.get("id"), score, len(cn)
    if mejor_id is not None:
        return mejor_id

    # 3. Ultima red: el "contiene" original, por si los terminos no cruzan.
    for c in categories:
        if not isinstance(c, dict):
            continue
        cn = ((c.get("name") or {}).get("es") or "").strip().lower()
        if cn and (target in cn or cn in target):
            return c.get("id")
    return None


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


# --- BUSQUEDA AMPLIADA (2026-08) -------------------------------------------
# El buscador de Tienda Nube exige que TODOS los terminos aparezcan en el NOMBRE
# del producto. Caso real 2026-07-23: una clienta pidio "elasticos Grishko", el
# modelo busco q="Elasticos Grishko" y dio 0 porque el producto se llama "Cintas
# Elastizadas Grishko" (tiene Grishko, no tiene Elasticos). El bot contesto "no
# tenemos" y la venta la cerro una humana a mano 1 hora despues.
#
# Cuando la busqueda exacta vuelve vacia se reintenta por termino suelto y se
# unen los resultados. El problema es el ORDEN: "Grishko" solo devuelve 19
# productos y el que ella queria esta en el puesto 13, fuera de lo que el modelo
# mira. Por eso se re-ordena por cuantos terminos de la consulta original
# aparecen en el nombre, comparando por PREFIJO: asi "elast" matchea tanto
# "Elasticos" como "Elastizadas", y "Cintas Elastizadas Grishko" -que matchea
# los dos terminos- queda primero.

import unicodedata

_STOPWORDS = {"de", "del", "la", "el", "los", "las", "para", "con", "sin",
              "y", "o", "un", "una", "por", "en", "al"}

_LARGO_PREFIJO = 5
_LARGO_MINIMO = 3


def sin_acentos(texto) -> str:
    """minusculas y sin tildes: 'Elásticos' -> 'elasticos'."""
    if not texto:
        return ""
    desarmado = unicodedata.normalize("NFD", str(texto).lower())
    return "".join(c for c in desarmado if unicodedata.category(c) != "Mn")


def query_terms(query) -> list:
    """Terminos utiles de la consulta, sin stopwords ni duplicados."""
    limpio = "".join(c if c.isalnum() else " " for c in sin_acentos(query))
    terminos = []
    for t in limpio.split():
        if len(t) >= _LARGO_MINIMO and t not in _STOPWORDS and t not in terminos:
            terminos.append(t)
    return terminos


def _prefijo(termino) -> str:
    return termino[:_LARGO_PREFIJO]


def name_match_score(nombre, terminos) -> int:
    """Cuantos terminos de la consulta aparecen en el nombre (por prefijo)."""
    objetivo = sin_acentos(nombre)
    palabras = "".join(c if c.isalnum() else " " for c in objetivo).split()
    score = 0
    for t in terminos:
        p = _prefijo(t)
        if any(w.startswith(p) for w in palabras):
            score += 1
    return score


# Marcas reales de la tienda (categorias bajo "Marcas" en Tienda Nube, 2026-08).
# Una marca SOLA no alcanza para que un producto entre: si alguien pide
# "leotardo negro Capezio", una puntera Capezio matchea la marca pero no es lo
# que pidio. Si se suma una marca nueva y no se agrega aca, lo unico que pasa es
# que el filtro afloja para esa marca: no se rompe nada.
MARCAS_TIENDA = {"grishko", "capezio", "sansha", "danca", "bunheads",
                 "pointe", "coach"}


def rank_widened_results(query, productos, limite=6, marcas=None) -> list:
    """Se queda SOLO con los que matchean la mayor cantidad de terminos.

    Precision antes que cantidad, a proposito. Un corte por ranking devolvia
    basura: con "leotardo negro Capezio" entraban punteras y spacers Capezio
    (matchean 'capezio') mezclados con los leotardos. Mostrarle almohadillas a
    quien pidio una malla es peor que no mostrarle nada.

    Quedandose con el tier maximo, "Elasticos Grishko" devuelve exactamente
    "Cintas Elastizadas Grishko" (unico que matchea los dos terminos), y
    "leotardo negro Capezio" devuelve solo leotardos. Los empates conservan el
    orden en que los devolvio la API (sort estable).
    """
    terminos = query_terms(query)
    if not terminos or not isinstance(productos, list):
        return []

    prefijos_marca = {_prefijo(m) for m in (MARCAS_TIENDA if marcas is None else marcas)}
    sin_marca = [t for t in terminos if _prefijo(t) not in prefijos_marca]
    # Si la consulta es SOLO marcas ("algo de Grishko"), no hay con que exigir mas.
    exigir_tipo = bool(sin_marca)

    puntuados = []
    for p in productos:
        if not isinstance(p, dict):
            continue
        nombre = p.get("name") or ""
        score = name_match_score(nombre, terminos)
        if score == 0:
            continue
        if exigir_tipo and name_match_score(nombre, sin_marca) == 0:
            continue  # matchea solo la marca: no es lo que pidieron
        puntuados.append((score, p))
    if not puntuados:
        return []
    mejor = max(score for score, _ in puntuados)
    return [p for score, p in puntuados if score == mejor][:limite]
