"""Filtro determinístico de salida (guardrail pre-envío).

Regla de negocio (2026-07): el bot solo puede enviar links e imágenes que las
tools devolvieron EN ESTE MISMO TURNO (más la home de la tienda). Una burbuja
con un link no verificado se descarta entera: si el link no salió de una tool
de este turno, la ficha de producto que lo acompaña tampoco es confiable.

Esto NO depende del prompt: es la garantía de última línea contra respuestas
con productos reciclados de turnos anteriores o inventados (caso real del
2026-01: consulta por medias respondida con zapatillas de punta de $210.000
que la tool nunca devolvió en ese turno).

Funciones puras, sin dependencias: testeables con tests/test_guardrails.py.
"""
import json
import re

_URL_RE = re.compile(r'https?://[^\s\)\]\}"\'<>]+', re.IGNORECASE)
_PRICE_NONE_RE = re.compile(r'(?i)precio:\s*\$?\s*(none|null)\b')

FALLBACK_TEXT = (
    "Uy, me mareé con ese último detalle y prefiero no pasarte info que no sea exacta. "
    "Contame de nuevo qué estás buscando y te muestro las opciones reales que tenemos!"
)


def normalize_url(url) -> str:
    """Normaliza para comparar: sin esquema, sin www, sin query/fragment, sin barra final."""
    if not url:
        return ""
    u = str(url).strip().lower()
    u = re.sub(r"^https?://", "", u)
    if u.startswith("www."):
        u = u[4:]
    u = u.split("?")[0].split("#")[0]
    return u.rstrip("/").rstrip(".,;:!")


def extract_urls(text) -> list:
    if not text:
        return []
    return [m.rstrip(".,;:!?") for m in _URL_RE.findall(str(text))]


def collect_urls(obj) -> set:
    """Toda URL presente en el resultado de una tool (estructura arbitraria)."""
    if obj is None:
        return set()
    if isinstance(obj, str):
        raw = obj
    else:
        try:
            raw = json.dumps(obj, ensure_ascii=False, default=str)
        except Exception:
            raw = str(obj)
    return set(extract_urls(raw))


def filter_outbound_messages(messages, allowed_urls, store_website=""):
    """Filtra la respuesta final antes de enviarla.

    messages: lista de dicts {text, imageUrl, ...}.
    Devuelve (mensajes_permitidos, motivos_de_bloqueo).

    - imageUrl no verificada -> se quita la imagen (la burbuja de texto sigue).
    - link no verificado en el texto -> se descarta la burbuja completa.
    - "Precio: $None"/null -> se reemplaza por "Precio: a consultar".
    """
    allowed = {normalize_url(u) for u in (allowed_urls or set()) if u}
    if store_website:
        allowed.add(normalize_url(store_website))
    allowed.discard("")

    kept, blocked = [], []
    for m in messages:
        msg = dict(m)
        text = msg.get("text") or ""
        image = msg.get("imageUrl")

        if image and normalize_url(image) not in allowed:
            blocked.append(f"imagen no verificada: {image}")
            msg["imageUrl"] = None

        bad = [u for u in extract_urls(text) if normalize_url(u) not in allowed]
        if bad:
            blocked.append(f"burbuja descartada por link no verificado: {bad[0]}")
            continue

        if text:
            msg["text"] = _PRICE_NONE_RE.sub("Precio: a consultar", text)

        if msg.get("text") or msg.get("imageUrl"):
            kept.append(msg)

    return kept, blocked
