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


# --- FUGA DE INTERNALS (2026-08) -------------------------------------------
# Caso real: una clienta preguntó por puntas para su hija y recibió por WhatsApp
# el JSON crudo {"tool_uses":[{"recipient_name":"functions.derivhumano",...}]}.
# El modelo, forzado a responder JSON (response_format=json_object), a veces
# ESCRIBE la invocación de la tool como texto en vez de emitirla como tool_call.
# Ese JSON caía en el "last resort: stringify" del parser y salía tal cual.
# Estas funciones lo detectan para que nunca llegue al cliente.

_LEAK_KEYS = {
    "tool_uses",
    "tool_calls",
    "multi_tool_use",
    "recipient_name",
    "function_call",
    "functions",
    "tool_name",
    "action_input",
    "agent_scratchpad",
}

_LEAK_TEXT_RE = re.compile(
    r'("(tool_uses|tool_calls|recipient_name|function_call|action_input)"\s*:)'
    r'|(functions\.[a-z_]\w*)'
    r'|(multi_tool_use\.parallel)',
    re.IGNORECASE,
)

TOOL_LEAK_FALLBACK_TEXT = (
    "Uy, se me trabó el sistema justo con tu mensaje y no quiero contestarte cualquier cosa. "
    "Me lo repetís y lo vemos?"
)

TOOL_LEAK_RETRY_HINT = (
    "SISTEMA: tu último output fue una invocación de herramienta ESCRITA COMO TEXTO. "
    "Eso es un error: si necesitás una herramienta, invocala de verdad (tool call). "
    'El contenido que devolvés SIEMPRE tiene que ser el JSON {"messages": [...]} '
    "dirigido a la clienta, nunca la llamada a la herramienta. "
    "Si en este turno YA ejecutaste una herramienta de verdad, NO la repitas: "
    "escribí directamente el mensaje para la clienta con los datos que ya tenés."
)


def _walk_keys(obj, depth=0):
    """Todas las claves de un objeto anidado (profundidad acotada)."""
    if depth > 6:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _walk_keys(v, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for item in obj[:50]:
            yield from _walk_keys(item, depth + 1)


def detect_internal_leak(output):
    """Devuelve el motivo si `output` es una llamada a tool y no una respuesta.

    Acepta el output crudo del agente: str, dict o lista. None = está limpio.
    """
    if output is None:
        return None

    # 1. Estructura: claves propias del protocolo de tools.
    if isinstance(output, (dict, list)):
        for key in _walk_keys(output):
            if key.lower() in _LEAK_KEYS:
                return f"clave interna en el output: {key}"
        try:
            raw = json.dumps(output, ensure_ascii=False, default=str)
        except Exception:
            raw = str(output)
    else:
        raw = str(output)

    # 2. Texto: el JSON pudo no parsear (truncado) y seguir siendo una tool call.
    match = _LEAK_TEXT_RE.search(raw)
    if match:
        return f"patrón de invocación de tool en el texto: {match.group(0)[:60]}"

    return None
