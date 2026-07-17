"""Planificador puro de la secuencia de envío a WhatsApp.

Convierte las burbujas del orquestador ({text, imageUrl}) en la lista MÍNIMA
de mensajes de WhatsApp. Regla principal (2026-07, Meta cobra por mensaje):
si una burbuja tiene imagen y texto, se envía UN solo mensaje de imagen con
caption en vez de dos mensajes separados. También fusiona el patrón viejo del
prompt (burbuja solo-imagen seguida de su burbuja de texto).

Límite real de caption en WhatsApp: 1024 caracteres (usamos 1000 de margen).
Funciones puras, sin dependencias: testeables con tests/test_sequence_planner.py.
"""
import re

MAX_CAPTION = 1000
MAX_TEXT = 400


def split_text(text, max_len=MAX_TEXT):
    """Divide un texto largo por oraciones en bloques de hasta max_len (misma
    lógica del splitter de emergencia original)."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_len:
        return [text]
    sentences = re.split(r"(?<=[.!?]) +", text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) < max_len:
            current += (" " + s if current else s)
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    return chunks


def _merge_orphan_images(messages):
    """Fusiona el patrón viejo: burbuja SOLO-imagen seguida de burbuja SOLO-texto
    (la ficha del producto) -> una sola burbuja con ambos."""
    merged, i = [], 0
    while i < len(messages):
        m = dict(messages[i])
        nxt = messages[i + 1] if i + 1 < len(messages) else None
        if (
            m.get("imageUrl")
            and not (m.get("text") or "").strip()
            and nxt is not None
            and (nxt.get("text") or "").strip()
            and not nxt.get("imageUrl")
        ):
            m["text"] = nxt["text"]
            merged.append(m)
            i += 2
        else:
            merged.append(m)
            i += 1
    return merged


def plan_send_actions(messages, max_caption=MAX_CAPTION):
    """Devuelve la lista de envíos a ejecutar, en orden:
    {"type": "image", "url": ..., "caption": ...|None} o {"type": "text", "text": ...}
    """
    actions = []
    for m in _merge_orphan_images(messages or []):
        image = m.get("imageUrl")
        text = (m.get("text") or "").strip()
        if image and text and len(text) <= max_caption:
            actions.append({"type": "image", "url": image, "caption": text})
        elif image:
            # Caption imposible (texto demasiado largo o inexistente):
            # imagen sola + texto aparte (comportamiento clásico).
            actions.append({"type": "image", "url": image, "caption": None})
            for part in split_text(text):
                actions.append({"type": "text", "text": part})
        else:
            for part in split_text(text):
                actions.append({"type": "text", "text": part})
    return actions
