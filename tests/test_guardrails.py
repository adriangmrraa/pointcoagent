"""Tests del guardrail pre-envío (orchestrator_service/guardrails.py).

Funciones puras: corren sin Redis, sin Postgres y sin API keys.
    python -m pytest tests/test_guardrails.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "orchestrator_service"))

from guardrails import normalize_url, extract_urls, collect_urls, filter_outbound_messages

WEB = "https://www.pointecoach.shop/"
TOOL_URL = "https://www.pointecoach.shop/productos/medias-convertibles-so-danca/"
TOOL_IMG = "https://dcdn-us.mitiendanube.com/stores/006/873/259/products/foto-1024-1024.jpg"
HALLUCINATED = "https://www.pointecoach.shop/productos/zapatillas-de-puntas-grishko-3007-pro/"


def msg(text=None, image=None):
    return {"part": 1, "total": 1, "text": text, "imageUrl": image}


# --- normalize_url / extract_urls ---

def test_normaliza_esquema_www_y_barra():
    assert normalize_url("https://www.Pointecoach.shop/productos/x/") == "pointecoach.shop/productos/x"
    assert normalize_url("http://pointecoach.shop/productos/x") == "pointecoach.shop/productos/x"

def test_normaliza_query_y_fragment():
    assert normalize_url(TOOL_URL + "?utm=abc#top") == normalize_url(TOOL_URL)

def test_extrae_urls_con_puntuacion_pegada():
    urls = extract_urls(f"Mira {TOOL_URL}. Te gusta?")
    assert urls == [TOOL_URL]

def test_collect_urls_de_estructuras():
    data = [{"url": TOOL_URL, "imageUrl": TOOL_IMG, "nested": {"x": f"ver {WEB}"}}]
    found = collect_urls(data)
    assert TOOL_URL in found and TOOL_IMG in found


# --- filter_outbound_messages: el caso real del log ---

def test_caso_real_burbuja_alucinada_se_descarta():
    """Consulta por medias respondida con puntas que la tool no devolvió (log 2026-01)."""
    allowed = {TOOL_URL, TOOL_IMG}
    messages = [
        msg("Mirá estas opciones:"),
        msg(f"Medias Convertibles\nPrecio: $29000\n{TOOL_URL}", TOOL_IMG),
        msg(f"Zapatillas de Puntas Grishko 3007 PRO\nPrecio: $210000\n{HALLUCINATED}"),
        msg(f"Si querés ver más, entrá a nuestra web: {WEB}"),
    ]
    kept, blocked = filter_outbound_messages(messages, allowed, WEB)
    texts = " ".join(m["text"] for m in kept)
    assert len(kept) == 3
    assert "3007 PRO" not in texts
    assert any("3007" in b or "grishko" in b.lower() for b in blocked)

def test_home_de_la_tienda_siempre_permitida():
    kept, blocked = filter_outbound_messages([msg(f"Entrá a {WEB}")], set(), WEB)
    assert len(kept) == 1 and not blocked

def test_home_no_habilita_todo_el_dominio():
    """Una URL alucinada del propio dominio NO pasa aunque la home esté permitida."""
    kept, blocked = filter_outbound_messages([msg(f"Mirá: {HALLUCINATED}")], set(), WEB)
    assert kept == [] and len(blocked) == 1

def test_imagen_no_verificada_se_quita_pero_el_texto_queda():
    kept, blocked = filter_outbound_messages(
        [msg("Hola! En qué te ayudo?", "https://evil.example.com/img.jpg")], set(), WEB
    )
    assert len(kept) == 1
    assert kept[0]["imageUrl"] is None
    assert len(blocked) == 1

def test_texto_sin_urls_pasa_intacto():
    kept, blocked = filter_outbound_messages(
        [msg("Hola! Como estas? Soy del equipo de Pointe Coach.")], set(), WEB
    )
    assert len(kept) == 1 and not blocked

def test_precio_none_se_reemplaza():
    kept, _ = filter_outbound_messages([msg("Punteras Ouch Pouch\nPrecio: $None\nBuenisimas")], set(), WEB)
    assert "Precio: a consultar" in kept[0]["text"]
    assert "None" not in kept[0]["text"]

def test_url_de_tool_con_variacion_de_esquema_pasa():
    kept, blocked = filter_outbound_messages(
        [msg(f"Ficha: {TOOL_URL.replace('https://www.', 'http://')}")], {TOOL_URL}, WEB
    )
    assert len(kept) == 1 and not blocked

def test_burbuja_vacia_tras_limpieza_se_elimina():
    kept, _ = filter_outbound_messages([msg(None, "https://evil.example.com/x.jpg")], set(), WEB)
    assert kept == []


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


# --- Fuga de internals (caso real 2026-08) ---------------------------------

import json

from guardrails import detect_internal_leak, TOOL_LEAK_FALLBACK_TEXT

# Payload textual EXACTO que recibio una clienta por WhatsApp.
FUGA_REAL = json.dumps({
    "tool_uses": [{
        "recipient_name": "functions.derivhumano",
        "parameters": {
            "reason": "Clienta quiere visitar el local / retirar / reservar",
            "contact_name": "Lorena Almiron",
            "summary": "Quiere comprar puntas para su hija Lourdes, calza 40.",
        },
    }]
}, ensure_ascii=False)


def test_detecta_la_fuga_real_de_whatsapp():
    assert detect_internal_leak(FUGA_REAL) is not None


def test_detecta_la_fuga_ya_parseada_como_dict():
    assert detect_internal_leak(json.loads(FUGA_REAL)) is not None


def test_detecta_tool_call_truncado_sin_json_valido():
    truncado = '{"tool_uses": [{"recipient_name": "functions.search_specific_produ'
    assert detect_internal_leak(truncado) is not None


def test_detecta_formato_tool_calls_y_function_call():
    assert detect_internal_leak({"tool_calls": [{"name": "orders"}]}) is not None
    assert detect_internal_leak({"function_call": {"name": "cupones_list"}}) is not None
    assert detect_internal_leak('{"action": "orders", "action_input": {"id": "1"}}') is not None


def test_respuesta_normal_no_es_fuga():
    ok = json.dumps({"messages": [
        {"text": "Hola! Como estas? Te muestro las puntas que tenemos:", "imageUrl": None},
        {"text": "Mochilas Pointe Coach\nPrecio: $40000\nVariantes: Gris, Negro",
         "imageUrl": "https://dcdn-us.mitiendanube.com/stores/006/mochila.jpg"},
    ]}, ensure_ascii=False)
    assert detect_internal_leak(ok) is None


def test_texto_de_clienta_con_palabras_parecidas_no_es_fuga():
    # Falsos positivos: la clienta puede escribir cualquier cosa.
    assert detect_internal_leak("Necesito puntas, mi hija calza 40. Me pasas los parameters?") is None
    assert detect_internal_leak({"messages": [{"text": "Que funciones tiene la mochila?"}]}) is None
    assert detect_internal_leak(None) is None


def test_fallback_de_fuga_no_expone_nada_tecnico():
    bajo = TOOL_LEAK_FALLBACK_TEXT.lower()
    for palabra in ("json", "tool", "error", "function", "sistema interno"):
        assert palabra not in bajo


def test_retry_hint_pide_no_repetir_tools_ya_ejecutadas():
    # Sin esto, el reintento por fuga puede hacer que derivhumano mande
    # un segundo mail al equipo. La guarda DURA vive en turn_state.py y se
    # testea en test_turn_state.py; esta es solo la capa blanda. (El primer
    # intento de guarda dura usaba un ContextVar reasignado y era inerte: este
    # test pasaba igual, porque solo mira un string.)
    from guardrails import TOOL_LEAK_RETRY_HINT
    bajo = TOOL_LEAK_RETRY_HINT.lower()
    assert "no la repitas" in bajo
    assert '{"messages": [...]}' in TOOL_LEAK_RETRY_HINT


def test_el_fallback_de_fuga_sobrevive_al_guardrail_de_urls():
    # Si el guardrail filtrara el propio fallback, la clienta no recibiria NADA.
    from guardrails import TOOL_LEAK_FALLBACK_TEXT
    msgs = [{"text": TOOL_LEAK_FALLBACK_TEXT, "imageUrl": None}]
    kept, blocked = filter_outbound_messages(msgs, set(), "https://www.pointecoach.shop")
    assert len(kept) == 1
    assert kept[0]["text"] == TOOL_LEAK_FALLBACK_TEXT
    assert blocked == []


# --- detect_fit_advice: consejos de calce inventados ------------------------
# El bot no sabe cómo le va a calzar una zapatilla: depende del pie. Si opina
# igual, la clienta compra el talle equivocado y termina en un cambio o en un
# viaje al local al vacío; para puntas además contradice el fitting. La
# prohibición está en el prompt, pero medido con el modelo real no alcanza: ante
# un pedido directo le hace caso a la clienta 8 de 8 veces.

from guardrails import detect_fit_advice  # noqa: E402


def bot(texto):
    return {"messages": [{"text": texto, "imageUrl": None}]}


CONSEJOS = [
    "Mirá, para las zapatillas de punta las hormas suelen ser chicas, por eso te conviene pedir un número más.",
    # Conversión de talles con el número suelto, sin la palabra "talle" detrás.
    # Caso real medido: se escapaba del guardia y el banco sí lo marcaba.
    "En talle 38 (equivale a 5 1/2 o 6 según numeración) hay algunas combinaciones.",
    "El 38 equivale a 5 1/2 en estas puntas.",
    "Si calzás 38, sería un talle 5 en Grishko.",
    "Tené en cuenta que las hormas son chicas y se recomienda elegir un número más del calce de calle.",
    "Si calzás 38 de calle, en puntas equivale al talle 5.",
    "Te va a quedar perfecta esa media punta.",
    "Las Grishko suelen calzar más chico que las Só Danca.",
    "Para un 38 de calle generalmente se recomienda pedir un talle más.",
    "Esa horma es angosta, te quedaría justa.",
]

LEGITIMOS = [
    "Las Grishko 3007 PRO vienen en talles 3 1/2 al 7 1/2. Decime cuál usás?",
    "Tenemos talle 38 disponible en rosa. Te paso el link.",
    "En M no me queda rosa té, pero lo tengo en negro y en bordo.",
    "Qué talle necesitás? Así te confirmo si lo tengo.",
    "Te derivamos con una asesora (FITTER), que está capacitada para que encuentres "
    "la mejor punta que se adecue a TU PIE, en breve se contacta con vos.",
    "Precio: $259000 Variantes: Talle: 3 1/2, 4, 4 1/2 | Anchos: XXX",
    "El costo y tiempo de envío se calculan al final de la compra según tu ubicación.",
    "Mirá estas opciones de leotardos que tenemos!",
    "Los talles disponibles son 4, 5 y 6.",
    "En talle 38 tenemos en ancho C y D.",
    "Ese producto no lo manejamos por acá, pero te puedo mostrar media puntas si querés!",
]


def test_detecta_cada_consejo_de_calce():
    for texto in CONSEJOS:
        assert detect_fit_advice(bot(texto)), texto


def test_no_dispara_con_mensajes_legitimos():
    for texto in LEGITIMOS:
        assert detect_fit_advice(bot(texto)) is None, texto


def test_solo_mira_el_texto_que_lee_la_clienta():
    # Si la DESCRIPCIÓN del catálogo habla de horma, eso lo escribió la tienda.
    # Medido: 0 de los 66 productos visibles tiene una descripción así, pero el
    # guardia no debe mirar ahí igual.
    salida = {"messages": [{"text": "Mirá, tenemos estas opciones. Precio: $259000"}],
              "description": "Horma angosta, suele calzar más chico"}
    assert detect_fit_advice(salida) is None


def test_entrada_rota_no_explota():
    for basura in (None, "", {}, {"messages": None}, {"messages": [None]},
                   {"messages": [{"text": None}]}, [], 42):
        detect_fit_advice(basura)      # no debe tirar excepción


def test_detecta_en_el_segundo_mensaje_de_la_secuencia():
    salida = {"messages": [{"text": "Hola! Mirá lo que encontré."},
                           {"text": "Para puntas te conviene pedir un número más."}]}
    assert detect_fit_advice(salida)
