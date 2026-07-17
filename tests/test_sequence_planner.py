"""Tests del planificador de secuencia de envío (whatsapp_service/sequence_planner.py).

Funciones puras: corren sin Redis ni APIs.
    python -m pytest tests/test_sequence_planner.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "whatsapp_service"))

from sequence_planner import plan_send_actions, split_text

IMG1 = "https://cdn.example.com/producto1.jpg"
IMG2 = "https://cdn.example.com/producto2.jpg"
FICHA = "Bolso Casey Carry-All\nPrecio: $114000\nVariantes: Negro\nhttps://tienda.com/p/1"


def test_formato_actual_burbuja_con_texto_e_imagen_es_un_solo_mensaje():
    """El agente hoy emite {text, imageUrl} juntos: debe salir 1 mensaje, no 2."""
    msgs = [
        {"text": "Hola! Mirá estas opciones:", "imageUrl": None},
        {"text": FICHA, "imageUrl": IMG1},
        {"text": FICHA, "imageUrl": IMG2},
        {"text": "Te puedo ayudar con algo más?", "imageUrl": None},
    ]
    actions = plan_send_actions(msgs)
    assert len(actions) == 4  # antes eran 6 envíos
    assert actions[1] == {"type": "image", "url": IMG1, "caption": FICHA}
    assert actions[2]["type"] == "image" and actions[2]["caption"] == FICHA


def test_formato_viejo_imagen_sola_mas_ficha_se_fusiona():
    """Patrón de 8 burbujas del prompt: solo-imagen seguida de solo-texto -> caption."""
    msgs = [
        {"text": "Intro", "imageUrl": None},
        {"text": None, "imageUrl": IMG1},
        {"text": FICHA, "imageUrl": None},
        {"text": None, "imageUrl": IMG2},
        {"text": FICHA, "imageUrl": None},
        {"text": "CTA final", "imageUrl": None},
    ]
    actions = plan_send_actions(msgs)
    assert len(actions) == 4  # antes eran 6 envíos
    assert actions[1] == {"type": "image", "url": IMG1, "caption": FICHA}
    assert actions[2] == {"type": "image", "url": IMG2, "caption": FICHA}


def test_caption_demasiado_largo_cae_al_modo_clasico():
    largo = "x" * 1500
    actions = plan_send_actions([{"text": largo, "imageUrl": IMG1}])
    assert actions[0] == {"type": "image", "url": IMG1, "caption": None}
    assert all(a["type"] == "text" for a in actions[1:])
    assert "".join(a["text"] for a in actions[1:]).replace(" ", "") == largo


def test_texto_largo_se_divide_por_oraciones():
    texto = ("Primera oración del mensaje. " * 25) + "Final!"
    partes = split_text(texto, max_len=400)
    assert len(partes) > 1
    assert all(len(p) <= 400 for p in partes)


def test_imagen_sola_sin_texto_siguiente_queda_sola():
    actions = plan_send_actions([{"text": None, "imageUrl": IMG1}])
    assert actions == [{"type": "image", "url": IMG1, "caption": None}]


def test_dos_imagenes_seguidas_no_se_fusionan_mal():
    msgs = [
        {"text": None, "imageUrl": IMG1},
        {"text": None, "imageUrl": IMG2},
        {"text": "ficha", "imageUrl": None},
    ]
    actions = plan_send_actions(msgs)
    assert actions[0] == {"type": "image", "url": IMG1, "caption": None}
    assert actions[1] == {"type": "image", "url": IMG2, "caption": "ficha"}


def test_lista_vacia_y_textos_vacios():
    assert plan_send_actions([]) == []
    assert plan_send_actions([{"text": "   ", "imageUrl": None}]) == []


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
