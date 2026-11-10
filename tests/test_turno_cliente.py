"""El turno de la clienta tal como lo ve el modelo: texto + sus fotos.

Antes la foto llegaba como un mensaje VACIO y el bot ni sabia que existia, asi
que adivinaba. Caso real 2026-11-09: una clienta reenvio la ficha de un pollerin
preguntando si tenia elastico en la cintura, y el bot le ofrecio elasticos para
zapatillas de punta y despues un leotardo de $86.000.

Estos tests necesitan las dependencias del servicio: corren con el venv
(.venv\\Scripts\\python.exe -m pytest), no con el python del sistema.
"""
import os
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "orchestrator_service"))

# main.py lee estas al importar; valores de relleno para no depender del .env
for clave, valor in [("OPENAI_API_KEY", "x"), ("REDIS_URL", "redis://localhost:6379"),
                     ("TIENDANUBE_STORE_ID", "0"), ("TIENDANUBE_ACCESS_TOKEN", "x")]:
    os.environ.setdefault(clave, valor)

main = pytest.importorskip("main", reason="requiere las dependencias del servicio (usar el venv)")

FOTO = "https://ejemplo.com/pollerin.jpg"
OTRA = "https://ejemplo.com/otra.jpg"


def _contenido(texto, imagenes):
    msgs = main.construir_turno_cliente(texto, imagenes)
    assert len(msgs) == 1
    return msgs[0].content


def test_solo_texto_sigue_siendo_texto_plano():
    # El 95% de los turnos. No tiene que cambiar de forma ni encarecerse.
    assert _contenido("tenes medias?", []) == "tenes medias?"
    assert _contenido("tenes medias?", None) == "tenes medias?"


def test_texto_mas_foto_viaja_como_dos_bloques():
    c = _contenido("Este polleria tiene elastico ?", [FOTO])
    assert isinstance(c, list) and len(c) == 2
    assert c[0] == {"type": "text", "text": "Este polleria tiene elastico ?"}
    assert c[1]["type"] == "image_url"
    assert c[1]["image_url"]["url"] == FOTO


def test_solo_foto_viaja_igual_sin_texto_vacio():
    # Mandar un bloque de texto vacio confunde al modelo.
    c = _contenido(None, [FOTO])
    assert isinstance(c, list) and len(c) == 1
    assert c[0]["type"] == "image_url"


def test_el_adjunto_que_no_se_puede_ver_avisa_en_vez_de_callar():
    # Un PDF o un archivo raro: antes llegaba un mensaje vacio y el bot adivinaba.
    c = _contenido(None, [])
    assert isinstance(c, str)
    assert c == main.ADJUNTO_NO_LEIBLE
    assert "no podés ver" in c.lower() or "no podes ver" in c.lower()


def test_el_aviso_de_adjunto_le_prohibe_adivinar():
    bajo = main.ADJUNTO_NO_LEIBLE.lower()
    assert "adivines" in bajo, "sin esto el modelo inventa de qué se trata"


def test_se_corta_la_cantidad_de_fotos():
    # Mas fotos no aportan y cada una suma ~2.400 tokens.
    c = _contenido("miralas", [FOTO, OTRA, "https://ejemplo.com/3.jpg", "https://ejemplo.com/4.jpg"])
    imagenes = [b for b in c if b["type"] == "image_url"]
    assert len(imagenes) == main.IMAGENES_POR_TURNO


def test_entradas_raras_no_explotan():
    assert isinstance(_contenido("", []), str)
    assert isinstance(_contenido(None, None), str)
    assert isinstance(_contenido("hola", []), str)


def test_la_foto_llega_al_prompt_del_agente():
    """La pieza que fallaba en silencio.

    Con el prompt viejo -("user", "{input}"), un template de STRING- una lista
    texto+imagen se convertia a str SIN dar error: la foto no llegaba nunca al
    modelo y nadie se enteraba. Este test fija que se preserve.
    """
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import SystemMessage

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="sistema"),
        MessagesPlaceholder(variable_name="chat_history"),
        MessagesPlaceholder(variable_name="input"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    turno = main.construir_turno_cliente("que es esto?", [FOTO])
    msgs = prompt.format_messages(input=turno, chat_history=[], agent_scratchpad=[])
    contenido = msgs[-1].content
    assert isinstance(contenido, list), "la foto se perdio: el prompt la convirtio a texto"
    assert any(b.get("type") == "image_url" for b in contenido)
