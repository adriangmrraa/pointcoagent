"""Tests del selector de proveedor de transcripción (whatsapp_service/transcription_provider.py).

Función pura: corre sin red ni API keys.
    python -m pytest tests/test_transcription_provider.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "whatsapp_service"))

from transcription_provider import (
    resolve_transcription_config,
    audio_format_from_mime,
    build_audio_chat_payload,
)

OA = "sk-openai-key"
GROQ = "gsk_groq-key"
OR = "sk-or-v1-key"


def test_default_es_openai_con_key_de_openai():
    c = resolve_transcription_config(None, None, None, OA)
    assert c["provider"] == "openai"
    assert c["url"] == "https://api.openai.com/v1/audio/transcriptions"
    assert c["model"] == "whisper-1"
    assert c["api_key"] == OA


def test_groq_usa_su_endpoint_modelo_y_key():
    c = resolve_transcription_config("https://api.groq.com/openai/v1", "whisper-large-v3-turbo", GROQ, OA)
    assert c["provider"] == "groq"
    assert c["url"] == "https://api.groq.com/openai/v1/audio/transcriptions"
    assert c["model"] == "whisper-large-v3-turbo"
    assert c["api_key"] == GROQ


def test_key_de_transcripcion_tiene_prioridad_sobre_openai():
    c = resolve_transcription_config("https://api.groq.com/openai/v1", "whisper-large-v3", GROQ, OA)
    assert c["api_key"] == GROQ
    assert c["api_key"] != OA


def test_sin_key_propia_cae_a_openai():
    """Si no se setea TRANSCRIPTION_API_KEY, usa OPENAI_API_KEY (default no rompe)."""
    c = resolve_transcription_config(None, None, None, OA)
    assert c["api_key"] == OA


def test_barra_final_en_base_url_no_duplica():
    c = resolve_transcription_config("https://api.groq.com/openai/v1/", None, GROQ, OA)
    assert c["url"] == "https://api.groq.com/openai/v1/audio/transcriptions"


def test_rollback_limpiando_vars_vuelve_a_openai():
    groq = resolve_transcription_config("https://api.groq.com/openai/v1", "whisper-large-v3", GROQ, OA)
    openai = resolve_transcription_config(None, None, None, OA)
    assert groq["provider"] == "groq"
    assert openai["provider"] == "openai"
    assert openai["api_key"] == OA


# --- Modo OpenRouter (audio vía chat multimodal) ---

def test_openrouter_usa_modo_chat_audio():
    c = resolve_transcription_config("https://openrouter.ai/api/v1", "google/gemini-2.0-flash-001", OR, OA)
    assert c["provider"] == "openrouter"
    assert c["mode"] == "chat_audio"
    assert c["chat_url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert c["api_key"] == OR


def test_openai_y_groq_usan_modo_whisper():
    assert resolve_transcription_config(None, None, None, OA)["mode"] == "whisper"
    assert resolve_transcription_config("https://api.groq.com/openai/v1", None, GROQ, OA)["mode"] == "whisper"


def test_audio_format_from_mime():
    assert audio_format_from_mime("audio/ogg; codecs=opus") == "ogg"
    assert audio_format_from_mime("audio/mpeg") == "mp3"
    assert audio_format_from_mime("audio/mp4") == "m4a"
    assert audio_format_from_mime("audio/wav") == "wav"
    assert audio_format_from_mime(None) == "ogg"  # WhatsApp default


def test_build_audio_chat_payload_estructura():
    p = build_audio_chat_payload("google/gemini-2.0-flash-001", "BASE64DATA", "ogg")
    assert p["model"] == "google/gemini-2.0-flash-001"
    content = p["messages"][0]["content"]
    tipos = [c["type"] for c in content]
    assert "text" in tipos and "input_audio" in tipos
    audio_part = next(c for c in content if c["type"] == "input_audio")
    assert audio_part["input_audio"] == {"data": "BASE64DATA", "format": "ogg"}


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
