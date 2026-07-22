"""Tests del selector de proveedor de transcripción (whatsapp_service/transcription_provider.py).

Función pura: corre sin red ni API keys.
    python -m pytest tests/test_transcription_provider.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "whatsapp_service"))

from transcription_provider import resolve_transcription_config

OA = "sk-openai-key"
GROQ = "gsk_groq-key"


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
