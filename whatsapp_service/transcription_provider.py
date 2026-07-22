"""Selector de proveedor de transcripción de audio (2026-07).

Los tres proveedores exponen el MISMO endpoint compatible con OpenAI
(/audio/transcriptions, multipart), así que se usa el mismo código para todos;
solo cambian base_url + key + modelo (rollback sin redeploy):

  - OpenAI     (default)  https://api.openai.com/v1     modelo whisper-1
  - OpenRouter            https://openrouter.ai/api/v1   modelo openai/whisper-1   (Whisper real, verificado 2026-07)
  - Groq                  https://api.groq.com/openai/v1 modelo whisper-large-v3-turbo

Config por variables de entorno:
  - TRANSCRIPTION_BASE_URL  (default https://api.openai.com/v1)
  - TRANSCRIPTION_MODEL     (default whisper-1)
  - TRANSCRIPTION_API_KEY   (si falta, cae a OPENAI_API_KEY)

Función pura y sin dependencias: testeable en tests/test_transcription_provider.py.
"""

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "whisper-1"


def resolve_transcription_config(base_url, model, transcription_key, openai_key):
    """Devuelve {url, model, api_key, provider} para la llamada de transcripción.

    La key propia de transcripción tiene prioridad; si no está, cae a la de OpenAI
    (así el default sigue funcionando con OPENAI_API_KEY, sin config nueva).
    """
    base = (base_url or DEFAULT_BASE_URL).strip().rstrip("/")
    key = transcription_key or openai_key
    if "openrouter.ai" in base:
        provider = "openrouter"
    elif "groq.com" in base:
        provider = "groq"
    elif "openai.com" in base:
        provider = "openai"
    else:
        provider = "custom"
    return {
        "url": base + "/audio/transcriptions",
        "model": (model or DEFAULT_MODEL).strip(),
        "api_key": key,
        "provider": provider,
    }
