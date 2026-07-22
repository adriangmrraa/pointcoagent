"""Selector de proveedor de transcripción de audio (2026-07).

OpenRouter NO revende transcripción de audio (/audio/transcriptions). Para sacar
el gasto de OpenAI también en los audios, se apunta a Groq, que tiene API
compatible con OpenAI y corre Whisper large-v3 (más barato/rápido).

Configurable por variables de entorno (rollback sin redeploy):
  - TRANSCRIPTION_BASE_URL  (default https://api.openai.com/v1 = OpenAI)
  - TRANSCRIPTION_MODEL     (default whisper-1)
  - TRANSCRIPTION_API_KEY   (si falta, cae a OPENAI_API_KEY)

Para migrar a Groq:
  TRANSCRIPTION_BASE_URL=https://api.groq.com/openai/v1
  TRANSCRIPTION_MODEL=whisper-large-v3-turbo
  TRANSCRIPTION_API_KEY=gsk_...

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
    if "groq.com" in base:
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
