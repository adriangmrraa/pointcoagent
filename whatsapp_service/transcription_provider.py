"""Selector de proveedor de transcripción de audio (2026-07).

Tres modos, elegidos por TRANSCRIPTION_BASE_URL (rollback sin redeploy):

  1. OpenAI  (default)      -> Whisper vía /audio/transcriptions (multipart)
  2. Groq                   -> Whisper large-v3 vía /audio/transcriptions (multipart, compatible)
  3. OpenRouter             -> modelo multimodal vía /chat/completions con audio en base64
                              (OpenRouter NO tiene endpoint de transcripción dedicado)

Config por variables de entorno:
  - TRANSCRIPTION_BASE_URL  (default https://api.openai.com/v1)
  - TRANSCRIPTION_MODEL     (default whisper-1)
  - TRANSCRIPTION_API_KEY   (si falta, cae a OPENAI_API_KEY)

Ejemplos:
  Groq:       TRANSCRIPTION_BASE_URL=https://api.groq.com/openai/v1
              TRANSCRIPTION_MODEL=whisper-large-v3-turbo   TRANSCRIPTION_API_KEY=gsk_...
  OpenRouter: TRANSCRIPTION_BASE_URL=https://openrouter.ai/api/v1
              TRANSCRIPTION_MODEL=google/gemini-2.0-flash-001   TRANSCRIPTION_API_KEY=sk-or-v1-...

Funciones puras y sin dependencias: testeable en tests/test_transcription_provider.py.
"""

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "whisper-1"

# Instrucción para el modo chat_audio: forzar transcripción literal, sin adornos.
AUDIO_TRANSCRIBE_PROMPT = (
    "Transcribí este audio al español rioplatense, palabra por palabra. "
    "No traduzcas, no resumas, no agregues comentarios ni signos de más. "
    "Devolvé ÚNICAMENTE el texto exacto de lo que se dijo."
)


def resolve_transcription_config(base_url, model, transcription_key, openai_key):
    """Devuelve {base_url, url, chat_url, model, api_key, provider, mode}.

    La key propia de transcripción tiene prioridad; si no está, cae a la de OpenAI
    (así el default sigue funcionando con OPENAI_API_KEY, sin config nueva).
    """
    base = (base_url or DEFAULT_BASE_URL).strip().rstrip("/")
    key = transcription_key or openai_key
    if "openrouter.ai" in base:
        provider, mode = "openrouter", "chat_audio"
    elif "groq.com" in base:
        provider, mode = "groq", "whisper"
    elif "openai.com" in base:
        provider, mode = "openai", "whisper"
    else:
        provider, mode = "custom", "whisper"
    return {
        "base_url": base,
        "url": base + "/audio/transcriptions",   # modo whisper (multipart)
        "chat_url": base + "/chat/completions",   # modo chat_audio (base64)
        "model": (model or DEFAULT_MODEL).strip(),
        "api_key": key,
        "provider": provider,
        "mode": mode,
    }


def audio_format_from_mime(mime_type):
    """Mapea el mime del audio de WhatsApp al 'format' que espera OpenRouter."""
    m = (mime_type or "").lower()
    if "ogg" in m or "opus" in m:
        return "ogg"
    if "mpeg" in m or "mp3" in m:
        return "mp3"
    if "mp4" in m or "m4a" in m or "aac" in m:
        return "m4a"
    if "wav" in m:
        return "wav"
    return "ogg"  # WhatsApp manda ogg/opus por defecto


def build_audio_chat_payload(model, audio_b64, audio_format="ogg"):
    """Arma el body de /chat/completions con el audio como input_audio (base64)."""
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": AUDIO_TRANSCRIBE_PROMPT},
                    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": audio_format}},
                ],
            }
        ],
    }
