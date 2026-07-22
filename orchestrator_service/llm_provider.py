"""Selector de proveedor del LLM por el nombre del modelo (2026-07).

Motivo: OpenAI no deja cargar saldo; OpenRouter revende los MISMOS modelos de
OpenAI con API compatible. La migración NO cambia las variables de entorno ni
reemplaza la key: se activa por el NOMBRE DEL MODELO.

Regla:
  - modelo con "/"  (ej. "openai/gpt-4.1-mini")  -> OpenRouter + OPENROUTER_API_KEY
  - modelo sin "/"  (ej. "gpt-4.1-mini")         -> OpenAI directo + OPENAI_API_KEY

Ambas keys conviven: la de OpenRouter no sirve contra los servidores de OpenAI
(Whisper sigue necesitando OpenAI). Rollback = volver el modelo a la versión
sin "/". Función pura y sin dependencias: testeable en tests/test_llm_provider.py.
"""

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def resolve_llm_provider(model, openai_key, openrouter_key):
    """Devuelve la config del cliente para el modelo dado.

    Returns dict: {model, api_key, base_url (None = OpenAI directo), provider}.
    No lee variables de entorno: recibe las keys ya resueltas por el caller.
    """
    model = (model or "").strip()
    if "/" in model:
        return {
            "model": model,
            "api_key": openrouter_key,
            "base_url": OPENROUTER_BASE_URL,
            "provider": "openrouter",
        }
    return {
        "model": model,
        "api_key": openai_key,
        "base_url": None,
        "provider": "openai",
    }
