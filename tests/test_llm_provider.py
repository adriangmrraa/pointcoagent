"""Tests del selector de proveedor LLM (orchestrator_service/llm_provider.py).

Función pura: corre sin red ni API keys.
    python -m pytest tests/test_llm_provider.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "orchestrator_service"))

from llm_provider import resolve_llm_provider, OPENROUTER_BASE_URL

OA = "sk-openai-key"
OR = "sk-or-v1-openrouter-key"


def test_modelo_sin_barra_usa_openai_directo():
    p = resolve_llm_provider("gpt-4.1-mini", OA, OR)
    assert p["provider"] == "openai"
    assert p["api_key"] == OA
    assert p["base_url"] is None  # None = cliente OpenAI por defecto
    assert p["model"] == "gpt-4.1-mini"


def test_modelo_con_barra_usa_openrouter():
    p = resolve_llm_provider("openai/gpt-4.1-mini", OA, OR)
    assert p["provider"] == "openrouter"
    assert p["api_key"] == OR
    assert p["base_url"] == OPENROUTER_BASE_URL
    assert p["model"] == "openai/gpt-4.1-mini"  # el nombre se pasa tal cual con el prefijo


def test_openrouter_NO_usa_la_key_de_openai():
    """El error clásico: mezclar keys. OpenRouter debe usar SU key, nunca la de OpenAI."""
    p = resolve_llm_provider("openai/gpt-4o", OA, OR)
    assert p["api_key"] == OR
    assert p["api_key"] != OA


def test_openai_directo_NO_usa_la_key_de_openrouter():
    p = resolve_llm_provider("gpt-4o-mini", OA, OR)
    assert p["api_key"] == OA
    assert p["api_key"] != OR


def test_rollback_instantaneo_quitando_la_barra():
    """Paso 6: volver el modelo sin '/' devuelve a OpenAI directo."""
    antes = resolve_llm_provider("openai/gpt-4.1-mini", OA, OR)
    despues = resolve_llm_provider("gpt-4.1-mini", OA, OR)
    assert antes["provider"] == "openrouter"
    assert despues["provider"] == "openai"
    assert despues["base_url"] is None


def test_otros_modelos_de_openrouter_tambien_matchean():
    # Cualquier proveedor con "/" cae en el flujo OpenRouter (API compatible)
    for m in ["openai/gpt-4o", "openai/gpt-4o-mini", "anthropic/claude-sonnet-5"]:
        assert resolve_llm_provider(m, OA, OR)["provider"] == "openrouter"


def test_espacios_y_none_no_rompen():
    assert resolve_llm_provider("  gpt-4.1-mini  ", OA, OR)["provider"] == "openai"
    assert resolve_llm_provider("", OA, OR)["provider"] == "openai"
    assert resolve_llm_provider(None, OA, OR)["provider"] == "openai"


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
