"""Test local: ¿openai/gpt-4.1-mini vía OpenRouter llama a las herramientas?

Compara el API viejo ('functions', el que alucinaba) contra el nuevo ('tools',
el del fix). La key se lee del .env LOCAL (git lo ignora): NUNCA se imprime ni
sale de tu máquina.

Uso:
  1. En el .env local agregá una línea:  OPENROUTER_API_KEY=sk-or-v1-tu-key
  2. python evals/test_openrouter_tools.py
  3. (después borrá esa línea del .env si querés)
"""
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
env = {}
for line in (ROOT / ".env").read_text(encoding="utf-8", errors="ignore").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip('"').strip("'")

KEY = env.get("OPENROUTER_API_KEY")
if not KEY:
    print("Falta OPENROUTER_API_KEY en el .env local. Agregá la línea y volvé a correr.")
    sys.exit(1)

MODEL = "openai/gpt-4.1-mini"
URL = "https://openrouter.ai/api/v1/chat/completions"
USER_MSG = "Pasame las mochilas disponibles"

# Herramienta de mentira (como la del bot). Esperamos que el modelo la LLAME.
TOOL_DESC = "Busca productos disponibles en la tienda por categoría. Usar para bolsos/mochilas."


def post(payload):
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return {"_error": f"HTTP {e.code}: {e.read().decode()[:200]}"}


def resumen(resp):
    if "_error" in resp:
        return "ERROR: " + resp["_error"]
    msg = (resp.get("choices") or [{}])[0].get("message", {})
    if msg.get("tool_calls"):
        names = [c.get("function", {}).get("name") for c in msg["tool_calls"]]
        return f"LLAMÓ A LA TOOL ✅  ({names})"
    if msg.get("function_call"):
        return f"LLAMÓ A LA FUNCIÓN ✅  ({msg['function_call'].get('name')})"
    txt = (msg.get("content") or "")[:80].replace("\n", " ")
    return f"NO llamó tool ❌ (respondió texto: {txt!r})"


base = {"model": MODEL, "messages": [{"role": "user", "content": USER_MSG}], "max_tokens": 100}

# 1) API MODERNO 'tools' (el del fix)
tools_payload = dict(base, tools=[{
    "type": "function",
    "function": {
        "name": "search_by_store_category",
        "description": TOOL_DESC,
        "parameters": {"type": "object", "properties": {"category_name": {"type": "string"}}, "required": ["category_name"]},
    },
}])

# 2) API VIEJO 'functions' (el que alucinaba)
functions_payload = dict(base, functions=[{
    "name": "search_by_store_category",
    "description": TOOL_DESC,
    "parameters": {"type": "object", "properties": {"category_name": {"type": "string"}}, "required": ["category_name"]},
}])

print(f"Probando {MODEL} vía OpenRouter con: {USER_MSG!r}\n")
print("API NUEVO 'tools' (el del fix que hicimos):")
print("   ->", resumen(post(tools_payload)))
print("\nAPI VIEJO 'functions' (el que estaba antes):")
print("   ->", resumen(post(functions_payload)))
print("\nSi 'tools' dice LLAMÓ y 'functions' dice NO llamó -> el fix es correcto.")
