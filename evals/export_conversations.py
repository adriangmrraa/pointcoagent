"""
Descarga las conversaciones reales del bot desde la API de administracion
del orchestrator (la misma que usa el panel platform_ui). No requiere SSH.

Uso:
    python evals/export_conversations.py
    python evals/export_conversations.py --url https://mi-orquestador.easypanel.host --token MI_ADMIN_TOKEN

Si no se pasan argumentos, lee ORCHESTRATOR_PUBLIC_URL y ADMIN_TOKEN del .env de la raiz.
Guarda cada conversacion como JSON legible en evals/data/conversations/
(carpeta ignorada por git: son datos personales de clientas).
"""
import argparse
import json
import os
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / "data" / "conversations"


def load_env_file(path: Path) -> dict:
    """Parser minimo de .env (sin dependencias)."""
    env = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        env[key.strip()] = val.strip().strip('"').strip("'")
    return env


def api_get(base_url: str, path: str, token: str):
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        headers={"X-Admin-Token": token, "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def prettify_assistant_content(content):
    """Los mensajes del asistente se guardan como JSON crudo; extrae el texto legible."""
    if not content or not isinstance(content, str):
        return content
    raw = content.strip()
    if not raw.startswith("{") and not raw.startswith("["):
        return content
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and isinstance(parsed.get("messages"), list):
            parts = []
            for m in parsed["messages"]:
                if isinstance(m, dict):
                    if m.get("text"):
                        parts.append(m["text"])
                    if m.get("imageUrl"):
                        parts.append(f"[imagen: {m['imageUrl']}]")
            return "\n---\n".join(parts) if parts else content
    except (json.JSONDecodeError, TypeError):
        pass
    return content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="URL publica del orchestrator (ej: https://xxx.easypanel.host)")
    parser.add_argument("--token", help="ADMIN_TOKEN del panel")
    parser.add_argument("--limit", type=int, default=0, help="Max conversaciones a bajar (0 = todas)")
    args = parser.parse_args()

    env = load_env_file(ROOT / ".env")
    base_url = args.url or env.get("ORCHESTRATOR_PUBLIC_URL") or os.getenv("ORCHESTRATOR_PUBLIC_URL")
    token = args.token or env.get("ADMIN_TOKEN") or os.getenv("ADMIN_TOKEN")

    if not base_url or not token:
        print("ERROR: falta la URL del orchestrator o el ADMIN_TOKEN.")
        print("Opcion A: agrega al .env de la raiz estas dos lineas y volve a correr:")
        print("    ORCHESTRATOR_PUBLIC_URL=https://tu-orquestador.easypanel.host")
        print("    ADMIN_TOKEN=el_token_del_panel")
        print("Opcion B: pasalos por argumento: --url ... --token ...")
        sys.exit(1)

    print(f"Conectando a {base_url} ...")
    try:
        chats = api_get(base_url, "/admin/chats", token)
    except urllib.error.HTTPError as e:
        print(f"ERROR HTTP {e.code}: {e.reason}")
        if e.code == 401:
            print("El ADMIN_TOKEN no coincide con el del orchestrator.")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"ERROR de conexion: {e.reason}")
        print("Verifica que la URL sea la publica del orchestrator (no la del panel).")
        sys.exit(1)

    if args.limit:
        chats = chats[: args.limit]
    print(f"Conversaciones encontradas: {len(chats)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok, failed = 0, 0
    for chat in chats:
        cid = chat.get("id")
        try:
            messages = api_get(base_url, f"/admin/chats/{cid}/messages", token)
        except Exception as e:
            print(f"  [fallo] {cid}: {e}")
            failed += 1
            continue

        for m in messages:
            if m.get("role") == "assistant":
                m["content_display"] = prettify_assistant_content(m.get("content"))

        phone = re.sub(r"[^0-9]", "", str(chat.get("external_user_id") or "desconocido"))
        fname = f"{phone or 'sin_numero'}_{str(cid)[:8]}.json"
        out = {
            "conversation": chat,
            "message_count": len(messages),
            "messages": messages,
        }
        (OUT_DIR / fname).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        ok += 1
        name = chat.get("display_name") or phone
        print(f"  [ok] {name}: {len(messages)} mensajes -> {fname}")

    print(f"\nListo. Guardadas {ok} conversaciones en {OUT_DIR} (fallidas: {failed})")
    print("Ahora podemos revisar juntos cuales convertir en casos de examen.")


if __name__ == "__main__":
    main()
