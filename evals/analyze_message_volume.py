"""Análisis de volumen mensual de mensajes del bot (para estimar costo Meta).

Descarga el historial completo vía la API admin del orchestrator y calcula,
mes a mes:
  - mensajes recibidos de clientas
  - respuestas del bot (turnos)
  - mensajes de WhatsApp REALES que esas respuestas generaron con el formato
    viejo (imagen y texto separados)
  - mensajes que generarían con el formato nuevo (imagen+caption fusionados)
  - mensajes manuales del equipo humano (no facturables por API)

Uso:
    python evals/analyze_message_volume.py
    python evals/analyze_message_volume.py --precio 0.05   # USD por mensaje para proyectar

Requiere en el .env: ORCHESTRATOR_PUBLIC_URL y ADMIN_TOKEN.
"""
import argparse
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "whatsapp_service"))
from sequence_planner import plan_send_actions, split_text  # noqa: E402


def load_env(path):
    env = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def api_get(base, path, token):
    req = urllib.request.Request(
        base.rstrip("/") + path,
        headers={"X-Admin-Token": token, "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def parse_bubbles(content):
    """Extrae la lista de burbujas {text, imageUrl} de una respuesta del bot."""
    if not content or not isinstance(content, str):
        return []
    raw = content.strip()
    if not raw.startswith("{") and not raw.startswith("["):
        return [{"text": raw, "imageUrl": None}]
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return [{"text": raw, "imageUrl": None}]
    if isinstance(parsed, dict) and isinstance(parsed.get("messages"), list):
        out = []
        for m in parsed["messages"]:
            if isinstance(m, dict) and (m.get("text") or m.get("imageUrl")):
                out.append({"text": m.get("text"), "imageUrl": m.get("imageUrl")})
        return out
    return [{"text": raw, "imageUrl": None}]


def count_old_format(bubbles):
    """Mensajes reales con el formato viejo: imagen y texto por separado,
    textos >400 divididos."""
    n = 0
    for b in bubbles:
        if b.get("imageUrl"):
            n += 1
        n += len(split_text(b.get("text")))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="URL pública del orchestrator")
    ap.add_argument("--token", help="ADMIN_TOKEN")
    ap.add_argument("--precio", type=float, default=None, help="USD por mensaje para proyección de costo")
    args = ap.parse_args()

    env = load_env(ROOT / ".env")
    base = args.url or env.get("ORCHESTRATOR_PUBLIC_URL")
    token = args.token or env.get("ADMIN_TOKEN")
    if not base or not token:
        print("ERROR: falta ORCHESTRATOR_PUBLIC_URL o ADMIN_TOKEN (en .env o por --url/--token)")
        sys.exit(1)

    print(f"Conectando a {base} ...")
    chats = api_get(base, "/admin/chats", token)
    print(f"Conversaciones: {len(chats)}")

    # mes -> contadores
    stats = defaultdict(lambda: {"user": 0, "turnos_bot": 0, "msgs_viejo": 0, "msgs_nuevo": 0, "humano": 0})
    fallidas = 0

    for idx, chat in enumerate(chats, 1):
        try:
            messages = api_get(base, f"/admin/chats/{chat['id']}/messages", token)
        except Exception:
            fallidas += 1
            continue
        for m in messages:
            mes = (m.get("created_at") or "")[:7]  # YYYY-MM
            if not mes:
                continue
            role = m.get("role")
            if role == "user":
                stats[mes]["user"] += 1
            elif role == "human_supervisor":
                stats[mes]["humano"] += 1
            elif role == "assistant":
                bubbles = parse_bubbles(m.get("content"))
                if not bubbles:
                    continue
                stats[mes]["turnos_bot"] += 1
                stats[mes]["msgs_viejo"] += count_old_format(bubbles)
                stats[mes]["msgs_nuevo"] += len(plan_send_actions(bubbles))
        if idx % 25 == 0:
            print(f"  ...procesadas {idx}/{len(chats)} conversaciones")

    if fallidas:
        print(f"AVISO: {fallidas} conversaciones no se pudieron leer")

    print()
    print(f"{'Mes':<9}{'Clientas':>10}{'Turnos bot':>12}{'Msgs (viejo)':>14}{'Msgs (nuevo)':>14}{'Ahorro':>9}{'Humanas':>9}")
    tot = {"user": 0, "turnos_bot": 0, "msgs_viejo": 0, "msgs_nuevo": 0, "humano": 0}
    for mes in sorted(stats):
        s = stats[mes]
        for k in tot:
            tot[k] += s[k]
        ahorro = (1 - s["msgs_nuevo"] / s["msgs_viejo"]) * 100 if s["msgs_viejo"] else 0
        print(f"{mes:<9}{s['user']:>10}{s['turnos_bot']:>12}{s['msgs_viejo']:>14}{s['msgs_nuevo']:>14}{ahorro:>8.0f}%{s['humano']:>9}")

    meses = max(len(stats), 1)
    ahorro_total = (1 - tot["msgs_nuevo"] / tot["msgs_viejo"]) * 100 if tot["msgs_viejo"] else 0
    print("-" * 77)
    print(f"{'TOTAL':<9}{tot['user']:>10}{tot['turnos_bot']:>12}{tot['msgs_viejo']:>14}{tot['msgs_nuevo']:>14}{ahorro_total:>8.0f}%{tot['humano']:>9}")
    print()
    print(f"Promedio mensual (sobre {meses} meses):")
    print(f"  - Formato viejo:  {tot['msgs_viejo'] / meses:,.0f} mensajes facturables/mes")
    print(f"  - Formato nuevo:  {tot['msgs_nuevo'] / meses:,.0f} mensajes facturables/mes")

    if args.precio:
        print()
        print(f"Proyección de costo a USD {args.precio}/mensaje:")
        print(f"  - Formato viejo:  USD {tot['msgs_viejo'] / meses * args.precio:,.2f}/mes")
        print(f"  - Formato nuevo:  USD {tot['msgs_nuevo'] / meses * args.precio:,.2f}/mes")


if __name__ == "__main__":
    main()
