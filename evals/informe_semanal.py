"""Informe semanal: cómo contestó el bot y dónde se equivocó.

Pensado para correr los viernes. Hace dos cosas:

  1. Exporta las conversaciones de la semana en un archivo legible, para leerlas
     como una charla de WhatsApp y no como JSON.
  2. Marca lo que conviene mirar: "no tenemos" sobre productos que SÍ hay,
     afirmaciones sobre calce que el bot suele inventar, correcciones del equipo
     humano justo después del bot, y charlas que quedaron sin respuesta.

Es de SOLO LECTURA. No toca la base, no manda mensajes, no deploya.

Uso:
    python evals/informe_semanal.py
    python evals/informe_semanal.py --dias 7 --salida "C:\\ruta\\carpeta"

El informe se guarda FUERA del repo (contiene nombres y teléfonos de clientas).
Por defecto en Documentos\\pointcoagent-informes.
"""
import argparse
import io
import json
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "orchestrator_service"))
from catalog import is_product_available, query_terms  # noqa: E402

NEGACION = re.compile(r"(?i)(no tenemos|no contamos con|no est[aá] en stock|no manejamos|"
                      r"no disponemos|no hay stock|sin stock|no tengo)")
CALCE = re.compile(r"(?i)(horma|calza|calce|te quedar|un talle m[aá]s|un n[uú]mero m[aá]s|"
                   r"suele[n]? ser m[aá]s|tiende[n]? a ser|equivale)")
# Si la respuesta trae un producto con precio o link, el bot ofrecio algo.
PRODUCTO = re.compile(r"(?i)(precio:\s*\$|pointecoach\.shop/productos)")
# Pedir el talle no es afirmar sobre el calce.
PREGUNTA_TALLE = re.compile(r"(?i)(decime|contame|me pas[aá]s|qu[eé] talle|cu[aá]l es (el|tu) talle|\?\s*$)")
CORRECCION = re.compile(r"(?i)^(no |en realidad|perd[oó]n|disculp|te refer[ií]|"
                        r"eso no|no es as[ií]|me parece que no)")
RUIDO = {"hola", "buenas", "gracias", "tenes", "tenés", "tienen", "tiene", "quiero",
         "necesito", "busco", "consulta", "precio", "cuanto", "cuánto", "sale", "para",
         "como", "cómo", "este", "esta", "ese", "esa", "eso", "que", "qué", "una",
         "unas", "unos", "los", "las", "del", "con", "sin", "por", "dale", "favor",
         "podes", "podés", "puedo", "seria", "sería", "talle", "color", "stock",
         "danza", "ballet", "chicas", "nena", "hija", "mandame", "mostrame"}

ETIQUETA = {"user": "CLIENTA", "assistant": "BOT    ", "human_supervisor": "EQUIPO "}


def cargar_env():
    ruta = RAIZ / ".env"
    if not ruta.exists():
        sys.exit("No encuentro el .env en la raíz del proyecto")
    env = {}
    for linea in io.open(ruta, encoding="utf-8", errors="replace"):
        linea = linea.strip()
        if "=" in linea and not linea.startswith("#"):
            k, v = linea.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def admin(env, path):
    req = urllib.request.Request(
        env["ORCHESTRATOR_PUBLIC_URL"].rstrip("/") + path,
        headers={"X-Admin-Token": env["ADMIN_TOKEN"], "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode("utf-8"))


_cache = {}


def catalogo_tiene(env, termino):
    if termino in _cache:
        return _cache[termino]
    url = (f"https://api.tiendanube.com/v1/{env['TIENDANUBE_STORE_ID']}/products?"
           + urllib.parse.urlencode({"q": termino, "per_page": 10}, encoding="utf-8"))
    req = urllib.request.Request(url, headers={
        "Authentication": f"bearer {env['TIENDANUBE_ACCESS_TOKEN']}",
        "User-Agent": "informe-semanal/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            datos = json.loads(r.read().decode("utf-8"))
        nombres = [(p.get("name", {}).get("es") or "?") for p in datos if is_product_available(p)]
    except Exception:
        nombres = []
    _cache[termino] = nombres
    return nombres


def textos_del_bot(contenido):
    if not contenido:
        return []
    bruto = contenido.strip()
    if not bruto.startswith("{"):
        return [bruto]
    try:
        d = json.loads(bruto)
    except Exception:
        return [bruto]
    if isinstance(d, dict) and isinstance(d.get("messages"), list):
        return [m.get("text") or "" for m in d["messages"] if isinstance(m, dict)]
    return [bruto]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dias", type=int, default=7)
    ap.add_argument("--salida", default=str(Path.home() / "OneDrive" / "Documentos" / "pointcoagent-informes"))
    args = ap.parse_args()

    env = cargar_env()
    desde = datetime.now(timezone.utc) - timedelta(days=args.dias)
    corte = desde.strftime("%Y-%m-%dT%H:%M:%S")
    carpeta = Path(args.salida)
    carpeta.mkdir(parents=True, exist_ok=True)
    destino = carpeta / f"semana-{datetime.now():%Y-%m-%d}.md"

    print(f"Trayendo los últimos {args.dias} días...")
    chats = admin(env, "/admin/chats")
    activas = [c for c in chats if (c.get("last_message_at") or "") >= corte]
    print(f"  {len(activas)} conversaciones con actividad, de {len(chats)} en total")

    stats = {"clientas": 0, "bot": 0, "equipo": 0, "imagenes": 0, "derivaciones": 0}
    hallazgos = []
    transcripciones = []

    for n, chat in enumerate(activas, 1):
        if n % 10 == 0:
            print(f"  ...{n}/{len(activas)}")
        try:
            msgs = [m for m in admin(env, f"/admin/chats/{chat['id']}/messages")
                    if (m.get("created_at") or "") >= corte]
        except Exception:
            continue
        if not msgs:
            continue

        lineas = [f"\n### {chat.get('external_user_id')}  ·  {chat.get('id')}\n"]
        for i, m in enumerate(msgs):
            rol = m.get("role")
            ts = (m.get("created_at") or "")[11:16]
            if rol == "user":
                stats["clientas"] += 1
                cont = (m.get("content") or "").strip()
                if not cont:
                    stats["imagenes"] += 1
                    cont = "[imagen o archivo]"
                lineas.append(f"- `{ts}` **CLIENTA** · {cont}")
            elif rol == "assistant":
                stats["bot"] += 1
                textos = [t for t in textos_del_bot(m.get("content")) if t]
                unido = " ".join(textos)
                lineas.append(f"- `{ts}` **BOT** · {' ⏐ '.join(t[:300] for t in textos)}")
                if "asesora" in unido.lower() or "en breve se comunica" in unido.lower():
                    stats["derivaciones"] += 1
                # Ofrecer una alternativa NO es un error: "no tengo eso exacto,
                # pero mirá esto" es buena atencion. El error es negar y no dar
                # nada, o dar algo de otro rubro.
                mostro_productos = bool(PRODUCTO.search(unido))
                # El calce solo preocupa cuando AFIRMA; preguntar el talle esta bien.
                if CALCE.search(unido) and not PREGUNTA_TALLE.search(unido):
                    hallazgos.append(("Afirma sobre calce o talle", chat, ts, unido[:220], ""))
                if NEGACION.search(unido):
                    pedido = next((a["content"] for a in reversed(msgs[:i])
                                   if a.get("role") == "user" and (a.get("content") or "").strip()), "")
                    if not mostro_productos:
                        hay = []
                        for t in [x for x in query_terms(pedido) if x not in RUIDO and len(x) > 3][:4]:
                            hay += catalogo_tiene(env, t)
                        etiqueta = ("Dijo que no y NO ofrecio nada"
                                    + (", y el catalogo podria tener" if hay else ""))
                        hallazgos.append((etiqueta, chat, ts, unido[:220],
                                          f"pidió {pedido[:70]!r}"
                                          + (f" · podria ser: {', '.join(hay[:3])}" if hay else "")))
                # ¿el equipo lo corrigió justo después?
                sig = msgs[i + 1] if i + 1 < len(msgs) else None
                if sig and sig.get("role") == "human_supervisor" and CORRECCION.match((sig.get("content") or "").strip()):
                    hallazgos.append(("El equipo lo corrigió", chat, ts, unido[:160],
                                      f"corrección: {(sig.get('content') or '')[:90]!r}"))
            else:
                stats["equipo"] += 1
                lineas.append(f"- `{ts}` *equipo* · {(m.get('content') or '')[:200]}")
        transcripciones.append("".join(f"{l}\n" for l in lineas))

    doc = [f"# Informe semanal — {datetime.now():%d/%m/%Y}",
           f"\nÚltimos {args.dias} días. {len(activas)} conversaciones con actividad.\n",
           "\n## Resumen\n",
           f"- Mensajes de clientas: **{stats['clientas']}**",
           f"- Respuestas del bot: **{stats['bot']}**",
           f"- Mensajes del equipo: **{stats['equipo']}**",
           f"- Imágenes recibidas: **{stats['imagenes']}**",
           f"- Derivaciones a humano: **{stats['derivaciones']}**",
           f"\n## Para mirar ({len(hallazgos)})\n"]
    if not hallazgos:
        doc.append("Sin hallazgos esta semana.\n")
    else:
        for tipo, chat, ts, texto, extra in hallazgos:
            doc.append(f"**{tipo}** · `{ts}` · {chat.get('external_user_id')}")
            if extra:
                doc.append(f"  - {extra}")
            doc.append(f"  - el bot dijo: {texto!r}")
            doc.append(f"  - conversación: `{chat.get('id')}`\n")
    doc.append("\n---\n\n## Conversaciones\n")
    doc += transcripciones

    io.open(destino, "w", encoding="utf-8").write("\n".join(doc))

    print()
    print(f"  clientas {stats['clientas']} · bot {stats['bot']} · equipo {stats['equipo']}"
          f" · imágenes {stats['imagenes']} · derivaciones {stats['derivaciones']}")
    print(f"  hallazgos para mirar: {len(hallazgos)}")
    print(f"\nInforme guardado en:\n  {destino}")


if __name__ == "__main__":
    main()
