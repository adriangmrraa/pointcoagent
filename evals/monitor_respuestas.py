"""Monitor de cómo está contestando el bot, para mirar después de un deploy.

Lee las conversaciones por la API admin del orchestrator y marca las respuestas
sospechosas, cruzándolas contra el catálogo real de Tienda Nube. Es de SOLO
LECTURA: no toca la base, no manda mensajes, no deploya.

Lo que marca:
  - "no tenemos" sobre algo que el catálogo SÍ tiene con stock. El error que más
    plata cuesta, y el que nadie reporta.
  - Turnos con imagen, para ver qué contestó el bot cuando le mandaron una foto.
  - Derivaciones a humano, para ver si dispararon donde correspondía.
  - Respuestas donde el bot afirma algo de talles o calce, que es donde inventa.

Uso:
    python evals/monitor_respuestas.py                 # últimas 24 horas
    python evals/monitor_respuestas.py --horas 72
    python evals/monitor_respuestas.py --horas 24 --detalle

Necesita en el .env: ORCHESTRATOR_PUBLIC_URL, ADMIN_TOKEN, TIENDANUBE_STORE_ID,
TIENDANUBE_ACCESS_TOKEN.
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

# Frases con las que el bot niega tener algo.
NEGACION = re.compile(
    r"(?i)(no tenemos|no contamos con|no está en stock|no esta en stock|"
    r"no manejamos|no disponemos|no hay stock|sin stock|no tengo)"
)
# Donde el bot suele inventar: calce, horma, equivalencias de talle.
CALCE = re.compile(
    r"(?i)(horma|calza|calce|te quedar|un talle m[aá]s|un n[uú]mero m[aá]s|"
    r"suele[n]? ser m[aá]s|tiende[n]? a ser|equivale)"
)
# Palabras vacías que no sirven para buscar en el catálogo.
RUIDO = {
    "hola", "buenas", "gracias", "tenes", "tenés", "tienen", "tiene", "quiero",
    "necesito", "busco", "consulta", "precio", "cuanto", "cuánto", "sale",
    "para", "como", "cómo", "este", "esta", "ese", "esa", "eso", "que", "qué",
    "una", "unas", "unos", "los", "las", "del", "con", "sin", "por", "dale",
    "favor", "podes", "podés", "puedo", "seria", "sería", "talle", "color",
    "stock", "danza", "ballet", "chicas", "nena", "hija", "mandame", "mostrame",
}


def cargar_env():
    env = {}
    ruta = RAIZ / ".env"
    if not ruta.exists():
        sys.exit("No encuentro el .env en la raíz del proyecto")
    for linea in io.open(ruta, encoding="utf-8", errors="replace"):
        linea = linea.strip()
        if "=" in linea and not linea.startswith("#"):
            k, v = linea.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def admin(env, path):
    req = urllib.request.Request(
        env["ORCHESTRATOR_PUBLIC_URL"].rstrip("/") + path,
        headers={"X-Admin-Token": env["ADMIN_TOKEN"], "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode("utf-8"))


_cache_catalogo = {}


def catalogo_tiene(env, termino):
    """Productos CON STOCK que matchean ese término. Cacheado por término."""
    if termino in _cache_catalogo:
        return _cache_catalogo[termino]
    url = (f"https://api.tiendanube.com/v1/{env['TIENDANUBE_STORE_ID']}/products?"
           + urllib.parse.urlencode({"q": termino, "per_page": 10}, encoding="utf-8"))
    req = urllib.request.Request(url, headers={
        "Authentication": f"bearer {env['TIENDANUBE_ACCESS_TOKEN']}",
        "User-Agent": "monitor-respuestas/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            datos = json.loads(r.read().decode("utf-8"))
        nombres = [(p.get("name", {}).get("es") or "?") for p in datos if is_product_available(p)]
    except Exception:
        nombres = []
    _cache_catalogo[termino] = nombres
    return nombres


def textos_del_bot(contenido):
    """Los textos que la clienta realmente leyó, sacados del JSON del bot."""
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
    ap.add_argument("--horas", type=int, default=24)
    ap.add_argument("--detalle", action="store_true", help="mostrar cada caso completo")
    args = ap.parse_args()

    env = cargar_env()
    desde = datetime.now(timezone.utc) - timedelta(hours=args.horas)
    print(f"Mirando las últimas {args.horas} horas (desde {desde:%Y-%m-%d %H:%M} UTC)\n")

    chats = admin(env, "/admin/chats")
    recientes = [c for c in chats
                 if (c.get("last_message_at") or "") >= desde.strftime("%Y-%m-%dT%H:%M:%S")]
    print(f"Conversaciones con actividad: {len(recientes)} de {len(chats)}\n")

    stats = {"turnos_bot": 0, "con_imagen": 0, "derivaciones": 0,
             "negaciones": 0, "negaciones_falsas": 0, "afirma_calce": 0}
    hallazgos = []

    for chat in recientes:
        try:
            msgs = admin(env, f"/admin/chats/{chat['id']}/messages")
        except Exception:
            continue
        msgs = [m for m in msgs if (m.get("created_at") or "") >= desde.strftime("%Y-%m-%dT%H:%M:%S")]
        for i, m in enumerate(msgs):
            if m.get("role") == "user" and not (m.get("content") or "").strip():
                stats["con_imagen"] += 1
            if m.get("role") != "assistant":
                continue
            stats["turnos_bot"] += 1
            textos = textos_del_bot(m.get("content"))
            unido = " ".join(textos)
            if "asesora" in unido.lower() or "en breve se comunica" in unido.lower():
                stats["derivaciones"] += 1
            if CALCE.search(unido):
                stats["afirma_calce"] += 1
                hallazgos.append(("AFIRMA SOBRE CALCE", chat, m, unido, None))
            if not NEGACION.search(unido):
                continue
            stats["negaciones"] += 1
            # ¿Qué había pedido la clienta justo antes?
            pedido = ""
            for anterior in reversed(msgs[:i]):
                if anterior.get("role") == "user" and (anterior.get("content") or "").strip():
                    pedido = anterior["content"]
                    break
            # ¿El catálogo lo tenía?
            encontrados = []
            for termino in [t for t in query_terms(pedido) if t not in RUIDO and len(t) > 3][:4]:
                encontrados += catalogo_tiene(env, termino)
            if encontrados:
                stats["negaciones_falsas"] += 1
                hallazgos.append(("DIJO QUE NO Y SÍ HABÍA", chat, m, unido, (pedido, encontrados)))

    print("=" * 72)
    print(f"  turnos del bot              {stats['turnos_bot']}")
    print(f"  mensajes con imagen         {stats['con_imagen']}")
    print(f"  derivaciones a humano       {stats['derivaciones']}")
    print(f"  dijo 'no tenemos'           {stats['negaciones']}")
    print(f"  ...y el catálogo SÍ tenía   {stats['negaciones_falsas']}   <-- lo que cuesta plata")
    print(f"  afirmaciones sobre calce    {stats['afirma_calce']}   <-- donde suele inventar")
    print("=" * 72)

    if not hallazgos:
        print("\nSin hallazgos en esta ventana.")
        return

    print(f"\n{len(hallazgos)} caso(s) para mirar:\n")
    for tipo, chat, m, unido, extra in hallazgos:
        print(f"  [{tipo}]  {(m.get('created_at') or '')[:19]}  {chat.get('external_user_id')}")
        print(f"     conversación: {chat.get('id')}")
        if extra:
            pedido, encontrados = extra
            print(f"     la clienta pidió: {pedido[:110]!r}")
            print(f"     el catálogo tiene: {', '.join(encontrados[:3])}")
        print(f"     el bot contestó: {unido[:200]!r}")
        if args.detalle:
            print(f"     completo: {unido}")
        print()


if __name__ == "__main__":
    main()
