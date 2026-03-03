# 01 · Pipeline de Entrada de Mensajes WhatsApp

> **Audiencia:** programadores. El código de referencia está en `whatsapp_service/main.py` pero los conceptos son agnósticos al lenguaje.

---

## Visión General

Cuando un usuario escribe por WhatsApp, el proveedor (YCloud, en este caso) hace un **HTTP POST** a tu servidor. Tu trabajo es:

1. Verificar que el mensaje es legítimo (seguridad)
2. Acumularlo si el usuario está escribiendo varios mensajes seguidos (debounce)
3. Enviarlo al motor de IA una sola vez, con todo el contexto junto

---

## Parte 1 — Seguridad: Verificación HMAC

### Por qué existe

Cualquiera puede hacer un POST a tu endpoint si la URL es pública. La firma HMAC garantiza que el mensaje viene realmente de YCloud y no fue modificado en tránsito.

### Cómo funciona

YCloud adjunta un header en cada webhook:

```
ycloud-signature: t=1712345678,s=a3f9c2...
```

- `t` = timestamp Unix del momento en que YCloud firmó
- `s` = HMAC-SHA256 del payload

Tu server reconstruye la firma esperada:

```
signed_payload = "{t}.{raw_body_as_string}"
expected_signature = HMAC_SHA256(key=WEBHOOK_SECRET, message=signed_payload)
```

Si `expected_signature != s` → rechazar con **401**.

### Validación de timestamp (anti-replay)

```
if abs(now() - t) > 300 seconds:
    reject(401, "Timestamp out of tolerance")
```

Esto impide que alguien capture un webhook legítimo y lo reenvíe horas después.

### Pseudocódigo (agnóstico al lenguaje)

```
function verify_signature(request):
    header = request.headers["ycloud-signature"]
    parts = parse_header(header)   # → {t: "...", s: "..."}

    if not parts.t or not parts.s:
        raise Unauthorized("Missing t or s")

    if abs(current_unix_time() - int(parts.t)) > 300:
        raise Unauthorized("Timestamp expired")

    raw_body = request.raw_body_bytes.decode("utf-8")
    payload_to_sign = parts.t + "." + raw_body
    expected = hmac_sha256(secret=WEBHOOK_SECRET, message=payload_to_sign)

    if not constant_time_compare(expected, parts.s):
        raise Unauthorized("Invalid signature")
```

> **Nota:** Usá siempre `constant_time_compare` (timing-safe equals) para evitar timing attacks.

---

## Parte 2 — Tipos de Eventos

El body del webhook tiene esta estructura:

```json
[
  {
    "type": "whatsapp.inbound_message.received",
    "id": "evt_abc123",
    "whatsappInboundMessage": {
      "from": "5491155550001",
      "to": "5491155550099",
      "type": "text",
      "text": { "body": "Hola, qué tal?" },
      "wamid": "wamid.abcdef...",
      "customerProfile": { "name": "María" }
    }
  }
]
```

Los tipos de eventos relevantes:

| `type` | Descripción |
|---|---|
| `whatsapp.inbound_message.received` | Mensaje nuevo del usuario |
| `whatsapp.message.echo` | Mensaje enviado **por el negocio** desde el panel |
| `whatsapp.smb.message.echoes` | Igual que el anterior (variante YCloud SMB) |

Los `echo` son críticos: cuando un humano del equipo responde manualmente, YCloud te avisa con un echo. Tu sistema debe usarlo para activar el **AI Lock** (ver doc 02).

---

## Parte 3 — El Sistema de Debounce con Redis

### El problema que resuelve

Los humanos frecuentemente mandan varios mensajes cortos seguidos:

```
[19:00:01] "Hola"
[19:00:03] "busco zapatillas"
[19:00:05] "de punta"
```

Sin debounce, tu IA respondería 3 veces a 3 preguntas distintas. Con debounce, espera a que el usuario termine y procesa el mensaje completo: `"Hola\nbusco zapatillas\nde punta"`.

### Las 3 claves de Redis

Para cada número de teléfono (`{num}`) se usan 3 claves:

| Clave | Tipo | TTL | Propósito |
|---|---|---|---|
| `buffer:{num}` | Lista | Manual | Cola de mensajes pendientes |
| `timer:{num}` | String | **12 seg** | "Silencio" del usuario. Se resetea con cada mensaje |
| `active_task:{num}` | String | **60 seg** | Previene lanzar 2 workers simultáneos |

### Flujo completo al recibir un mensaje de texto

```
┌─────────────────────────────────────────────────────┐
│  Webhook recibido                                   │
│                                                     │
│  1. RPUSH buffer:{num}  ← agregar al final          │
│     payload: {text, wamid, event_id}                │
│                                                     │
│  2. SETEX timer:{num} 12 "1"  ← (re)iniciar timer  │
│                                                     │
│  3. ¿Existe active_task:{num}?                      │
│     SÍ → return "buffering_updated"  (ya hay worker)│
│     NO → SETEX active_task:{num} 60 "1"             │
│           lanzar_worker_en_background()             │
│           return "buffering_started"                │
└─────────────────────────────────────────────────────┘
```

### El Worker en Background

El worker corre un loop que tiene dos fases:

#### Fase A — Esperar silencio (debounce loop)

```
while true:
    sleep(2 segundos)
    if TTL(timer:{num}) <= 0:
        break   # El usuario paró de escribir → procesar
    # Si TTL > 0, el usuario sigue escribiendo → seguir esperando
```

#### Fase B — Procesar el buffer

```
L = LLEN(buffer:{num})           # ¿Cuántos mensajes hay?
if L == 0: break                 # Buffer vacío, terminar

items = LRANGE(buffer:{num}, 0, L-1)   # Tomar todos
joined_text = JOIN(items, "\n")  # Unir con salto de línea

# Los IDs del ÚLTIMO mensaje se usan para deduplicación
event_id  = items[-1].event_id
wamid     = items[-1].wamid

# Enviar al orchestrator
result = POST("/chat", {text: joined_text, event_id: event_id, ...})

# Procesar y enviar la respuesta al usuario
if result.send:
    send_sequence(result.messages, user_number)

# Limpiar SOLO los mensajes que procesamos (no tocar los nuevos)
LTRIM(buffer:{num}, L, -1)

# ¿Llegaron mensajes nuevos mientras respondíamos?
if LLEN(buffer:{num}) > 0:
    SETEX timer:{num} 5 "1"   # Re-debounce corto de 5s
    continue                   # Volver al loop
else:
    break
```

### Resumen de timings

| Evento | Tiempo |
|---|---|
| Debounce poll (chequeo del timer) | cada **2 seg** |
| Silencio mínimo para procesar | **12 seg** sin mensajes nuevos |
| Lock máximo de la tarea worker | **60 seg** |
| Re-debounce si llegan mensajes tardíos | **5 seg** |

> **Ejemplo:** El usuario manda 3 mensajes en 8 segundos. Cada mensaje resetea el timer a 12s. Cuando para de escribir, el worker detecta que el timer expiró (en el próximo poll de 2s) y procesa los 3 mensajes como uno solo. Tiempo total de espera desde el último mensaje: entre 2 y 14 segundos.

---

## Parte 4 — Mensajes de Audio (sin debounce)

Los audios **no entran al buffer**. Se procesan inmediatamente porque:
- No tiene sentido "acumular" varios audios (cada uno es una idea completa)
- La transcripción ya tiene su propio delay

```
Recibir audio
    ↓
Descargar el archivo de audio desde la URL de YCloud
    ↓
Enviar a OpenAI Whisper (POST /v1/audio/transcriptions, modelo whisper-1)
    ↓
Obtener texto transcripto
    ↓
Enviar al orchestrator con el texto (igual que un mensaje de texto normal)
    ↓
Procesar respuesta y enviar al usuario
```

Lo mismo aplica para imágenes y documentos (sin transcripción, van directos al orchestrator con la URL del media).

---

## Parte 5 — Deduplicación

El orchestrator puede recibir el mismo `event_id` más de una vez (YCloud reintenta el webhook si no obtiene un 2xx). Para evitar responder dos veces:

```
Al recibir un /chat:
    key = "dedup:{event_id}"
    if EXISTS(key):
        return {status: "duplicate", send: false}
    else:
        SET(key, "1", ttl=60 segundos)
        # procesar normalmente
```

El WhatsApp service, al recibir `status: "duplicate"`, ignora silenciosamente (no envía nada).

---

## Diagrama de Flujo Completo (Entrada)

```
YCloud
  │
  │ POST /webhook/ycloud
  ▼
[Verify HMAC] ──── FAIL ──→ 401
  │
  │ OK
  ▼
[Parse event type]
  │
  ├─ "whatsapp.inbound_message.received"
  │     │
  │     ├─ type == "text"
  │     │     │
  │     │     ▼
  │     │  RPUSH buffer + SETEX timer(12s)
  │     │     │
  │     │     └─ [Si no hay worker] → lanzar background worker
  │     │                               └─ debounce loop → procesar
  │     │
  │     ├─ type == "audio"
  │     │     ├─ Descargar audio
  │     │     ├─ Whisper transcripción
  │     │     └─ → orchestrator (directo, sin buffer)
  │     │
  │     └─ type == "image" / "document"
  │           └─ → orchestrator (directo, con URL del media)
  │
  └─ "whatsapp.message.echo" / "whatsapp.smb.message.echoes"
        └─ → orchestrator con event_type="echo"
              └─ Activa AI Lock 24h (ver doc 02)
```
