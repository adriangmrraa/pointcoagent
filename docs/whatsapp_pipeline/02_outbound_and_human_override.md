# 02 · Pipeline de Salida, Human Override y AI Lock

> **Audiencia:** programadores. El código de referencia está en `whatsapp_service/main.py` y `orchestrator_service/main.py`. Los conceptos son agnósticos al lenguaje.

---

## Visión General

Una vez que el motor de IA genera una respuesta, hay dos sistemas críticos:

1. **Send Sequence:** cómo se entregan las burbujas al usuario de forma natural (con typing indicators, delays y splits de texto)
2. **Human Override / AI Lock:** cómo el bot "se calla" cuando un humano toma el control de la conmversación

---

## Parte 1 — El Formato de Respuesta del Orchestrator

El orchestrator devuelve siempre un JSON estructurado:

```json
{
  "status": "ok",
  "send": true,
  "messages": [
    { "text": "Hola! Acá tenés opciones:", "imageUrl": null },
    { "text": "Zapatillas Grishko\n$55.000\nhttps://...", "imageUrl": "https://cdn.../img.jpg" },
    { "text": "Si querés ver más, entrá a la web.", "imageUrl": null }
  ]
}
```

| Campo | Valores posibles | Significado |
|---|---|---|
| `status` | `ok`, `duplicate`, `ignored`, `error` | Estado del procesamiento |
| `send` | `true` / `false` | Si hay que enviar algo al usuario |
| `messages` | Lista de `{text, imageUrl}` | Las "burbujas" a enviar en orden |

Si `send == false` (ej: `status="ignored"` por human override), **no se envía nada**. La conversación se mantiene en silencio.

---

## Parte 2 — Send Sequence: Envío Natural por Burbujas

### El problema que resuelve

WhatsApp es una app de mensajería. Las respuestas largas en un solo mensaje se ven robotizadas. El sistema envía las respuestas como **múltiples burbujas separadas**, con delay entre ellas para simular una persona escribiendo.

### Flujo para cada burbuja `{text, imageUrl}`

```
Para cada burbuja en messages[]:

  Si tiene imageUrl:
    → typing_indicator()        # Muestra "escribiendo..."
    → sleep(4 segundos)         # Da tiempo para que cargue la imagen
    → send_image(url)
    → mark_as_read()

  Si tiene text:
    ¿El texto tiene más de 400 caracteres?
      SÍ → dividir por oraciones/puntos (split seguro)
      NO → enviarlo como está

    Para cada parte del texto:
      → typing_indicator()
      → sleep(1.5 segundos)     # Simula tiempo de escritura
      → send_text(parte)
      → mark_as_read()

  → sleep(2 segundos)           # Pausa entre burbujas
```

### Timings de `send_sequence`

| Acción | Delay |
|---|---|
| Antes de enviar imagen | **4 seg** |
| Antes de enviar texto | **1.5 seg** |
| Entre burbuja y burbuja | **2 seg** |

> **Por qué 4 segundos para imágenes:** Las imágenes tardan en cargar en el cliente. Si enviás el texto explicativo antes de que la imagen aparezca, el usuario la ve "después" del contexto y la experiencia se rompe. El delay asegura que la imagen llega primero.

### Safety Splitter (Layer 2)

Si el orchestrator por alguna razón devuelve un texto mayor a 400 caracteres en una sola burbuja, el `send_sequence` lo parte automáticamente antes de enviarlo:

```
if len(text) > 400:
    parts = split_by_sentence_end(text)  # Divide en "." "!" "?"
    # Luego agrupa las partes en chunks de máx 400 chars
    for chunk in chunks:
        enviar(chunk)
```

Esto es una salvaguarda de última línea: idealmente el orchestrator ya genera burbujas cortas por diseño (instrucción en el system prompt).

### Pseudocódigo completo

```
function send_sequence(messages, user_number, business_number, inbound_id):
    client = YCloudClient(api_key, business_number)

    # Marcar el mensaje entrante como leído y mostrar "escribiendo..."
    try:
        client.mark_as_read(inbound_id)
        client.typing_indicator(inbound_id)
    catch: pass  # No crítico si falla

    for msg in messages:
        # --- Imagen ---
        if msg.imageUrl:
            try: client.typing_indicator(inbound_id)
            catch: pass
            sleep(4)
            client.send_image(user_number, msg.imageUrl)
            try: client.mark_as_read(inbound_id)
            catch: pass

        # --- Texto ---
        if msg.text:
            parts = [msg.text]
            if len(msg.text) > 400:
                parts = safe_split(msg.text, max_len=400)

            for part in parts:
                try: client.typing_indicator(inbound_id)
                catch: pass
                sleep(1.5)
                client.send_text(user_number, part)
                try: client.mark_as_read(inbound_id)
                catch: pass

        sleep(2)  # Pausa entre burbujas
```

### API de YCloud usada

Todas las llamadas van a `https://api.ycloud.com/v2`:

| Acción | Endpoint | Body |
|---|---|---|
| Enviar texto | `POST /whatsapp/messages/sendDirectly` | `{from, to, type:"text", text:{body}}` |
| Enviar imagen | `POST /whatsapp/messages/sendDirectly` | `{from, to, type:"image", image:{link}}` |
| Marcar como leído | `POST /whatsapp/inboundMessages/{id}/markAsRead` | `{}` |
| Indicador de escritura | `POST /whatsapp/inboundMessages/{id}/typingIndicator` | `{}` |

Todas las llamadas llevan el header `X-API-Key: {YCLOUD_API_KEY}`.

### Reintentos automáticos

Las llamadas a YCloud tienen un retry automático:
- **Máx intentos:** 2
- **Espera entre reintentos:** exponencial (1s, luego hasta 4s)
- **Solo reintenta en:** errores HTTP (timeouts, 5xx). No reintenta en 4xx (errores de configuración).

---

## Parte 3 — Human Override: El AI Lock

### Concepto

Cuando un humano del equipo interviene en una conversación (por ejemplo, para hacer el seguimiento de un fitting o resolver un problema complejo), el bot debe **callarse completamente** para no interrumpir al humano.

Este mecanismo se llama **Human Override** o **AI Lock**.

### ¿Cuándo se activa?

Se activa en dos situaciones:

**A) La tool `derivhumano` se ejecuta exitosamente**

Cuando el agente de IA decide derivar la conversación a un humano (porque el usuario lo pidió, tiene un problema complejo, quiere un fitting, etc.), la tool hace:

```sql
UPDATE chat_conversations
SET
    human_override_until = NOW() + INTERVAL '24 hours',
    status = 'human_handling'
WHERE id = {conversation_id}
```

**B) Llega un echo de WhatsApp**

Cuando alguien del equipo responde manualmente desde el panel de WhatsApp Business, YCloud envía un evento `whatsapp.smb.message.echoes` (o `whatsapp.message.echo`). Al recibirlo, el orchestrator activa el mismo lock:

```sql
UPDATE chat_conversations
SET human_override_until = NOW() + INTERVAL '24 hours'
WHERE external_user_id = {user_phone}
```

Esto es automático: **si el equipo habla, el bot se calla sin que nadie lo configure**.

### ¿Cuánto dura el lock?

**24 horas exactas** desde el momento de activación. Después se levanta automáticamente.

> **Diseño intencional:** 24 horas es suficiente para que el equipo resuelva el caso, y el bot retoma sin intervención manual.

### ¿Cómo se respeta el lock?

Al inicio de **cada request** al endpoint `/chat` del orchestrator, antes de ejecutar la IA:

```
function handle_chat(event):

    conversation = get_or_create_conversation(event.from_number)

    # ← CHEQUEO DEL LOCK (primero que todo)
    if conversation.human_override_until is not null:
        if now() < conversation.human_override_until:
            return {
                status: "ignored",
                send: false
            }

    # Si llega aquí, el lock no está activo → continuar con la IA
    run_ai_agent(event)
```

El WhatsApp service, al recibir `status: "ignored"`, **no envía nada**. El usuario puede seguir escribiendo, pero el bot permanece en silencio total hasta que el lock expire.

### Diagrama del AI Lock

```
Usuario escribe → whatsapp_service → orchestrator /chat
                                          │
                                    ¿human_override_until < now()?
                                          │
                    ┌─────────────────────┴──────────────────────┐
                    │ SÍ (lock activo)                           │ NO (libre)
                    ▼                                             ▼
             {status: "ignored",                        Ejecutar agente IA
              send: false}                              → generar respuesta
                    │                                             │
                    ▼                                             ▼
         WhatsApp Service:                              WhatsApp Service:
         no envía nada                                  send_sequence(messages)
         (silencio total)
```

### Flujo completo de derivación + lock

```
1. Usuario: "Quiero hablar con alguien"
2. IA detecta la intención → llama tool `derivhumano`
3. derivhumano():
   a. Envía email SMTP al equipo con resumen del caso
   b. UPDATE chat_conversations SET human_override_until = NOW() + 24h
   c. Retorna mensaje de cierre para la IA
4. IA genera el mensaje de despedida: "➡ Te derivamos con una asesora..."
5. send_sequence() envía ese mensaje al usuario
6. LOCK ACTIVO: los próximos 24h, cualquier mensaje del usuario es ignorado
7. El equipo contacta al usuario directamente por WhatsApp
8. Cuando el equipo responde → echo → lock se renueva otros 24h
9. Después de 24h sin actividad del equipo → lock expira → bot retoma
```

---

## Parte 4 — Los Echoes: Detección de Intervención Humana

### Qué es un echo

Cuando alguien del equipo escribe desde el **panel de WhatsApp Business** (no desde la app personal), YCloud notifica a tu webhook con un evento tipo echo:

```json
{
  "type": "whatsapp.smb.message.echoes",
  "whatsappMessage": {
    "from": "5491155550099",   ← número del NEGOCIO (bot)
    "to": "5491155550001",     ← número del CLIENTE
    "type": "text",
    "text": { "body": "Hola! Ya te contacto en unos minutos." }
  }
}
```

### Cómo procesarlo

```
Recibir echo
    │
    ├─ Extraer: user_phone = msg.to, bot_phone = msg.from
    │
    ├─ Construir payload para /chat:
    │    {
    │      from_number: user_phone,   # ← el ID de la conversación
    │      to_number: bot_phone,
    │      text: "[contenido]",
    │      event_type: "whatsapp.message.echo"
    │    }
    │
    └─ POST al orchestrator /chat
          └─ El orchestrator detecta event_type=="echo"
                └─ Activa human_override_until = NOW() + 24h
                └─ Retorna {status: "ignored", send: false}
```

> **Crítico:** El echo NO requiere respuesta al usuario. Solo sirve para activar el AI lock. Por eso el orchestrator responde `send: false`.

---

## Parte 5 — Endpoint Interno: `/messages/send`

El orchestrator puede enviar mensajes proactivos al usuario (por ejemplo, notificaciones) sin esperar a que el usuario escriba. Para esto llama al endpoint interno del whatsapp_service:

```
POST /messages/send
Headers:
  X-Internal-Token: {INTERNAL_API_TOKEN}

Body:
  {
    "to": "5491155550001",
    "text": "Tu pedido está en camino!"
  }
```

Este endpoint está protegido con token interno. Solo los servicios de la red interna pueden llamarlo.

---

## Parte 6 — Seguridad entre Servicios

Toda comunicación **interna** (entre microservicios) se protege con un token compartido:

```
# WhatsApp Service → Orchestrator
Header: X-Internal-Token: {INTERNAL_API_TOKEN}

# Orchestrator → WhatsApp Service /messages/send
Header: X-Internal-Token: {INTERNAL_API_TOKEN}
```

Si el token no coincide → **401 Unauthorized**.

El `INTERNAL_API_TOKEN` **no** es el mismo que las API keys de OpenAI o YCloud. Es una clave interna generada para proteger la comunicación entre tus propios servicios.

---

## Resumen de Variables de Entorno Necesarias

| Variable | Dónde se usa | Descripción |
|---|---|---|
| `YCLOUD_API_KEY` | whatsapp_service | API key de YCloud para enviar mensajes |
| `YCLOUD_WEBHOOK_SECRET` | whatsapp_service | Secret para verificar firma HMAC |
| `OPENAI_API_KEY` | whatsapp_service | Para transcripción Whisper |
| `INTERNAL_API_TOKEN` | ambos servicios | Token de comunicación interna |
| `ORCHESTRATOR_SERVICE_URL` | whatsapp_service | URL del orchestrator (ej: `http://orchestrator:8000`) |
| `REDIS_URL` | whatsapp_service | URL de Redis para buffer y timers |
| `HANDOFF_EMAIL` | orchestrator | Email destino para derivaciones |
| `SMTP_HOST/PORT/USER/PASS` | orchestrator | Config SMTP para enviar emails de derivación |
