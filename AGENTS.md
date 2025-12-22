# AGENTS.md: Guía para Agentes de IA

Este documento centraliza la información necesaria para que los agentes de inteligencia artificial comprendan y ejecuten tareas en este repositorio.

## Descripción del Proyecto

Este proyecto es una implementación de un agente de chat para WhatsApp, diseñado para interactuar con la plataforma de e-commerce Tienda Nube. El sistema utiliza una arquitectura de microservicios distribuida con LangChain (v0.1.0) para la orquestación.

## Arquitectura de Microservicios

El sistema se compone de dos microservicios principales coordinados mediante `docker-compose.yml`:

1.  **`whatsapp_service` (Puerta de Enlace & Percepción)**
    *   **Propósito:** Interfaz con YCloud (WhatsApp).
    *   **Buffering Inteligente:** Ventana de **16 segundos** para agrupar ráfagas de mensajes del usuario.
    *   **Audio (Whisper):** Detecta notas de voz, las descarga de YCloud y las transcribe usando OpenAI Whisper (`whisper-1`) antes de enviarlas al orquestador.
    *   **Entrega Secuencial:** Envía ráfagas de burbujas ("Burst Format") con indicadores de escritura y esperas de **4 segundos** para una experiencia humana.
    *   **Tecnologías:** Python, FastAPI, Redis, requests.

2.  **`orchestrator_service` (El Cerebro)**
    *   **Propósito:** Procesa la lógica de negocio y decide qué herramientas usar.
    *   **Modelo:** GPT-4o-mini (especializado en JSON estructurado).
    *   **Estrategia Híbrida de Herramientas:**
        *   **Directas (HTTP):** `productsq`, `productsq_category`, `orders` (Consulta directa a la API de Tienda Nube para velocidad).
        *   **MCP (n8n Bridge):** `cupones_list`, `sendemail` (Vía puente manual HTTP/SSE que soporta handshakes stateful y persistencia de sesión).
    *   **Reglas de Negocio:**
        *   **Ventaneo:** Máximo 3-4 productos por turno con link a la web.
        *   **Veracidad:** Descripciones textuales (extraídas palabra por palabra de la tienda).
        *   **Normalización:** Corrección automática de errores ortográficos del cliente (ej: "matatarsiana" -> "metatarsianas").
    *   **Seguridad:** Redis Lock de **80 segundos** por usuario para evitar colisiones y duplicados.
    *   **Tecnologías:** LangChain 0.1.0, PostgreSQL (historial/dedup), Redis.

## Stack Tecnológico

*   **Lenguaje:** Python 3.11
*   **Orquestación:** LangChain 0.1.0 (PINNED)
*   **Bases de Datos:**
    *   Redis: Buffering, Caching y Locking de concurrencia.
    *   PostgreSQL: Persistencia de mensajes y deduplicación.
*   **Seguridad:** Firma HMAC para webhooks y `X-Internal-Token` para red cerrada.

## Lógica de Conversación

1.  **Entrada:** `whatsapp_service` recibe texto o audio. Si es audio, lo transcribe.
2.  **Buffer:** Se acumulan mensajes en Redis durante un período de silencio de 2s (máx 16s total).
3.  **Chat:** El Orquestador recibe el bloque de texto, consulta el historial en Postgres y ejecuta tools. Genera una respuesta JSON con una lista de mensajes.
4.  **Salida:** `whatsapp_service` recorre la lista enviando imagen y texto por separado con delays de 4s para un efecto natural.

## Variables Críticas
Asegúrate de que ambos servicios tengan cargadas:
- `OPENAI_API_KEY` (Para GPT-4 y Whisper)
- `TIENDANUBE_SERVICE_URL` y tokens correspondientes.
- `YCLOUD_API_KEY` y `YCLOUD_WEBHOOK_SECRET`.
- `INTERNAL_API_TOKEN`.
