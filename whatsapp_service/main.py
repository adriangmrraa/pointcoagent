import os
import hmac
import hashlib
import time
import uuid
import asyncio
import redis
import httpx
import structlog
import json
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from ycloud_client import YCloudClient
from sequence_planner import plan_send_actions
from transcription_provider import (
    resolve_transcription_config,
    build_audio_chat_payload,
    audio_format_from_mime,
)

# Initialize config
load_dotenv()

# Config handling
_config_cache = {}

async def get_config(name: str, default: str = None) -> str:
    # 1. Check local cache
    if name in _config_cache:
        return _config_cache[name]
    
    # 2. Check local Environment
    val = os.getenv(name)
    if val:
        _config_cache[name] = val
        return val
        
    # 3. Query Orchestrator
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{ORCHESTRATOR_URL}/admin/internal/credentials/{name}",
                headers={"X-Internal-Token": INTERNAL_API_TOKEN},
                timeout=5.0
            )
            if resp.status_code == 200:
                val = resp.json().get("value")
                if val:
                    _config_cache[name] = val
                    return val
    except Exception as e:
        logger.warning("config_fetch_failed", name=name, error=str(e))
        
    return default

# Initialize startup values (can be overridden later)
YCLOUD_API_KEY = os.getenv("YCLOUD_API_KEY")
YCLOUD_WEBHOOK_SECRET = os.getenv("YCLOUD_WEBHOOK_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Transcripción de audio: OpenRouter NO la revende. Default OpenAI; para sacar el
# gasto de OpenAI, apuntar a Groq (whisper-large-v3). Rollback = limpiar estas vars.
TRANSCRIPTION_BASE_URL = os.getenv("TRANSCRIPTION_BASE_URL", "https://api.openai.com/v1")
TRANSCRIPTION_MODEL = os.getenv("TRANSCRIPTION_MODEL", "whisper-1")
TRANSCRIPTION_API_KEY = os.getenv("TRANSCRIPTION_API_KEY")
INTERNAL_API_TOKEN = os.getenv("INTERNAL_API_TOKEN")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_SERVICE_URL", "http://orchestrator_service:8000")

# Tiempos del bot (ajustables desde EasyPanel sin tocar código, en segundos)
DEBOUNCE_SECONDS = int(os.getenv("DEBOUNCE_SECONDS", "5"))
DELAY_IMAGE_SECONDS = float(os.getenv("DELAY_IMAGE_SECONDS", "1.5"))
DELAY_TEXT_SECONDS = float(os.getenv("DELAY_TEXT_SECONDS", "0.8"))
DELAY_BETWEEN_BUBBLES_SECONDS = float(os.getenv("DELAY_BETWEEN_BUBBLES_SECONDS", "1.0"))

# Initialize structlog
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)
logger = structlog.get_logger()

# Initialize Redis
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# --- Models ---
class OrchestratorMessage(BaseModel):
    part: Optional[int] = None
    total: Optional[int] = None
    text: Optional[str] = None
    imageUrl: Optional[str] = None
    needs_handoff: bool = False
    handoff: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

class OrchestratorResult(BaseModel):
    status: str
    send: bool
    text: Optional[str] = None
    messages: List[OrchestratorMessage] = Field(default_factory=list)

class SendMessage(BaseModel):
    to: str
    text: str

# FastAPI App
app = FastAPI(
    title="WhatsApp Service",
    description="A service to handle WhatsApp interactions and forward them to the orchestrator.",
)

# Metrics
SERVICE_NAME = "whatsapp_service"
REQUESTS = Counter("http_requests_total", "Total Request Count", ["service", "endpoint", "method", "status"])
LATENCY = Histogram("http_request_latency_seconds", "Request Latency", ["service", "endpoint"])

# --- Middleware ---
@app.middleware("http")
async def add_metrics_and_logs(request: Request, call_next):
    start_time = time.time()
    correlation_id = request.headers.get("X-Correlation-Id") or request.headers.get("traceparent")
    response = await call_next(request)
    process_time = time.time() - start_time
    status_code = response.status_code
    REQUESTS.labels(service=SERVICE_NAME, endpoint=request.url.path, method=request.method, status=status_code).inc()
    LATENCY.labels(service=SERVICE_NAME, endpoint=request.url.path).observe(process_time)
    logger.bind(
        service=SERVICE_NAME, correlation_id=correlation_id, status_code=status_code,
        method=request.method, endpoint=request.url.path, latency_ms=round(process_time * 1000, 2)
    ).info("request_completed" if status_code < 400 else "request_failed")
    return response

# --- Helpers ---
async def verify_signature(request: Request):
    signature_header = request.headers.get("ycloud-signature")
    if not signature_header: raise HTTPException(status_code=401, detail="Missing signature header")
    try:
        parts = {k: v for k, v in [p.split("=") for p in signature_header.split(",")]}
        t, s = parts.get("t"), parts.get("s")
    except: raise HTTPException(status_code=401, detail="Invalid signature format")
    if not t or not s: raise HTTPException(status_code=401, detail="Missing timestamp or signature")
    if abs(time.time() - int(t)) > 300: raise HTTPException(status_code=401, detail="Timestamp out of tolerance")
    raw_body = await request.body()
    signed_payload = f"{t}.{raw_body.decode('utf-8')}"
    expected = hmac.new(YCLOUD_WEBHOOK_SECRET.encode("utf-8"), signed_payload.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, s): raise HTTPException(status_code=401, detail="Invalid signature")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10),
       retry=retry_if_exception_type(httpx.HTTPError))
async def forward_to_orchestrator(payload: dict, headers: dict):
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0)) as client:
        response = await client.post(f"{ORCHESTRATOR_URL}/chat", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

async def transcribe_audio(audio_url: str, correlation_id: str, mime_type: str = None) -> Optional[str]:
    """Descarga el audio de YCloud y lo transcribe. Tres modos según TRANSCRIPTION_BASE_URL:
    OpenAI/Groq (Whisper multipart) u OpenRouter (modelo multimodal vía chat)."""
    # Resolver key: la propia de transcripción tiene prioridad, si no cae a OpenAI
    trans_key = await get_config("TRANSCRIPTION_API_KEY", TRANSCRIPTION_API_KEY)
    openai_key = await get_config("OPENAI_API_KEY", OPENAI_API_KEY)
    cfg = resolve_transcription_config(TRANSCRIPTION_BASE_URL, TRANSCRIPTION_MODEL, trans_key, openai_key)
    if not cfg["api_key"]:
        logger.error("missing_transcription_api_key", provider=cfg["provider"],
                     note="Falta TRANSCRIPTION_API_KEY u OPENAI_API_KEY")
        return None

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            # 1. Download audio
            audio_res = await client.get(audio_url)
            audio_res.raise_for_status()
            audio_data = audio_res.content

            logger.info("transcription_provider_selected", provider=cfg["provider"],
                        mode=cfg["mode"], model=cfg["model"], correlation_id=correlation_id)

            # 2a. Modo OpenRouter: audio en base64 dentro de /chat/completions
            if cfg["mode"] == "chat_audio":
                import base64
                audio_b64 = base64.b64encode(audio_data).decode()
                fmt = audio_format_from_mime(mime_type)
                payload = build_audio_chat_payload(cfg["model"], audio_b64, fmt)
                headers = {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}
                resp = await client.post(cfg["chat_url"], headers=headers, json=payload)
                resp.raise_for_status()
                choices = (resp.json() or {}).get("choices") or []
                if not choices:
                    logger.error("transcription_empty_choices", provider=cfg["provider"], correlation_id=correlation_id)
                    return None
                return (choices[0].get("message") or {}).get("content")

            # 2b. Modo Whisper (OpenAI / Groq): multipart a /audio/transcriptions
            files = {"file": ("audio.ogg", audio_data, "audio/ogg")}
            headers = {"Authorization": f"Bearer {cfg['api_key']}"}
            data = {"model": cfg["model"]}
            trans_res = await client.post(cfg["url"], headers=headers, files=files, data=data)
            trans_res.raise_for_status()
            return trans_res.json().get("text")
    except Exception as e:
        logger.error("transcription_failed", error=str(e), correlation_id=correlation_id)
        return None

async def _notify_typing(inbound_id: str, business_number: str, correlation_id: str):
    """Marca leído y muestra 'escribiendo...' apenas llega el mensaje del usuario,
    para que la espera del procesamiento se sienta más corta."""
    try:
        v_ycloud = await get_config("YCLOUD_API_KEY", YCLOUD_API_KEY)
        if not v_ycloud or not inbound_id:
            return
        client = YCloudClient(v_ycloud, business_number)
        await client.mark_as_read(inbound_id, correlation_id)
        await client.typing_indicator(inbound_id, correlation_id)
    except Exception as e:
        logger.warning("typing_on_receipt_failed", error=str(e), correlation_id=correlation_id)

async def send_sequence(messages: List[OrchestratorMessage], user_number: str, business_number: str, inbound_id: str, correlation_id: str):
    v_ycloud = await get_config("YCLOUD_API_KEY", YCLOUD_API_KEY)
    client = YCloudClient(v_ycloud, business_number)

    try:
        await client.mark_as_read(inbound_id, correlation_id)
        await client.typing_indicator(inbound_id, correlation_id)
    except: pass

    # Plan mínimo de mensajes: imagen+texto viajan juntos como caption
    # (menos mensajes facturables por Meta y secuencia más rápida).
    actions = plan_send_actions([{"text": m.text, "imageUrl": m.imageUrl} for m in messages])
    logger.info("send_plan_ready", bubbles_in=len(messages), messages_out=len(actions), correlation_id=correlation_id)

    for i, action in enumerate(actions):
        try:
            try: await client.typing_indicator(inbound_id, correlation_id)
            except: pass

            if action["type"] == "image":
                await asyncio.sleep(DELAY_IMAGE_SECONDS)
                await client.send_image(user_number, action["url"], correlation_id, caption=action.get("caption"))
            else:
                await asyncio.sleep(DELAY_TEXT_SECONDS)
                await client.send_text(user_number, action["text"], correlation_id)

            # Pausa breve entre mensajes para preservar el orden de entrega
            if i < len(actions) - 1:
                await asyncio.sleep(DELAY_BETWEEN_BUBBLES_SECONDS)

        except Exception as e:
            logger.error("sequence_step_error", error=str(e), correlation_id=correlation_id)

# --- Background Task ---
async def process_user_buffer(from_number: str, business_number: str, customer_name: Optional[str], event_id: str, provider_message_id: str):
    buffer_key, timer_key, lock_key = f"buffer:{from_number}", f"timer:{from_number}", f"active_task:{from_number}"
    correlation_id = str(uuid.uuid4())
    log = logger.bind(correlation_id=correlation_id, from_number=from_number[-4:])
    try:
        while True:
            # 1. Debounce Phase: Wait until user stopped typing
            while True:
                await asyncio.sleep(1)
                if redis_client.ttl(timer_key) <= 0: break
            
            # 2. Atomic Fetch: How many messages are we starting with?
            L = redis_client.llen(buffer_key)
            if L == 0: break
            
            raw_items = redis_client.lrange(buffer_key, 0, L-1)
            parsed_items = []
            for item in raw_items:
                try:
                    parsed_items.append(json.loads(item))
                except:
                    # Fallback for legacy items or unexpected formats
                    parsed_items.append({"text": item, "wamid": provider_message_id, "event_id": event_id})
            
            joined_text = "\n".join([i["text"] for i in parsed_items])
            # We use the LAST message IDs to identify this batch in the orchestrator (deduplication)
            current_event_id = parsed_items[-1].get("event_id") or event_id
            current_wamid = parsed_items[-1].get("wamid") or provider_message_id

            inbound_event = {
                "provider": "ycloud", 
                "event_id": current_event_id, 
                "provider_message_id": current_wamid,
                "from_number": from_number, "to_number": business_number, "text": joined_text, "customer_name": customer_name,
                "event_type": "whatsapp.inbound_message.received", "correlation_id": correlation_id
            }
            headers = {"X-Correlation-Id": correlation_id}
            if INTERNAL_API_TOKEN: headers["X-Internal-Token"] = INTERNAL_API_TOKEN
                 
            log.info("forwarding_to_orchestrator", text_preview=joined_text[:50])
            raw_res = await forward_to_orchestrator(inbound_event, headers)
            log.info("orchestrator_response_received", status=raw_res.get("status"), send=raw_res.get("send"))
            
            try:
                orch_res = OrchestratorResult(**raw_res)
            except Exception as e:
                log.error("orchestrator_parse_error", error=str(e), raw=raw_res)
                # Cleanup and break to avoid stuck state
                redis_client.ltrim(buffer_key, L, -1)
                break

            if orch_res.status == "duplicate":
                log.info("ignoring_duplicate_response")
                redis_client.ltrim(buffer_key, L, -1)
                break

            if orch_res.send:
                if not YCLOUD_API_KEY:
                    log.error("missing_ycloud_api_key", note="Cannot send sequence without API key")
                else:
                    msgs = orch_res.messages
                    if not msgs and orch_res.text:
                        msgs = [OrchestratorMessage(text=orch_res.text)]
                    
                    if msgs:
                        img_count = len([m for m in msgs if m.imageUrl])
                        log.info("starting_send_sequence", count=len(msgs), images_found=img_count)
                        await send_sequence(msgs, from_number, business_number, current_event_id, correlation_id)
            
            # 3. ATOMIC TRIM: Remove only the messages we just processed
            redis_client.ltrim(buffer_key, L, -1)
            
            # 4. LOOP CHECK: If more messages arrived during the sequence, process them immediately
            if redis_client.llen(buffer_key) == 0:
                break
            else:
                log.info("new_messages_while_responding", remaining=redis_client.llen(buffer_key))
                # Reset timer to force a small fresh debounce for the new messages
                redis_client.setex(timer_key, DEBOUNCE_SECONDS, "1")

    except Exception as e:
        log.error("buffer_process_error", error=str(e))
    finally:
        # Buffer is handled by ltrim inside the loop or error
        for k in [lock_key, timer_key]:
            try:
                redis_client.delete(k)
            except:
                pass

# --- Endpoints ---
@app.get("/metrics")
def metrics(): return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/ready")
def ready():
    if not YCLOUD_WEBHOOK_SECRET: raise HTTPException(status_code=503, detail="Configuration missing")
    return {"status": "ok"}

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/webhook/ycloud")
async def ycloud_webhook(request: Request):
    logger.info("webhook_hit", headers=str(request.headers))
    await verify_signature(request)
    correlation_id = request.headers.get("traceparent") or str(uuid.uuid4())
    try: body = await request.json()
    except: raise HTTPException(status_code=400, detail="Invalid JSON")
    
    event = body[0] if isinstance(body, list) and body else body
    event_type = event.get("type")
    
    # --- 1. Handle Inbound Messages ---
    if event_type == "whatsapp.inbound_message.received":
        msg = event.get("whatsappInboundMessage", {})
        from_n, to_n, name = msg.get("from"), msg.get("to"), msg.get("customerProfile", {}).get("name")
        msg_type = msg.get("type")
        
        # A. Text Messages -> Buffer (Debounce)
        if msg_type == "text":
            text = msg.get("text", {}).get("body")
            if text:
                buffer_key, timer_key, lock_key = f"buffer:{from_n}", f"timer:{from_n}", f"active_task:{from_n}"
                # Store message as JSON to preserve IDs for the atomic loop
                redis_client.rpush(buffer_key, json.dumps({
                    "text": text,
                    "wamid": msg.get("wamid") or event.get("id"),
                    "event_id": event.get("id")
                }))
                redis_client.setex(timer_key, DEBOUNCE_SECONDS, "1")

                # Feedback inmediato mientras el bot piensa
                asyncio.create_task(_notify_typing(msg.get("wamid") or event.get("id"), to_n, correlation_id))

                if not redis_client.get(lock_key):
                    redis_client.setex(lock_key, 60, "1")
                    asyncio.create_task(process_user_buffer(from_n, to_n, name, event.get("id"), msg.get("wamid") or event.get("id")))
                    return {"status": "buffering_started", "correlation_id": correlation_id}
                return {"status": "buffering_updated", "correlation_id": correlation_id}
        
        # B. Media Messages -> Immediate Forward (No Buffer)
        media_list = []
        text_content = None
        
        if msg_type == "image":
            node = msg.get("image", {})
            text_content = node.get("caption")
            media_list.append({
                "type": "image", 
                "url": node.get("link"), 
                "mime_type": node.get("mime_type"),
                "provider_id": node.get("id")
            })
            
        elif msg_type == "document":
            node = msg.get("document", {})
            text_content = node.get("caption")
            media_list.append({
                "type": "document", 
                "url": node.get("link"), 
                "mime_type": node.get("mime_type"), 
                "file_name": node.get("filename"),
                "provider_id": node.get("id")
            })
            
        elif msg_type == "audio":
            node = msg.get("audio", {})
            media_list.append({
                "type": "audio", 
                "url": node.get("link"), 
                "mime_type": node.get("mime_type"),
                "provider_id": node.get("id")
            })
            # Transcription
            if node.get("link"):
                logger.info("audio_received_starting_transcription", correlation_id=correlation_id)
                # La transcripción tarda: mostrar "escribiendo..." mientras tanto
                asyncio.create_task(_notify_typing(msg.get("wamid") or event.get("id"), to_n, correlation_id))
                transcription = await transcribe_audio(node.get("link"), correlation_id, node.get("mime_type"))
                if transcription:
                     text_content = transcription
                     
        if media_list:
             # Construct payload compatible with InboundChatEvent + Media extension
             payload = {
                "provider": "ycloud", 
                "event_id": event.get("id"), 
                "provider_message_id": msg.get("wamid") or event.get("id"),
                "from_number": from_n, 
                "to_number": to_n, 
                "text": text_content, # Can be None/null
                "customer_name": name,
                "event_type": "whatsapp.inbound_message.received", 
                "correlation_id": correlation_id,
                "media": media_list
             }
             headers = {"X-Correlation-Id": correlation_id}
             if INTERNAL_API_TOKEN: headers["X-Internal-Token"] = INTERNAL_API_TOKEN
             
             # Send to Orchestrator and Process Response
             try:
                 raw_res = await forward_to_orchestrator(payload, headers)
                 
                 orch_res = OrchestratorResult(**raw_res)
                 if orch_res.send:
                     if not YCLOUD_API_KEY:
                         logger.error("missing_ycloud_api_key_media_reply")
                     else:
                         msgs = orch_res.messages
                         if not msgs and orch_res.text:
                             msgs = [OrchestratorMessage(text=orch_res.text)]
                         
                         if msgs:
                             await send_sequence(msgs, from_n, to_n, event.get("id"), correlation_id)
             except Exception as e:
                 logger.error("media_response_processing_error", error=str(e))
                 
             return {"status": "media_and_response_processed", "count": len(media_list)}
             
        return {"status": "ignored_type_or_empty", "type": msg_type}

    # --- 2. Handle Echoes (Manual Messages) ---
    elif event_type == "whatsapp.message.echo" or event_type == "whatsapp.smb.message.echoes":
        logger.info("echo_received", correlation_id=correlation_id, evt_type=event_type)
        msg = event.get("whatsappMessage", {}) or event.get("message", {})
        
        user_phone = msg.get("to")
        bot_phone = msg.get("from")
        
        text = None
        msg_type = msg.get("type")
        
        if msg_type == "text":
            text = msg.get("text", {}).get("body")
        elif msg_type == "audio":
            text = "[Audio enviado]"
        elif msg_type == "image":
            text = msg.get("image", {}).get("caption") or "[Imagen enviada]"
        elif msg_type == "document":
            text = msg.get("document", {}).get("caption") or "[Documento enviado]"
        elif msg_type == "video":
             text = msg.get("video", {}).get("caption") or "[Video enviado]"
        
        # If we have text (real or fallback) and a user phone, forward it
        if text and user_phone:
             payload = {
                "provider": "ycloud", 
                "event_id": event.get("id"), 
                "provider_message_id": msg.get("wamid") or event.get("id"),
                "from_number": user_phone,     # Ensuring this maps to 'external_user_id' in DB
                "to_number": bot_phone,
                "text": text,
                "event_type": "whatsapp.message.echo", # Standardize for Orchestrator
                "correlation_id": correlation_id
             }
             headers = {"X-Correlation-Id": correlation_id}
             if INTERNAL_API_TOKEN: headers["X-Internal-Token"] = INTERNAL_API_TOKEN
             
             try:
                 await forward_to_orchestrator(payload, headers)
                 return {"status": "echo_forwarded", "type": msg_type}
             except Exception as e:
                 logger.error("echo_forward_failed", error=str(e))
                 return {"status": "error_forwarding_echo"}
                 
    return {"status": "ignored_event_type", "type": event_type}

@app.post("/messages/send")
async def send_message(message: SendMessage, request: Request):
    """Internal endpoint for sending manual messages from orchestrator."""
    token = request.headers.get("X-Internal-Token")
    if token != INTERNAL_API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    correlation_id = request.headers.get("X-Correlation-Id") or str(uuid.uuid4())
    # Retrieve config
    v_ycloud = await get_config("YCLOUD_API_KEY", YCLOUD_API_KEY)
    # We need to know which business number to use - for now assume default or pass in body if model updated
    # To keep it simple for now, we use the default env var logic inside YCloudClient via send_sequence or re-instantiate
    # Ideally SendMessage model should include 'from_number' (business number)
    
    # Since SendMessage is simple (to, text), we try to get a business number from config or context
    # But YCloudClient needs it.
    # Hack: We initialize YCloudClient with a dummy if needed, but it really needs the sender ID.
    # Let's check send_sequence usage: client = YCloudClient(v_ycloud, business_number)
    
    # IMPROVEMENT: The request should probably provide the separate business number/ID
    # For this MVP, let's assume global YCloud config unless passed
    
    try:
        # Re-use send_sequence logic but for a single message
        # We need to wrap it as OrchestratorMessage
        orch_msg = OrchestratorMessage(text=message.text)
        
        # We need the 'from' number (the bot's number). 
        # Since we don't have it in the simple body, we might need to assume it's the one in ENV or context.
        # However, for multi-tenant, orchestrator MUST tell us.
        # Let's inspect headers or just rely on YCloudClient to default if allowed.
        # Actually YCloudClient requires `from_phone_number`. 
        
        # Updated Logic: We will parse `from_number` from query param or header if available, or fetch from config
        business_number = request.query_params.get("from_number")
        if not business_number:
            # Fallback to env or fetch
             business_number = await get_config("YCLOUD_Phone_Number_ID") # Placeholder
        
        if not business_number:
             # Basic fallback
             business_number = "default"

        # Initialize Client
        client = YCloudClient(v_ycloud, business_number)
        
        # Send
        await client.send_text(message.to, message.text, correlation_id)
        return {"status": "sent", "correlation_id": correlation_id}
        
    except Exception as e:
        logger.error("manual_send_failed", error=str(e), correlation_id=correlation_id)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

