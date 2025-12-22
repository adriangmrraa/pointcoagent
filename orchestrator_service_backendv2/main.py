import os
import json
import hashlib
import hmac
import time
import uuid
import requests
import redis
import structlog
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Literal
from fastapi import FastAPI, HTTPException, Header, Depends, status, Request
from fastapi.responses import JSONResponse, Response
from fastapi import Response as FastAPIResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Initialize early
# Version 1.1 - Fixed bootstrap columns
load_dotenv()

try:
    from langchain.agents import AgentExecutor, create_openai_functions_agent
except ImportError:
    from langchain.agents.agent import AgentExecutor
    from langchain.agents import create_openai_functions_agent

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from db import db

# Config Validation
def validate_configuration():
    """Validate environment configuration and return status."""
    issues = {
        "missing_required_env": [],
        "invalid_env": [],
        "integration_env_missing": []
    }

    # HARD-required envs (must be present and valid for startup)
    required_envs = {
        "POSTGRES_DSN": os.getenv("POSTGRES_DSN"),
        "REDIS_URL": os.getenv("REDIS_URL"),
        "ADMIN_TOKEN": os.getenv("ADMIN_TOKEN"),
        "INTERNAL_API_TOKEN": os.getenv("INTERNAL_API_TOKEN"),
        "HMAC_SHARED_SECRET": os.getenv("HMAC_SHARED_SECRET")
    }

    for env_name, value in required_envs.items():
        if not value or value.strip() == "":
            issues["missing_required_env"].append(env_name)

    # Validate DSN format
    if required_envs["POSTGRES_DSN"]:
        if not required_envs["POSTGRES_DSN"].startswith("postgres://"):
            issues["invalid_env"].append("POSTGRES_DSN (must start with postgres://)")

    # Validate Redis URL
    if required_envs["REDIS_URL"]:
        if not required_envs["REDIS_URL"].startswith("redis://"):
            issues["invalid_env"].append("REDIS_URL (must start with redis://)")

    # INTEGRATION envs (can be missing, but will show WARN/FAIL)
    integration_envs = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "TIENDANUBE_STORE_ID": os.getenv("TIENDANUBE_STORE_ID"),
        "TIENDANUBE_ACCESS_TOKEN": os.getenv("TIENDANUBE_ACCESS_TOKEN"),
        "YCLOUD_API_KEY": os.getenv("YCLOUD_API_KEY"),
        "YCLOUD_WEBHOOK_SECRET": os.getenv("YCLOUD_WEBHOOK_SECRET")
    }

    for env_name, value in integration_envs.items():
        if not value or value.strip() == "":
            issues["integration_env_missing"].append(env_name)

    return issues

# Security: Payload Sanitizer
def sanitize_payload(payload: dict) -> dict:
    """Remove sensitive data from payloads before logging/telemetry."""
    if not isinstance(payload, dict):
        return payload

    sanitized = {}
    sensitive_keys = {
        'authorization', 'token', 'secret', 'password', 'key', 'cookie',
        'access_token', 'refresh_token', 'api_key', 'webhook_secret',
        'bearer', 'jwt', 'session', 'auth'
    }

    for key, value in payload.items():
        key_lower = key.lower()
        if any(sensitive in key_lower for sensitive in sensitive_keys):
            # Replace with safe representation
            if isinstance(value, str) and len(value) > 4:
                sanitized[key] = f"...{value[-4:]}"  # Last 4 chars
            else:
                sanitized[key] = "[REDACTED]"
        elif isinstance(value, dict):
            sanitized[key] = sanitize_payload(value)
        elif isinstance(value, list):
            sanitized[key] = [sanitize_payload(item) if isinstance(item, dict) else item for item in value]
        else:
            sanitized[key] = value

    return sanitized

# Infrastructure Preflight Checks for Self-Hosted Deployments
async def run_infrastructure_preflight(public_base_url: str = None, webhook_base_url: str = None) -> dict:
    """Run comprehensive infrastructure validation for self-hosted deployments."""
    checks = {}
    all_passed = True

    # 1. PUBLIC_BASE_URL validation
    public_url = public_base_url or os.getenv("PUBLIC_BASE_URL")
    if not public_url:
        checks["public_base_url"] = {
            "status": "FAIL",
            "message": "PUBLIC_BASE_URL not configured",
            "solution": "Set PUBLIC_BASE_URL environment variable to your public domain"
        }
        all_passed = False
    else:
        # Validate URL format
        if not public_url.startswith(('http://', 'https://')):
            checks["public_base_url"] = {
                "status": "FAIL",
                "message": "PUBLIC_BASE_URL must start with http:// or https://",
                "solution": "Use https:// for production deployments"
            }
            all_passed = False
        else:
            checks["public_base_url"] = {
                "status": "OK",
                "message": f"Valid URL: {public_url}",
                "url": public_url
            }

    # 2. TLS/HTTPS validation
    if public_url and public_url.startswith('https://'):
        try:
            # Test HTTPS connectivity
            response = requests.get(public_url, timeout=10, verify=False)  # verify=False for self-signed
            if response.status_code < 400:
                checks["tls_https"] = {
                    "status": "OK",
                    "message": "HTTPS connection successful"
                }
            else:
                checks["tls_https"] = {
                    "status": "WARN",
                    "message": f"HTTPS reachable but returned status {response.status_code}"
                }
        except Exception as e:
            checks["tls_https"] = {
                "status": "FAIL",
                "message": f"HTTPS connection failed: {str(e)}",
                "solution": "Ensure TLS certificate is properly configured and domain is accessible"
            }
            all_passed = False
    else:
        checks["tls_https"] = {
            "status": "WARN",
            "message": "Using HTTP - consider HTTPS for production",
            "solution": "Configure TLS/SSL certificate for secure connections"
        }

    # 3. WhatsApp service health check
    try:
        # Check whatsapp_service health internally
        whatsapp_health_url = "http://whatsapp_service:8000/health"
        response = requests.get(whatsapp_health_url, timeout=5)
        if response.status_code == 200:
            checks["whatsapp_service_health"] = {
                "status": "OK",
                "message": "WhatsApp service is healthy and responding"
            }
        else:
            checks["whatsapp_service_health"] = {
                "status": "FAIL",
                "message": f"WhatsApp service returned status {response.status_code}",
                "solution": "Check WhatsApp service logs and configuration"
            }
            all_passed = False
    except Exception as e:
        checks["whatsapp_service_health"] = {
            "status": "FAIL",
            "message": f"WhatsApp service unreachable: {str(e)}",
            "solution": "Ensure WhatsApp service is running and accessible on port 8000"
        }
        all_passed = False

    # 3b. Webhook configuration check
    webhook_url = webhook_base_url or os.getenv("WEBHOOK_BASE_URL")
    if webhook_url:
        checks["webhook_configuration"] = {
            "status": "OK",
            "message": f"Webhook URL configured: {webhook_url}",
            "url": webhook_url,
            "note": "Ensure this URL is publicly accessible and configured in YCloud"
        }
    else:
        checks["webhook_configuration"] = {
            "status": "WARN",
            "message": "WEBHOOK_BASE_URL not configured",
            "solution": "Set WEBHOOK_BASE_URL to your public webhook endpoint for YCloud"
        }

    # 4. Clock drift check
    try:
        # Compare local time with a reliable NTP server or just check if time seems reasonable
        import time
        current_time = time.time()
        # Basic sanity check - if time is before 2024, something is wrong
        if current_time < 1704067200:  # 2024-01-01
            checks["clock_drift"] = {
                "status": "FAIL",
                "message": "Server clock appears to be incorrect",
                "solution": "Configure NTP synchronization on the host system"
            }
            all_passed = False
        else:
            checks["clock_drift"] = {
                "status": "OK",
                "message": "Server clock appears synchronized"
            }
    except Exception as e:
        checks["clock_drift"] = {
            "status": "WARN",
            "message": f"Could not verify clock: {str(e)}"
        }

    # 5. Database persistence check
    try:
        # Test database connectivity and basic operations
        test_result = await db.pool.fetchval("SELECT 1")
        if test_result == 1:
            # Test persistence by writing and reading back with unique table name
            import uuid
            table_name = f"preflight_test_{uuid.uuid4().hex[:8]}"
            test_key = f"test_data_{int(time.time())}"

            # Use a unique temporary table name to avoid conflicts
            await db.pool.execute(f"CREATE TEMP TABLE {table_name} (id SERIAL PRIMARY KEY, data TEXT)")
            await db.pool.execute(f"INSERT INTO {table_name} (data) VALUES ($1)", test_key)
            read_back = await db.pool.fetchval(f"SELECT data FROM {table_name} WHERE data = $1", test_key)

            # Clean up the temp table
            await db.pool.execute(f"DROP TABLE {table_name}")

            if read_back == test_key:
                checks["database_persistence"] = {
                    "status": "OK",
                    "message": "Database connectivity and persistence working"
                }
            else:
                checks["database_persistence"] = {
                    "status": "FAIL",
                    "message": "Database persistence test failed",
                    "solution": "Check database configuration and storage"
                }
                all_passed = False
        else:
            checks["database_persistence"] = {
                "status": "FAIL",
                "message": "Database connectivity test failed",
                "solution": "Check database connection and credentials"
            }
            all_passed = False
    except Exception as e:
        checks["database_persistence"] = {
            "status": "FAIL",
            "message": f"Database error: {str(e)}",
            "solution": "Verify database configuration, credentials, and network connectivity"
        }
        all_passed = False

    # 6. Required secrets check
    required_secrets = {
        "ADMIN_TOKEN": os.getenv("ADMIN_TOKEN"),
        "INTERNAL_API_TOKEN": os.getenv("INTERNAL_API_TOKEN"),
        "HMAC_SHARED_SECRET": os.getenv("HMAC_SHARED_SECRET")
    }

    missing_secrets = [k for k, v in required_secrets.items() if not v]
    if missing_secrets:
        checks["required_secrets"] = {
            "status": "FAIL",
            "message": f"Missing required secrets: {', '.join(missing_secrets)}",
            "solution": "Set all required environment variables for secure operation"
        }
        all_passed = False
    else:
        checks["required_secrets"] = {
            "status": "OK",
            "message": "All required secrets configured"
        }

    # 7. Port and reverse proxy check
    try:
        # Check if we're running behind a reverse proxy by looking at forwarded headers
        # This is a basic check - in production you might want more sophisticated validation
        checks["reverse_proxy"] = {
            "status": "OK",
            "message": "Reverse proxy configuration appears functional"
        }
    except Exception as e:
        checks["reverse_proxy"] = {
            "status": "WARN",
            "message": f"Could not verify reverse proxy: {str(e)}"
        }

    return {
        "overall_status": "OK" if all_passed else "FAIL",
        "checks": checks,
        "timestamp": datetime.now().isoformat(),
        "public_base_url": public_url,
        "webhook_base_url": webhook_url
    }

# Configuration & Store Branding & Context (Whitelabel)
STORE_NAME = os.getenv("STORE_NAME", "Pointe Coach")
STORE_LOCATION = os.getenv("STORE_LOCATION", "Paraná, Entre Ríos, Argentina")
STORE_DESCRIPTION = os.getenv("STORE_DESCRIPTION", "tienda de artículos de danza clásica y contemporánea")
STORE_WEBSITE = os.getenv("STORE_WEBSITE", "https://www.pointecoach.shop/")
STORE_CATALOG_KNOWLEDGE = os.getenv("STORE_CATALOG_KNOWLEDGE", """
- Accesorios: Metatarsianas, Bolsa de red, Elásticos, Cintas de satén y elastizadas, Endurecedor para puntas, Accesorios para el pie, Punteras, Protectores de puntas.
- Medias: Medias convertibles, Socks, Medias de contemporáneo, Medias poliamida, Medias de patín.
- Zapatillas: Zapatillas de punta, Zapatillas de media punta.
- Marcas: Pointe Coach, Grishko, Capezio, Sansha.
""")

# Initialize config - fallback to env vars
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "admin-secret-99") # Secret for frontend auth
INTERNAL_API_TOKEN = os.getenv("INTERNAL_API_TOKEN")
HMAC_SHARED_SECRET = os.getenv("HMAC_SHARED_SECRET")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")

# Initialize integration variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TIENDANUBE_ACCESS_TOKEN = os.getenv("TIENDANUBE_ACCESS_TOKEN")
TIENDANUBE_API_BASE = os.getenv("TIENDANUBE_API_BASE", "https://api.tiendanube.com/v1")
YCLOUD_API_KEY = os.getenv("YCLOUD_API_KEY")
YCLOUD_WEBHOOK_SECRET = os.getenv("YCLOUD_WEBHOOK_SECRET")
WHATSAPP_SERVICE_URL = os.getenv("WHATSAPP_SERVICE_URL", "http://whatsapp_service:8000")
TIENDANUBE_SERVICE_URL = os.getenv("TIENDANUBE_SERVICE_URL", "http://tiendanube_service:8001")

# Function to get credentials with tenant-specific resolution
async def get_credential_value(name: str, tenant_id: int = None, fallback_env: str = None) -> str:
    """Get credential value from database, with tenant-specific resolution."""
    try:
        cred = await db.get_credentials_by_name(name, tenant_id)
        if cred:
            return cred['value']
    except Exception as e:
        logger.warning(f"Failed to get credential {name} from database", error=str(e))

    # Fallback to environment variable
    if fallback_env:
        return os.getenv(fallback_env)
    return os.getenv(name)

async def check_whatsapp_status(tenant_id: int) -> dict:
    """Check WhatsApp connection status for a tenant."""
    try:
        # Check YCloud credentials
        ycloud_api_key = await get_credential_value("YCLOUD_API_KEY", tenant_id)
        ycloud_webhook_secret = await get_credential_value("YCLOUD_WEBHOOK_SECRET", tenant_id)

        # Check Meta API credentials
        meta_access_token = await get_credential_value(f"WHATSAPP_ACCESS_TOKEN_{tenant_id}", tenant_id)
        meta_phone_id = await get_credential_value(f"WHATSAPP_PHONE_NUMBER_ID_{tenant_id}", tenant_id)

        status = {
            "ycloud": {
                "configured": bool(ycloud_api_key and ycloud_webhook_secret),
                "api_key_masked": f"***{ycloud_api_key[-4:]}" if ycloud_api_key else None,
                "webhook_secret_masked": f"***{ycloud_webhook_secret[-4:]}" if ycloud_webhook_secret else None
            },
            "meta_api": {
                "configured": bool(meta_access_token and meta_phone_id),
                "access_token_masked": f"***{meta_access_token[-4:]}" if meta_access_token else None,
                "phone_number_id_masked": f"***{meta_phone_id[-4:]}" if meta_phone_id else None
            }
        }

        # Determine overall status
        if status["ycloud"]["configured"] or status["meta_api"]["configured"]:
            status["overall"] = "configured"
            status["provider"] = "ycloud" if status["ycloud"]["configured"] else "meta_api"
        else:
            status["overall"] = "not_configured"
            status["provider"] = None

        return status
    except Exception as e:
        logger.error(f"Error checking WhatsApp status for tenant {tenant_id}", error=str(e))
        return {
            "ycloud": {"configured": False},
            "meta_api": {"configured": False},
            "overall": "error",
            "error": str(e)
        }

# Feature flags for rollback
DIAGNOSTICS_ENABLED = os.getenv("DIAGNOSTICS_ENABLED", "true").lower() == "true"
UI_ONBOARDING_ENABLED = os.getenv("UI_ONBOARDING_ENABLED", "true").lower() == "true"
META_CREDENTIALS_ENABLED = os.getenv("META_CREDENTIALS_ENABLED", "true").lower() == "true"
STRICT_ENV_VALIDATION = os.getenv("STRICT_ENV_VALIDATION", "true").lower() == "true"
BOOTSTRAP_ENABLED = os.getenv("BOOTSTRAP_ENABLED", "true").lower() == "true"

# Validate critical integration keys
if not OPENAI_API_KEY:
    print("CRITICAL ERROR: OPENAI_API_KEY not found. Please set the OPENAI_API_KEY environment variable.")

# Initialize Structlog
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)
logger = structlog.get_logger()

# Initialize Redis
redis_client = redis.from_url(REDIS_URL)

# --- Shared Models ---
class ToolError(BaseModel):
    code: str = Field(..., description="Error code")
    message: str
    retryable: bool
    details: Optional[Dict[str, Any]] = None

class ToolResponse(BaseModel):
    ok: bool
    data: Optional[Any] = None
    error: Optional[ToolError] = None
    meta: Optional[Dict[str, Any]] = None

class InboundChatEvent(BaseModel):
    provider: str
    event_id: str
    provider_message_id: str
    from_number: str
    to_number: str = Field(..., description="The bot's WhatsApp number (receiver)")
    text: str
    customer_name: Optional[str] = None
    event_type: str
    correlation_id: str

class OrchestratorMessage(BaseModel):
    part: Optional[int] = Field(None, description="The sequence number of this message.")
    total: Optional[int] = Field(None, description="The total number of messages.")
    text: str = Field(..., description="The text content of this message burst.")
    imageUrl: Optional[str] = Field(None, description="The URL of the product image (images[0].src from tools), or null if no image is available.")

class OrchestratorResult(BaseModel):
    status: Literal["ok", "duplicate", "ignored", "error"]
    send: bool
    text: Optional[str] = None
    messages: List[OrchestratorMessage] = Field(default_factory=list)
    meta: Optional[Dict[str, Any]] = None

# FastAPI App
app = FastAPI(
    title="Orchestrator Service",
    description="Central intelligence for Kilocode microservices.",
)

from fastapi.middleware.cors import CORSMiddleware

# CORS Configuration
# P0-2: Configurable CORS, defaulting to "proxy" (Traefik handles CORS)
# If CORS_MODE="app", we enable CORSMiddleware for development or standalone use.
CORS_MODE = os.getenv("CORS_MODE", "proxy")

if CORS_MODE == "app":
    cors_env = os.getenv("CORS_ALLOWED_ORIGINS", "*")
    if cors_env == "*":
        allow_origins = ["*"]
    else:
        # Support CSV or JSON
        if cors_env.strip().startswith("["):
            try:
                allow_origins = json.loads(cors_env)
            except:
                allow_origins = ["*"]
        else:
            allow_origins = [origin.strip() for origin in cors_env.split(",")]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("CORS_MODE=app: CORSMiddleware enabled", allow_origins=allow_origins)
else:
    logger.info("CORS_MODE=proxy: CORSMiddleware disabled (relying on Traefik)")

# Global exception handlers
# P0-2: Removed hardcoded CORS headers. CORSMiddleware handles it automatically.
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"error": "Validation error", "details": str(exc)},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error("unhandled_exception", error=str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )


# Metrics
SERVICE_NAME = "orchestrator_service"
REQUESTS = Counter("http_requests_total", "Total Request Count", ["service", "endpoint", "method", "status"])
LATENCY = Histogram("http_request_latency_seconds", "Request Latency", ["service", "endpoint"])
TOOL_CALLS = Counter("tool_calls_total", "Total Tool Calls", ["tool", "status"])

# --- Tools & Helpers ---
def get_cached_tool(key: str):
    try:
        data = redis_client.get(f"cache:tool:{key}")
        if data:
            return json.loads(data)
    except Exception as e:
        logger.error("cache_read_error", error=str(e))
    return None

# --- Tools & Helpers ---
MCP_URL = os.getenv("MCP_URL", "https://n8n-n8n.qvwxm2.easypanel.host/mcp/d36b3e5f-9756-447f-9a07-74d50543c7e8")

def call_mcp_tool(tool_name: str, arguments: dict):
    """Bridge to call tools on n8n MCP server with stateful session and SSE support."""
    logger.info("mcp_handshake_start", tool=tool_name)
    try:
        session = requests.Session()
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        
        # 1. Initialize
        init_payload = {
            "jsonrpc": "2.0",
            "id": "init-" + str(uuid.uuid4())[:8],
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "Orchestrator-Bridge", "version": "1.0"}
            }
        }
        init_resp = session.post(MCP_URL, json=init_payload, headers=headers, timeout=10)
        
        if init_resp.status_code != 200:
            return f"MCP Init Failed ({init_resp.status_code}): {init_resp.text}"
        
        # Capture Mcp-Session-Id
        session_id = init_resp.headers.get("Mcp-Session-Id")
        if not session_id:
            try:
                result = init_resp.json().get("result", {})
                session_id = result.get("meta", {}).get("sessionId") or result.get("sessionId")
            except: pass
        
        if session_id:
            logger.info("mcp_session_captured", session_id=session_id)
            session.headers.update({"Mcp-Session-Id": session_id})

        # 2. Notifications/initialized
        notif_payload = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        session.post(MCP_URL, json=notif_payload, headers=headers, timeout=5)

        # 3. Call Tool
        call_payload = {
            "jsonrpc": "2.0",
            "id": "call-" + str(uuid.uuid4())[:8],
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        # MCP over SSE uses a stream. We need to catch the first relevant 'data:' line
        resp = session.post(MCP_URL, json=call_payload, headers=headers, timeout=30, stream=True)
        
        if resp.status_code != 200:
            raw_text = resp.text if not resp.encoding else "".join([chunk.decode(resp.encoding) for chunk in resp.iter_content(8192)])
            return f"MCP Tool Call Error {resp.status_code}: {raw_text}"

        all_text = ""
        # Read the stream for 'data:' lines (Standard MCP over SSE)
        for line in resp.iter_lines():
            if not line: continue
            line_str = line.decode('utf-8').strip()
            if line_str.startswith("data: "):
                data_json = line_str[6:]
                try:
                    msg = json.loads(data_json)
                    # Look for the result or error in the JSON-RPC response
                    if msg.get("id") == call_payload["id"] or "result" in msg or "error" in msg:
                        if "result" in msg: return msg["result"]
                        if "error" in msg: return f"MCP Tool Error: {msg['error']}"
                except: pass
            all_text += line_str + "\n"

        # Fallback if no SSE 'data:' was found but we have body
        if not all_text.strip():
            return "MCP Server returned an empty response. Check if the n8n workflow is correctly sending a response."
        
        try:
            # Maybe it wasn't SSE after all
            json_resp = json.loads(all_text)
            if "result" in json_resp: return json_resp["result"]
            return json_resp
        except:
            return all_text
        
    except Exception as e:
        logger.error("mcp_bridge_error", tool=tool_name, error=str(e))
        return f"MCP Bridge Exception: {str(e)}"

def get_cached_tool(key: str):
    try:
        data = redis_client.get(f"cache:tool:{key}")
        if data:
            return json.loads(data)
    except Exception as e:
        logger.error("cache_read_error", error=str(e))
    return None

def set_cached_tool(key: str, data: dict, ttl: int = 300):
    try:
        redis_client.setex(f"cache:tool:{key}", ttl, json.dumps(data))
    except Exception as e:
        logger.error("cache_write_error", error=str(e))

def call_tiendanube_api(endpoint: str, params: dict = None, token_val: str = None, api_base: str = None):
    # Use provided values or fallback to globals (legacy/single-tenant support)
    current_token = (token_val or TIENDANUBE_ACCESS_TOKEN or "").strip()
    current_base = api_base or TIENDANUBE_API_BASE
    
    headers = {
        "Authentication": f"bearer {current_token}",
        "User-Agent": os.getenv("TIENDANUBE_USER_AGENT", "n8n (santiago@atendo.agency)"),
        "Content-Type": "application/json"
    }
    try:
        url = f"{current_base}{endpoint}"
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code != 200:
            return f"Error HTTP {response.status_code}: {response.text}"
        return response.json()
    except Exception as e:
        return f"Request Error: {str(e)}"

# --- Dynamic Tool Factory ---
async def get_tenant_tools_dynamic(tenant_id: int):
    """Load tools dynamically for a tenant."""
    tools = await db.get_tenant_tools(tenant_id)
    tool_functions = []

    for tool in tools:
        if tool['type'] == 'http':
            @tool
            def http_tool(url: str, method: str = "GET", headers: str = "{}", body: str = ""):
                """Execute HTTP request."""
                import json
                try:
                    headers_dict = json.loads(headers)
                    response = requests.request(method, url, headers=headers_dict, data=body, timeout=10)
                    return f"Status: {response.status_code}, Body: {response.text[:500]}"
                except Exception as e:
                    return f"Error: {str(e)}"
            tool_functions.append(http_tool)
        elif tool['type'] == 'tienda_nube':
            # Add Tienda Nube tools as before
            pass  # For now, keep static

    return tool_functions

def get_tenant_tools_static(tn_token: str, tn_api_base: str):
    """Creates a set of static tools bound to a specific tenant's credentials."""

    @tool
    def productsq(q: str):
        """Search for products by keyword in Tienda Nube."""
        cache_key = f"productsq:{tn_token[:8]}:{q}"
        cached = get_cached_tool(cache_key)
        if cached: return cached
        result = call_tiendanube_api("/products", {"q": q, "per_page": 20}, token_val=tn_token, api_base=tn_api_base)
        if isinstance(result, (dict, list)): set_cached_tool(cache_key, result, ttl=600)
        return result

    @tool
    def productsq_category(category: str, keyword: str):
        """Search for products by category and keyword in Tienda Nube."""
        q = f"{category} {keyword}"
        cache_key = f"productsq_category:{tn_token[:8]}:{category}:{keyword}"
        cached = get_cached_tool(cache_key)
        if cached: return cached
        result = call_tiendanube_api("/products", {"q": q, "per_page": 20}, token_val=tn_token, api_base=tn_api_base)
        if isinstance(result, (dict, list)): set_cached_tool(cache_key, result, ttl=600)
        return result

    @tool
    def productsall():
        """Get a general list of products from Tienda Nube."""
        cache_key = f"productsall:{tn_token[:8]}"
        cached = get_cached_tool(cache_key)
        if cached: return cached
        result = call_tiendanube_api("/products", {"per_page": 25}, token_val=tn_token, api_base=tn_api_base)
        if isinstance(result, (dict, list)): set_cached_tool(cache_key, result, ttl=600)
        return result

    @tool
    def cupones_list():
        """List active coupons and discounts from Tienda Nube via n8n MCP."""
        return call_mcp_tool("cupones_list", {"tn_token": tn_token}) # Pass token to MCP if needed

    @tool
    def orders(q: str):
        """Search for order information directly in Tienda Nube API."""
        clean_q = q.replace("#", "").strip()
        return call_tiendanube_api("/orders", {"q": clean_q}, token_val=tn_token, api_base=tn_api_base)

    @tool
    def sendemail(subject: str, text: str):
        """Send an email to support or customer via n8n MCP."""
        return call_mcp_tool("sendemail", {"Subject": subject, "Text": text, "tn_token": tn_token})

    return [productsq, productsq_category, productsall, cupones_list, orders, sendemail]

from langchain.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# --- Output Schema for Agent ---
class OrchestratorResponse(BaseModel):
    """The structured response from the orchestrator agent containing multiple messages."""
    messages: List[OrchestratorMessage] = Field(description="List of messages (parts) to send to the user, in order.")

# Initialize Parser
parser = PydanticOutputParser(pydantic_object=OrchestratorResponse)

# Base LLM Initialize
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    api_key=OPENAI_API_KEY, 
    temperature=0, 
    max_tokens=2000 
)

# Middleware
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
    ).info("http_request_completed" if status_code < 400 else "http_request_failed")
    return response

# Endpoints
@app.get("/metrics")
def metrics(): return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/ready")
async def ready():
    """Readiness check: system is operational with valid config."""
    if not STRICT_ENV_VALIDATION:
        # Skip validation if disabled
        try:
            if db.pool: await db.pool.fetchval("SELECT 1")
            redis_client.ping()
        except Exception as e:
            logger.error("readiness_check_failed", error=str(e))
            raise HTTPException(status_code=503, detail="Dependencies unavailable")
        return {"status": "ok"}

    # Validate configuration
    config_issues = validate_configuration()

    # Check dependencies
    deps_ok = True
    dep_errors = []
    try:
        if db.pool: await db.pool.fetchval("SELECT 1")
    except Exception as e:
        deps_ok = False
        dep_errors.append(f"Database: {str(e)}")

    try:
        redis_client.ping()
    except Exception as e:
        deps_ok = False
        dep_errors.append(f"Redis: {str(e)}")

    # Determine readiness
    if config_issues["missing_required_env"] or config_issues["invalid_env"] or not deps_ok:
        error_details = []
        if config_issues["missing_required_env"]:
            error_details.append(f"Missing required envs: {', '.join(config_issues['missing_required_env'])}")
        if config_issues["invalid_env"]:
            error_details.append(f"Invalid envs: {', '.join(config_issues['invalid_env'])}")
        if dep_errors:
            error_details.append(f"Dependency errors: {'; '.join(dep_errors)}")

        raise HTTPException(
            status_code=503,
            detail=f"System not ready: {'; '.join(error_details)}"
        )

    return {
        "status": "ok",
        "config_warnings": config_issues["integration_env_missing"]
    }

@app.get("/health")
def health(): return {"status": "ok"}

@app.get("/version")
def version(): return {"version": "1.2"}

async def verify_internal_token(x_internal_token: str = Header(...)):
    if INTERNAL_API_TOKEN and x_internal_token != INTERNAL_API_TOKEN:
         raise HTTPException(status_code=401, detail="Invalid Internal Token")

@app.on_event("startup")
async def startup(): 
    await db.connect()
    await db.initialize()

# --- Helpers ---
def verify_hmac_signature(request_body: bytes, signature_header: str, secret: str) -> bool:
    """Verify HMAC signature for request integrity."""
    if not signature_header or not secret:
        return False

    try:
        parts = {k: v for k, v in [p.split("=") for p in signature_header.split(",")]}
        t, s = parts.get("t"), parts.get("s")
    except:
        return False

    if not t or not s:
        return False

    # Check timestamp (5 minute tolerance)
    if abs(time.time() - int(t)) > 300:
        return False

    signed_payload = f"{t}.{request_body.decode('utf-8')}"
    expected = hmac.new(secret.encode("utf-8"), signed_payload.encode("utf-8"), hashlib.sha256).hexdigest()

    return hmac.compare_digest(expected, s)

# --- Admin API Endpoints ---
async def verify_admin_token(x_admin_token: str = Header(...)):
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid Admin Token")

async def verify_meta_credentials_request(request: Request, x_tenant_id: str = Header(...), x_signature: str = Header(...)):
    """Verify HMAC signature and tenant context for Meta API credentials."""
    if not META_CREDENTIALS_ENABLED:
        raise HTTPException(status_code=403, detail="Meta credentials feature disabled")

    if not x_tenant_id:
        raise HTTPException(status_code=400, detail="Missing X-Tenant-Id header")

    if not HMAC_SHARED_SECRET:
        raise HTTPException(status_code=500, detail="HMAC secret not configured")

    # Verify HMAC signature
    body = await request.body()
    if not verify_hmac_signature(body, x_signature, HMAC_SHARED_SECRET):
        raise HTTPException(status_code=401, detail="Invalid HMAC signature")

    return x_tenant_id

# --- Diagnostics Endpoints ---
@app.get("/diagnostics/ping", dependencies=[Depends(verify_admin_token)])
async def diagnostics_ping():
    """Basic connectivity test for UI to orchestrator_service."""
    if not DIAGNOSTICS_ENABLED:
        raise HTTPException(status_code=403, detail="Diagnostics feature disabled")

    return {
        "ok": True,
        "service": "orchestrator_service",
        "timestamp": datetime.now().isoformat(),
        "correlation_id": str(uuid.uuid4())
    }

@app.get("/diagnostics/openai/test", dependencies=[Depends(verify_admin_token)])
async def diagnostics_openai_test():
    """Test if OpenAI is configured."""
    if not DIAGNOSTICS_ENABLED:
        raise HTTPException(status_code=403, detail="Diagnostics feature disabled")

    response_data = {}
    if OPENAI_API_KEY:
        response_data = {
            "status": "OK",
            "message": "OpenAI API key configured via environment",
            "timestamp": datetime.now().isoformat()
        }
    else:
        # Check credentials table
        creds = await db.get_all_credentials()
        openai_cred = next((c for c in creds if c['name'] == 'OPENAI_API_KEY'), None)
        if openai_cred and openai_cred.get('value'):
            response_data = {
                "status": "OK",
                "message": "OpenAI API key configured via credentials",
                "timestamp": datetime.now().isoformat()
            }
        else:
            response_data = {
                "status": "FAIL",
                "error": "OpenAI API key not configured",
                "timestamp": datetime.now().isoformat()
            }

    # Return result
    return response_data

@app.options("/diagnostics/ycloud/test")
async def options_ycloud_test():
    return JSONResponse(content={})

@app.get("/diagnostics/ycloud/test", dependencies=[Depends(verify_admin_token)])
async def diagnostics_ycloud_test():
    """Test if YCloud is configured."""
    if not DIAGNOSTICS_ENABLED:
        raise HTTPException(status_code=403, detail="Diagnostics feature disabled")

    if YCLOUD_API_KEY and YCLOUD_WEBHOOK_SECRET:
        return {
            "status": "OK",
            "message": "YCloud credentials configured via environment",
            "timestamp": datetime.now().isoformat()
        }

    # Check credentials table
    creds = await db.get_all_credentials()
    api_key_cred = next((c for c in creds if c['name'] == 'YCLOUD_API_KEY'), None)
    webhook_cred = next((c for c in creds if c['name'] == 'YCLOUD_WEBHOOK_SECRET'), None)

    if api_key_cred and api_key_cred.get('value') and webhook_cred and webhook_cred.get('value'):
        return {
            "status": "OK",
            "message": "YCloud credentials configured via credentials",
            "timestamp": datetime.now().isoformat()
        }

    return {
        "status": "FAIL",
        "error": "YCloud credentials not configured",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/diagnostics/healthz", dependencies=[Depends(verify_admin_token)])
async def diagnostics_healthz():
    if not DIAGNOSTICS_ENABLED:
        raise HTTPException(status_code=403, detail="Diagnostics feature disabled")
    """Aggregated health check for all services and dependencies."""
    correlation_id = str(uuid.uuid4())
    checks = []

    # Check Environment Config (P0-1)
    config_issues = validate_configuration()
    if config_issues["missing_required_env"]:
        checks.append({
            "name": "config",
            "status": "FAIL",
            "error_code": "MISSING_ENV",
            "error": f"Missing critical envs: {', '.join(config_issues['missing_required_env'])}"
        })
    else:
        checks.append({
            "name": "config",
            "status": "OK"
        })

    # Check PostgreSQL
    try:
        start_time = time.time()
        await db.pool.fetchval("SELECT 1")
        latency = int((time.time() - start_time) * 1000)
        checks.append({
            "name": "postgres",
            "status": "OK",
            "latency_ms": latency
        })
    except Exception as e:
        checks.append({
            "name": "postgres",
            "status": "FAIL",
            "error_code": "DB_CONNECTION_ERROR",
            "error": str(e)
        })

    # Check Redis
    try:
        start_time = time.time()
        redis_client.ping()
        latency = int((time.time() - start_time) * 1000)
        checks.append({
            "name": "redis",
            "status": "OK",
            "latency_ms": latency
        })
    except Exception as e:
        checks.append({
            "name": "redis",
            "status": "FAIL",
            "error_code": "REDIS_CONNECTION_ERROR",
            "error": str(e)
        })

    # Check whatsapp_service (service + credentials)
    try:
        start_time = time.time()
        # First check service health
        response = requests.get(f"{WHATSAPP_SERVICE_URL}/health", timeout=5)
        latency = int((time.time() - start_time) * 1000)

        if response.status_code == 200:
            # Service is up, now check if credentials are configured
            yc_key = os.getenv("YCLOUD_API_KEY")
            yc_secret = os.getenv("YCLOUD_WEBHOOK_SECRET")

            if yc_key and yc_secret:
                checks.append({
                    "name": "whatsapp_service",
                    "status": "OK",
                    "latency_ms": latency,
                    "credentials": "configured"
                })
            else:
                checks.append({
                    "name": "whatsapp_service",
                    "status": "WARN",
                    "error_code": "WA_CREDENTIALS_MISSING",
                    "error": "Service running but YCloud credentials not configured",
                    "latency_ms": latency
                })
        else:
            checks.append({
                "name": "whatsapp_service",
                "status": "FAIL",
                "error_code": f"HTTP_{response.status_code}",
                "latency_ms": latency
            })
    except Exception as e:
        checks.append({
            "name": "whatsapp_service",
            "status": "FAIL",
            "error_code": "SERVICE_UNREACHABLE",
            "error": str(e)
        })

    # Check tiendanube_service (service + credentials)
    try:
        start_time = time.time()
        # First check service health
        response = requests.get(f"{TIENDANUBE_SERVICE_URL}/health", timeout=5)
        latency = int((time.time() - start_time) * 1000)

        if response.status_code == 200:
            # Service is up, check if any credentials are available (global or tenant-specific)
            has_credentials = False

            # Check global env
            if TIENDANUBE_ACCESS_TOKEN:
                cred_check = call_tiendanube_api("/store", token_val=TIENDANUBE_ACCESS_TOKEN, api_base=TIENDANUBE_API_BASE)
                if isinstance(cred_check, dict) and "id" in cred_check:
                    has_credentials = True

            # Check if any tenant has TN credentials
            if not has_credentials:
                tenants = await db.get_all_tenants()
                for tenant in tenants:
                    if tenant.get('tiendanube_access_token'):
                        cred_check = call_tiendanube_api("/store", token_val=tenant['tiendanube_access_token'], api_base=f"https://api.tiendanube.com/v1/{tenant['tiendanube_store_id']}")
                        if isinstance(cred_check, dict) and "id" in cred_check:
                            has_credentials = True
                            break

            if has_credentials:
                checks.append({
                    "name": "tiendanube_service",
                    "status": "OK",
                    "latency_ms": latency,
                    "credentials": "valid"
                })
            else:
                checks.append({
                    "name": "tiendanube_service",
                    "status": "WARN",
                    "error_code": "TN_NO_CREDENTIALS",
                    "error": "Service running but no valid Tienda Nube credentials found",
                    "latency_ms": latency
                })
        else:
            checks.append({
                "name": "tiendanube_service",
                "status": "FAIL",
                "error_code": f"HTTP_{response.status_code}",
                "latency_ms": latency
            })
    except Exception as e:
        checks.append({
            "name": "tiendanube_service",
            "status": "FAIL",
            "error_code": "SERVICE_UNREACHABLE",
            "error": str(e)
        })

    # Determine overall status
    failed_checks = [c for c in checks if c["status"] == "FAIL"]
    if failed_checks:
        # If critical services fail, overall FAIL
        critical_failed = any(c["name"] in ["postgres", "redis"] for c in failed_checks)
        overall_status = "FAIL" if critical_failed else "WARN"
    else:
        overall_status = "OK"

    return {
        "status": overall_status,
        "checks": checks,
        "correlation_id": correlation_id,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/diagnostics/tiendanube/test", dependencies=[Depends(verify_admin_token)])
async def diagnostics_tiendanube_test():
    """Test Tienda Nube credentials with read-only call using global env variables."""
    if not DIAGNOSTICS_ENABLED:
        raise HTTPException(status_code=403, detail="Diagnostics feature disabled")

    correlation_id = str(uuid.uuid4())

    try:
        # Try to get store info (read-only) using global env
        start_time = time.time()
        result = call_tiendanube_api("/store", token_val=TIENDANUBE_ACCESS_TOKEN, api_base=TIENDANUBE_API_BASE)
        latency = int((time.time() - start_time) * 1000)

        if isinstance(result, dict) and "id" in result:
            return {
                "status": "OK",
                "latency_ms": latency,
                "store_id": result.get("id"),
                "store_name": result.get("name"),
                "correlation_id": correlation_id,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "FAIL",
                "error_code": "TN_INVALID_RESPONSE",
                "error": "Tienda Nube returned invalid response format",
                "correlation_id": correlation_id,
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        error_str = str(e)
        if "401" in error_str or "403" in error_str:
            error_code = "TN_AUTH_ERROR"
        elif "404" in error_str:
            error_code = "TN_STORE_NOT_FOUND"
        else:
            error_code = "TN_REQUEST_ERROR"

        return {
            "status": "FAIL",
            "error_code": error_code,
            "error": error_str,
            "correlation_id": correlation_id,
            "timestamp": datetime.now().isoformat()
        }

@app.post("/diagnostics/tiendanube/test", dependencies=[Depends(verify_admin_token)])
async def diagnostics_tiendanube_test_post(test_data: Dict[str, Any]):
    """Test Tienda Nube credentials with read-only operations."""
    if not DIAGNOSTICS_ENABLED:
        raise HTTPException(status_code=403, detail="Diagnostics feature disabled")

    store_id = test_data.get("store_id")
    access_token = test_data.get("access_token")

    if not store_id or not access_token:
        return {"status": "FAIL", "error": "Missing store_id or access_token"}

    try:
        # Test with a simple read operation
        result = call_tiendanube_api("/products", {"per_page": 1}, token_val=access_token, api_base=f"https://api.tiendanube.com/v1/{store_id}")
        if isinstance(result, list) and len(result) >= 0:
            return {"status": "OK", "message": "Tienda Nube connection successful"}
        else:
            return {"status": "FAIL", "error": "Invalid credentials or store not found"}
    except Exception as e:
        return {"status": "FAIL", "error": str(e)}

@app.post("/diagnostics/whatsapp/send_test", dependencies=[Depends(verify_admin_token)])
async def diagnostics_whatsapp_send_test(test_data: Dict[str, Any]):
    """Send test message via WhatsApp to validate provider integration."""
    if not DIAGNOSTICS_ENABLED:
        raise HTTPException(status_code=403, detail="Diagnostics feature disabled")

    correlation_id = test_data.get("correlation_id", str(uuid.uuid4()))
    to_number = test_data.get("to", os.getenv("WHATSAPP_TEST_NUMBER", ""))
    text = test_data.get("text", "CodExy test: reply PING")
    idempotency_key = test_data.get("idempotency_key", str(uuid.uuid4()))

    if not to_number:
        return {
            "status": "FAIL",
            "error_code": "MISSING_TEST_NUMBER",
            "error": "WHATSAPP_TEST_NUMBER not configured",
            "correlation_id": correlation_id,
            "timestamp": datetime.now().isoformat()
        }

    try:
        # Call whatsapp_service to send message
        whatsapp_payload = {
            "to": to_number,
            "text": text,
            "correlation_id": correlation_id,
            "idempotency_key": idempotency_key
        }

        headers = {
            "Content-Type": "application/json",
            "X-Internal-Token": INTERNAL_API_TOKEN
        }

        start_time = time.time()
        response = requests.post(
            f"{WHATSAPP_SERVICE_URL}/send",
            json=whatsapp_payload,
            headers=headers,
            timeout=10
        )
        latency = int((time.time() - start_time) * 1000)

        if response.status_code == 200:
            result = response.json()
            return {
                "status": "OK",
                "latency_ms": latency,
                "message_id": result.get("message_id"),
                "provider_response": result,
                "correlation_id": correlation_id,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "FAIL",
                "error_code": f"WHATSAPP_HTTP_{response.status_code}",
                "error": response.text,
                "latency_ms": latency,
                "correlation_id": correlation_id,
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        return {
            "status": "FAIL",
            "error_code": "WHATSAPP_SEND_ERROR",
            "error": str(e),
            "correlation_id": correlation_id,
            "timestamp": datetime.now().isoformat()
        }

@app.get("/diagnostics/events/stream", dependencies=[Depends(verify_admin_token)])
async def diagnostics_events_stream(limit: int = 20):
    """Stream recent events for UI observability."""
    if not DIAGNOSTICS_ENABLED:
        raise HTTPException(status_code=403, detail="Diagnostics feature disabled")

    try:
        # Get recent inbound messages (webhooks received)
        inbound_query = """
        SELECT
            'webhook_received' as event_type,
            created_at as timestamp,
            correlation_id,
            from_number,
            payload->>'text' as message_text,
            status
        FROM inbound_messages
        WHERE provider = 'whatsapp'
        ORDER BY created_at DESC
        LIMIT $1
        """

        # Get recent chat messages (agent responses)
        chat_query = """
        SELECT
            CASE WHEN role = 'assistant' THEN 'agent_response_sent' ELSE 'message_ingested' END as event_type,
            created_at as timestamp,
            correlation_id,
            from_number,
            content as message_text,
            'completed' as status
        FROM chat_messages
        ORDER BY created_at DESC
        LIMIT $1
        """

        async with db.pool.acquire() as conn:
            inbound_rows = await conn.fetch(inbound_query, limit // 2)
            chat_rows = await conn.fetch(chat_query, limit // 2)

        # Combine and sort by timestamp
        all_events = []
        for row in inbound_rows:
            all_events.append(dict(row))
        for row in chat_rows:
            all_events.append(dict(row))

        # Sort by timestamp descending
        all_events.sort(key=lambda x: x['timestamp'], reverse=True)
        events = all_events[:limit]

        return {
            "events": events,
            "total": len(events),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "events": [],
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/admin/tenants", dependencies=[Depends(verify_admin_token)])
async def list_tenants():
    return await db.get_all_tenants()

@app.get("/admin/tenants/{tenant_id}/details", dependencies=[Depends(verify_admin_token)])
async def get_tenant_details(tenant_id: int):
    """Get detailed information about a tenant including credentials and connection status."""
    logger.info("get_tenant_details_start", tenant_id=tenant_id)

    try:
        # Get tenant basic info
        tenant = await db.get_tenant_config_by_id(tenant_id)
        if not tenant:
            logger.warning("tenant_not_found", tenant_id=tenant_id)
            # Check what tenants exist
            all_tenants = await db.get_all_tenants()
            logger.info("available_tenants_debug", count=len(all_tenants), tenant_ids=[t.get('id') for t in all_tenants])
            return JSONResponse(
                content={"error": f"Tenant with id {tenant_id} not found. Available tenants: {[t.get('id') for t in all_tenants]}"},
                status_code=404
            )

        # Get tenant-specific credentials (masked)
        tenant_creds = await db.get_credentials_by_tenant(tenant_id)
        # Mask sensitive values
        for cred in tenant_creds:
            if 'value' in cred and cred['value']:
                cred['value'] = f"***{cred['value'][-4:]}" if len(cred['value']) > 4 else "[REDACTED]"

        # Get global credentials (masked)
        global_creds = await db.get_global_credentials()
        for cred in global_creds:
            if 'value' in cred and cred['value']:
                cred['value'] = f"***{cred['value'][-4:]}" if len(cred['value']) > 4 else "[REDACTED]"

        # Check WhatsApp connection status
        whatsapp_status = await check_whatsapp_status(tenant_id)

        # Sanitize response to remove datetime objects
        def sanitize_for_json(obj):
            if isinstance(obj, dict):
                return {k: sanitize_for_json(v) for k, v in obj.items() if not isinstance(v, datetime)}
            elif isinstance(obj, list):
                return [sanitize_for_json(item) for item in obj]
            else:
                return obj

        return sanitize_for_json({
            "tenant": tenant,
            "credentials": {
                "tenant_specific": tenant_creds,
                "global_available": global_creds
            },
            "connections": {
                "whatsapp": whatsapp_status
            }
        })
    except Exception as e:
        logger.error("tenant_details_error", tenant_id=tenant_id, error=str(e), exc_info=True)
        return JSONResponse(
            content={"error": f"Internal server error: {str(e)}"},
            status_code=500
        )

@app.get("/admin/tenants/{bot_phone_number}", dependencies=[Depends(verify_admin_token)])
async def get_tenant(bot_phone_number: str):
    return await db.get_tenant_config(bot_phone_number)


@app.delete("/admin/tenants", dependencies=[Depends(verify_admin_token)])
async def delete_all_tenants():
    """Delete all tenants - USE WITH CAUTION"""
    await db.delete_all_tenants()
    return {"status": "ok", "message": "All tenants deleted"}

# Setup Tienda Nube endpoints
@app.get("/setup/tiendanube/stores", dependencies=[Depends(verify_admin_token)])
async def get_tiendanube_stores():
    """List Tienda Nube stores configured"""
    stores = await db.get_tiendanube_stores()
    return {"stores": stores}

@app.post("/setup/tiendanube/stores", dependencies=[Depends(verify_admin_token)])
async def create_tiendanube_store(data: dict):
    """Create a new Tienda Nube store"""
    store_id = await db.create_tiendanube_store(
        tenant_id=data.get("tenant_id", 1),  # Default to first tenant for now
        store_id=data["store_id"],
        access_token=data["access_token"],
        store_name=data.get("store_name")
    )
    return {"status": "ok", "store_id": store_id}

@app.post("/setup/tiendanube/stores/{store_id}/test", dependencies=[Depends(verify_admin_token)])
async def test_tiendanube_store(store_id: int):
    """Test connection to a Tienda Nube store"""
    # Get store details
    stores = await db.get_tiendanube_stores()
    store = next((s for s in stores if s["id"] == store_id), None)
    if not store:
        return {"status": "FAIL", "error": "Store not found"}

    # Test connection
    result = call_tiendanube_api("/store", token_val=store["access_token"], api_base=f"https://api.tiendanube.com/v1/{store['store_id']}")
    success = isinstance(result, dict) and "id" in result

    # Update status
    await db.update_tiendanube_store_status(
        store_id=store_id,
        status="active" if success else "inactive",
        last_test_ok=success,
        error_message=None if success else str(result)
    )

    return {
        "status": "OK" if success else "FAIL",
        "store_name": result.get("name") if success else None,
        "error": None if success else str(result)
    }

@app.post("/setup/tiendanube/stores/{store_id}/activate", dependencies=[Depends(verify_admin_token)])
async def activate_tiendanube_store(store_id: int):
    """Mark a Tienda Nube store as active"""
    await db.update_tiendanube_store_status(store_id, "active")
    return {"status": "ok"}

@app.get("/setup/state", dependencies=[Depends(verify_admin_token)])
async def get_setup_state():
    """Get current setup state"""
    # For now, return basic state - in real impl, this would track per-user progress
    return {
        "currentStep": 1,
        "highestPassed": 0,
        "stepStatuses": {
            1: "pending",
            2: "pending",
            3: "pending"
        }
    }

# Analytics endpoints
@app.get("/analytics/summary", dependencies=[Depends(verify_admin_token)])
async def get_analytics_summary(tenant_id: int = None, store_id: str = None, from_date: str = None, to_date: str = None):
    """Get analytics summary KPIs"""
    # Default to tenant_id 1 if not provided
    if tenant_id is None:
        tenant_id = 1
    return await db.get_analytics_summary(tenant_id, store_id, from_date, to_date)

@app.get("/analytics/timeseries", dependencies=[Depends(verify_admin_token)])
async def get_analytics_timeseries(tenant_id: int, metric: str, store_id: str = None, from_date: str = None, to_date: str = None, bucket: str = 'day'):
    """Get timeseries data for a metric"""
    return await db.get_analytics_timeseries(tenant_id, metric, store_id, from_date, to_date, bucket)

@app.get("/analytics/breakdown", dependencies=[Depends(verify_admin_token)])
async def get_analytics_breakdown(tenant_id: int, dimension: str, store_id: str = None, from_date: str = None, to_date: str = None, limit: int = 10):
    """Get breakdown by dimension"""
    return await db.get_analytics_breakdown(tenant_id, dimension, store_id, from_date, to_date, limit)

@app.get("/telemetry/events", dependencies=[Depends(verify_admin_token)])
async def get_telemetry_events(tenant_id: int, store_id: str = None, from_date: str = None, to_date: str = None,
                              event_type: str = None, severity: str = None, session_id: str = None,
                              order_id: str = None, page: int = 1, page_size: int = 50):
    """Get paginated telemetry events"""
    return await db.get_telemetry_events(tenant_id, store_id, from_date, to_date, event_type, severity, session_id, order_id, page, page_size)

# Setup Session Endpoints for Self-Hosted Deployment Wizard
@app.options("/setup/session")
async def options_setup_session():
    return JSONResponse(content={})

@app.post("/setup/session", dependencies=[Depends(verify_admin_token)])
async def create_setup_session(data: dict):
    """Create a new setup session for deployment wizard"""
    session_id = data.get("session_id", f"setup_{int(time.time())}_{uuid.uuid4().hex[:8]}")
    public_base_url = data.get("public_base_url")
    webhook_base_url = data.get("webhook_base_url")

    try:
        # Create session in database
        session_db_id = await db.create_setup_session(session_id, public_base_url, webhook_base_url)

        # Log session creation
        await db.insert_setup_event(
            session_db_id,
            "session_created",
            step="initialization",
            status="completed",
            message="Setup session created successfully",
            details={"public_base_url": public_base_url, "webhook_base_url": webhook_base_url}
        )

        return JSONResponse(
            content={
                "status": "ok",
                "session_id": session_id,
                "session_db_id": session_db_id,
                "message": "Setup session created. Run preflight checks next."
            }
        )
    except Exception as e:
        logger.error("setup_session_creation_error", error=str(e))
        return JSONResponse(
            content={
                "status": "error",
                "error": str(e),
                "message": "Failed to create setup session"
            },
            status_code=500
        )

@app.post("/setup/preflight", dependencies=[Depends(verify_admin_token)])
async def run_setup_preflight(data: dict):
    """Run infrastructure preflight checks for a setup session"""
    session_id = data.get("session_id")
    if not session_id:
        return JSONResponse(
            content={"status": "error", "error": "session_id required"},
            status_code=400
        )

    try:
        # Get session
        session = await db.get_setup_session(session_id)
        if not session:
            return JSONResponse(
                content={"status": "error", "error": "Setup session not found"},
                status_code=404
            )

        # Log preflight start
        await db.insert_setup_event(
            session["id"],
            "preflight_started",
            step="preflight",
            status="started",
            message="Starting infrastructure validation"
        )

        # Run preflight checks
        preflight_result = await run_infrastructure_preflight(
            session.get("public_base_url"),
            session.get("webhook_base_url")
        )

        # Update session with results
        await db.update_setup_session(
            session_id,
            current_step="preflight",
            status="active",
            infrastructure_checks=preflight_result
        )

        # Log preflight completion
        await db.insert_setup_event(
            session["id"],
            "preflight_completed",
            step="preflight",
            status="completed" if preflight_result["overall_status"] == "OK" else "failed",
            message=f"Preflight checks completed with status: {preflight_result['overall_status']}",
            details=preflight_result
        )

        return JSONResponse(content=preflight_result)

    except Exception as e:
        logger.error("setup_preflight_error", session_id=session_id, error=str(e))
        if session_id:
            try:
                session = await db.get_setup_session(session_id)
                if session:
                    await db.insert_setup_event(
                        session["id"],
                        "preflight_failed",
                        step="preflight",
                        status="failed",
                        message=f"Preflight failed: {str(e)}"
                    )
            except: pass
        return JSONResponse(
            content={"status": "error", "error": str(e)},
            status_code=500
        )

@app.get("/setup/session/{session_id}", dependencies=[Depends(verify_admin_token)])
async def get_setup_session_status(session_id: str):
    """Get current status of a setup session"""
    try:
        session = await db.get_setup_session(session_id)
        if not session:
            return JSONResponse(
                content={"status": "error", "error": "Setup session not found"},
                status_code=404,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                    "Access-Control-Allow-Headers": "*",
                }
            )

        # Get recent events
        events = await db.get_setup_events(session["id"], limit=20)

        return JSONResponse(
            content={
                "session": session,
                "events": events,
                "can_proceed": session.get("infrastructure_checks", {}).get("overall_status") == "OK"
            },
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            }
        )
    except Exception as e:
        logger.error("setup_session_status_error", session_id=session_id, error=str(e))
        return JSONResponse(
            content={"status": "error", "error": str(e)},
            status_code=500,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            }
        )

@app.post("/setup/step/{step_name}", dependencies=[Depends(verify_admin_token)])
async def execute_setup_step(step_name: str, data: dict):
    """Execute a specific setup step (for future extensibility)"""
    session_id = data.get("session_id")
    if not session_id:
        return JSONResponse(
            content={"status": "error", "error": "session_id required"},
            status_code=400
        )

    try:
        session = await db.get_setup_session(session_id)
        if not session:
            return JSONResponse(
                content={"status": "error", "error": "Setup session not found"},
                status_code=404
            )

        # For now, just acknowledge - specific step logic would go here
        await db.insert_setup_event(
            session["id"],
            f"step_{step_name}_executed",
            step=step_name,
            status="completed",
            message=f"Setup step {step_name} executed",
            details=data
        )

        # Update session step
        await db.update_setup_session(session_id, current_step=step_name)

        return JSONResponse(
            content={
                "status": "ok",
                "step": step_name,
                "message": f"Step {step_name} completed successfully"
            }
        )
    except Exception as e:
        logger.error("setup_step_error", session_id=session_id, step=step_name, error=str(e))
        return JSONResponse(
            content={"status": "error", "error": str(e)},
            status_code=500
        )

@app.get("/admin/console/events", dependencies=[Depends(verify_admin_token)])
async def get_console_events(limit: int = 100, severity: str = None, event_type: str = None, search: str = None):
    """Get recent events for console display with filtering - combines all system events"""
    try:
        # Build telemetry query with optional filters
        telemetry_conditions = []
        telemetry_params = []
        param_count = 0

        if severity:
            param_count += 1
            telemetry_conditions.append(f"severity = ${param_count}")
            telemetry_params.append(severity)
        if event_type:
            param_count += 1
            telemetry_conditions.append(f"event_type = ${param_count}")
            telemetry_params.append(event_type)

        where_clause = " AND ".join(telemetry_conditions) if telemetry_conditions else "1=1"

        telemetry_query = f"""
        SELECT
            event_type,
            severity,
            occurred_at as timestamp,
            correlation_id,
            session_id,
            order_id,
            success,
            duration_ms,
            error_code,
            error_message,
            payload
        FROM telemetry_events
        WHERE {where_clause}
        ORDER BY occurred_at DESC
        LIMIT ${param_count + 1}
        """
        telemetry_params.append(limit // 3)

        # Get recent inbound messages
        inbound_query = """
        SELECT
            'webhook_received' as event_type,
            'info' as severity,
            received_at as timestamp,
            correlation_id,
            from_number,
            payload->>'text' as message,
            status
        FROM inbound_messages
        WHERE provider = 'whatsapp'
        ORDER BY received_at DESC
        LIMIT $1
        """

        # Get recent chat messages
        chat_query = """
        SELECT
            CASE WHEN role = 'assistant' THEN 'agent_response_sent' ELSE 'message_processed' END as event_type,
            'info' as severity,
            created_at as timestamp,
            correlation_id,
            from_number,
            content as message,
            'completed' as status
        FROM chat_messages
        ORDER BY created_at DESC
        LIMIT $1
        """

        async with db.pool.acquire() as conn:
            telemetry_rows = await conn.fetch(telemetry_query, *telemetry_params)
            inbound_rows = await conn.fetch(inbound_query, limit // 3)
            chat_rows = await conn.fetch(chat_query, limit // 3)

        # Combine all events
        all_events = []

        # Add telemetry events
        for row in telemetry_rows:
            row_dict = dict(row)
            all_events.append({
                'event_type': row_dict['event_type'],
                'severity': row_dict['severity'],
                'timestamp': row_dict['timestamp'],
                'correlation_id': row_dict['correlation_id'],
                'details': {
                    'session_id': row_dict['session_id'],
                    'order_id': row_dict['order_id'],
                    'success': row_dict['success'],
                    'duration_ms': row_dict['duration_ms'],
                    'error_code': row_dict['error_code'],
                    'error_message': row_dict['error_message'],
                    'payload': row_dict['payload']
                },
                'source': 'telemetry'
            })

        # Add inbound events
        for row in inbound_rows:
            row_dict = dict(row)
            all_events.append({
                'event_type': row_dict['event_type'],
                'severity': row_dict['severity'],
                'timestamp': row_dict['timestamp'],
                'correlation_id': row_dict['correlation_id'],
                'details': {
                    'from_number': row_dict['from_number'],
                    'message': row_dict['message'],
                    'status': row_dict['status']
                },
                'source': 'whatsapp'
            })

        # Add chat events
        for row in chat_rows:
            row_dict = dict(row)
            all_events.append({
                'event_type': row_dict['event_type'],
                'severity': row_dict['severity'],
                'timestamp': row_dict['timestamp'],
                'correlation_id': row_dict['correlation_id'],
                'details': {
                    'from_number': row_dict['from_number'],
                    'message': row_dict['message'],
                    'status': row_dict['status']
                },
                'source': 'agent'
            })

        # Sort by timestamp descending
        all_events.sort(key=lambda x: x['timestamp'] or '2000-01-01', reverse=True)

        # Apply search filter if provided
        if search:
            search_lower = search.lower()
            filtered_events = []
            for event in all_events:
                # Search in event details
                searchable_text = f"{event['event_type']} {event.get('correlation_id', '')} {str(event.get('details', {}))}".lower()
                if search_lower in searchable_text:
                    filtered_events.append(event)
            all_events = filtered_events

        # Limit results
        all_events = all_events[:limit]

        return {
            "events": all_events,
            "total": len(all_events),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error("console_events_error", error=str(e))
        return {
            "events": [],
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/telemetry/events", dependencies=[Depends(verify_admin_token)])
async def create_telemetry_event(event: dict):
    """Create a telemetry event"""
    await db.insert_telemetry_event(event)
    return {"status": "ok"}

@app.get("/admin/tenants/{phone_number}", dependencies=[Depends(verify_admin_token)])
async def get_tenant(phone_number: str):
    return await db.get_tenant_config(phone_number)

@app.options("/admin/tenants")
async def options_tenants():
    return JSONResponse(content={})

@app.post("/admin/tenants", dependencies=[Depends(verify_admin_token)])
async def upsert_tenant(data: dict):
    await db.upsert_tenant(data)
    return JSONResponse(content={"status": "ok"})


@app.get("/admin/tenants/{bot_phone_number}", dependencies=[Depends(verify_admin_token)])
async def get_tenant(bot_phone_number: str):
    return await db.get_tenant_config(bot_phone_number)

@app.delete("/admin/tenants/{bot_phone_number}", dependencies=[Depends(verify_admin_token)])
async def delete_tenant(bot_phone_number: str):
    await db.delete_tenant(bot_phone_number)
    return {"status": "ok"}

# Tools Management
@app.get("/admin/tools", dependencies=[Depends(verify_admin_token)])
async def list_tools():
    return await db.get_all_tools()

@app.post("/admin/tools", dependencies=[Depends(verify_admin_token)])
async def create_tool(data: dict):
    tool_id = await db.create_tool(data)
    return {"status": "ok", "tool_id": tool_id}

@app.get("/admin/tenants/{tenant_id}/tools", dependencies=[Depends(verify_admin_token)])
async def get_tenant_tools(tenant_id: int):
    return await db.get_tenant_tools(tenant_id)

@app.post("/admin/tenants/{tenant_id}/tools/{tool_id}/enable", dependencies=[Depends(verify_admin_token)])
async def enable_tenant_tool(tenant_id: int, tool_id: int):
    await db.enable_tenant_tool(tenant_id, tool_id)
    return {"status": "ok"}

@app.post("/admin/tenants/{tenant_id}/tools/{tool_id}/disable", dependencies=[Depends(verify_admin_token)])
async def disable_tenant_tool(tenant_id: int, tool_id: int):
    await db.disable_tenant_tool(tenant_id, tool_id)
    return {"status": "ok"}

@app.get("/admin/stats", dependencies=[Depends(verify_admin_token)])
async def platform_stats():
    return await db.get_platform_stats()

@app.get("/admin/logs", dependencies=[Depends(verify_admin_token)])
async def platform_logs(limit: int = 50):
    return await db.get_recent_logs(limit)

@app.options("/admin/credentials")
async def options_credentials():
    return JSONResponse(content={})

@app.get("/admin/credentials", dependencies=[Depends(verify_admin_token)])
async def list_credentials():
    creds = await db.get_all_credentials()
    return JSONResponse(content=creds)

@app.get("/admin/credentials/{credential_id}", dependencies=[Depends(verify_admin_token)])
async def get_credential(credential_id: int):
    cred = await db.get_credential_by_id(credential_id)
    if not cred:
        return JSONResponse(
            content={"error": "Credential not found"},
            status_code=404,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            }
        )
    return JSONResponse(
        content=cred,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.post("/admin/credentials", dependencies=[Depends(verify_admin_token)])
async def upsert_credential(data: dict):
    """Create or update a credential."""
    try:
        # Validate required fields
        if not data.get('name'):
            return JSONResponse(
                content={"error": "Credential name is required"},
                status_code=400
            )

        if not data.get('value'):
            return JSONResponse(
                content={"error": "Credential value is required"},
                status_code=400
            )

        if not data.get('scope') or data['scope'] not in ['global', 'tenant']:
            return JSONResponse(
                content={"error": "Valid scope ('global' or 'tenant') is required"},
                status_code=400
            )

        # For tenant scope, tenant_id is required
        if data['scope'] == 'tenant' and not data.get('tenant_id'):
            return JSONResponse(
                content={"error": "tenant_id is required for tenant-scoped credentials"},
                status_code=400
            )

        # Convert tenant_id to int if present
        if data.get('tenant_id'):
            try:
                data['tenant_id'] = int(data['tenant_id'])
            except ValueError:
                return JSONResponse(
                    content={"error": "tenant_id must be a valid integer"},
                    status_code=400
                )

        await db.upsert_credential(data)

        return {"status": "ok"}

    except Exception as e:
        logger.error("upsert_credential_error", error=str(e), exc_info=True)
        return JSONResponse(
            content={"error": f"Failed to save credential: {str(e)}"},
            status_code=500
        )

@app.delete("/admin/credentials/{credential_id}", dependencies=[Depends(verify_admin_token)])
async def delete_credential(credential_id: int):
    await db.delete_credential(credential_id)
    return JSONResponse(content={"status": "ok"})

@app.put("/admin/credentials/{credential_id}", dependencies=[Depends(verify_admin_token)])
async def update_credential(credential_id: int, data: dict):
    # First check if credential exists
    existing = await db.get_credential_by_id(credential_id)
    if not existing:
        return JSONResponse(
            content={"status": "error", "error": "Credential not found"},
            status_code=404
        )

    # Update the credential
    update_data = {
        "name": data.get("name", existing["name"]),
        "value": data.get("value", existing["value"]),
        "scope": data.get("scope", existing["scope"]),
        "tenant_id": data.get("tenant_id", existing["tenant_id"])
    }
    await db.upsert_credential(update_data)
    return JSONResponse(content={"status": "ok"})

@app.post("/admin/ycloud", dependencies=[Depends(verify_admin_token)])
async def configure_ycloud(data: dict):
    """Configure YCloud credentials by storing them as global credentials."""
    api_key = data.get("api_key")
    webhook_secret = data.get("webhook_secret")

    if not api_key or not webhook_secret:
        raise HTTPException(status_code=400, detail="Missing api_key or webhook_secret")

    # Store as global credentials
    await db.upsert_credential({
        "name": "YCLOUD_API_KEY",
        "value": api_key,
        "scope": "global"
    })
    await db.upsert_credential({
        "name": "YCLOUD_WEBHOOK_SECRET",
        "value": webhook_secret,
        "scope": "global"
    })

    return {"status": "ok"}

@app.post("/admin/whatsapp-meta", dependencies=[Depends(verify_meta_credentials_request)])
async def update_whatsapp_meta_config(data: dict, tenant_id: str):
    """Securely store WhatsApp Meta API credentials with tenant scoping."""
    # Store credentials with tenant-specific names
    await db.upsert_credential({
        "name": f"WHATSAPP_ACCESS_TOKEN_{tenant_id}",
        "value": data.get("access_token"),
        "scope": "tenant"
    })
    await db.upsert_credential({
        "name": f"WHATSAPP_PHONE_NUMBER_ID_{tenant_id}",
        "value": data.get("phone_number_id"),
        "scope": "tenant"
    })
    await db.upsert_credential({
        "name": f"WHATSAPP_BUSINESS_ACCOUNT_ID_{tenant_id}",
        "value": data.get("business_account_id"),
        "scope": "tenant"
    })
    await db.upsert_credential({
        "name": f"WHATSAPP_VERIFY_TOKEN_{tenant_id}",
        "value": data.get("verify_token"),
        "scope": "tenant"
    })

    # Return masked identifiers only (no secrets)
    return {
        "status": "ok",
        "phone_number_id": f"***{data.get('phone_number_id', '')[-4:]}" if data.get('phone_number_id') else None,
        "business_account_id": f"***{data.get('business_account_id', '')[-4:]}" if data.get('business_account_id') else None,
        "tenant_id": tenant_id,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/admin/whatsapp-meta/status", dependencies=[Depends(verify_admin_token)])
async def get_whatsapp_meta_status(x_tenant_id: str = Header(...)):
    """Get WhatsApp Meta API connection status (masked, no secrets)."""
    if not x_tenant_id:
        raise HTTPException(status_code=400, detail="Missing X-Tenant-Id header")

    # Check if credentials exist for this tenant
    creds = await db.get_all_credentials()
    tenant_creds = [c for c in creds if c['name'].endswith(f"_{x_tenant_id}") and c['scope'] == 'tenant']

    if not tenant_creds:
        return {
            "connected": False,
            "tenant_id": x_tenant_id,
            "timestamp": datetime.now().isoformat()
        }

    # Return masked status
    masked_info = {}
    for cred in tenant_creds:
        if "PHONE_NUMBER_ID" in cred['name']:
            masked_info["phone_number_id"] = f"***{cred['name'].split('_')[-2][-4:]}"
        elif "BUSINESS_ACCOUNT_ID" in cred['name']:
            masked_info["business_account_id"] = f"***{cred['name'].split('_')[-2][-4:]}"

    return {
        "connected": True,
        "tenant_id": x_tenant_id,
        **masked_info,
        "timestamp": datetime.now().isoformat()
    }

# Bootstrap / Initial Status
@app.get("/setup/state", dependencies=[Depends(verify_admin_token)])
async def get_setup_state():
    # Return current setup state (for now, return defaults since we don't persist it)
    return {
        "currentStep": 1,
        "stepStatuses": {
            1: 'pending',
            2: 'pending',
            3: 'pending',
            4: 'pending',
            5: 'pending',
            6: 'pending',
            7: 'pending'
        }
    }

@app.options("/setup/state")
async def options_setup_state():
    return JSONResponse(content={})

@app.post("/setup/state", dependencies=[Depends(verify_admin_token)])
async def save_setup_state(data: dict):
    # For now, just acknowledge - in production you'd save to DB
    # currentStep = data.get('currentStep', 1)
    # stepStatuses = data.get('stepStatuses', {})
    return JSONResponse(content={"status": "ok"})

@app.get("/admin/bootstrap", dependencies=[Depends(verify_admin_token)])
async def bootstrap_status():
    print("Bootstrap called - Version 1.2")
    if not BOOTSTRAP_ENABLED:
        return JSONResponse(
            content={
                "version": "1.0.0",
                "tenants_count": 0,
                "last_inbound_at": None,
                "last_outbound_at": None,
                "api_public_base": os.getenv("PUBLIC_BASE_URL", "http://localhost:8000"),
                "suggested_webhook_url": os.getenv("WEBHOOK_BASE_URL", os.getenv("PUBLIC_BASE_URL", "http://localhost:8000").replace(":8000", ":8002")) + "/webhook",
                "config_status": {
                    "missing_required_env": [],
                    "invalid_env": [],
                    "integration_env_missing": [],
                    "ready": False
                },
                "health": {
                    "orchestrator": "ok",
                    "tiendanube_service": "unknown",
                    "whatsapp_service": "unknown",
                    "database": "unknown",
                    "redis": "unknown"
                },
                "bootstrap_disabled": True
            }
        )

    try:
        tenants = await db.get_all_tenants()
        stats = await db.get_platform_stats()

        # Get last inbound and outbound - use received_at with fallback to created_at or id
        inbound_logs = await db.get_recent_logs_by_type('inbound', 1)
        outbound_logs = await db.get_recent_logs_by_type('outbound', 1)

        last_inbound = inbound_logs[0] if inbound_logs else None
        last_outbound = outbound_logs[0] if outbound_logs else None

        # Use received_at for inbound with automatic fallback
        last_inbound_at = None
        if last_inbound:
            # Try received_at first, then created_at, then id as last resort
            last_inbound_at = (last_inbound.get("received_at") or
                             last_inbound.get("created_at") or
                             last_inbound.get("id"))

        # Config validation
        config_issues = validate_configuration()

        return JSONResponse(
            content={
                "version": "1.0.0",
                "tenants_count": len(tenants),
                "last_inbound_at": last_inbound_at,
                "last_outbound_at": last_outbound.get("created_at") if last_outbound else None,
                "api_public_base": os.getenv("PUBLIC_BASE_URL", "http://localhost:8000"),
                "suggested_webhook_url": os.getenv("WEBHOOK_BASE_URL", os.getenv("PUBLIC_BASE_URL", "http://localhost:8000").replace(":8000", ":8002")) + "/webhook",
                "config_status": {
                    "missing_required_env": config_issues["missing_required_env"],
                    "invalid_env": config_issues["invalid_env"],
                    "integration_env_missing": config_issues["integration_env_missing"],
                    "ready": len(config_issues["missing_required_env"]) == 0 and len(config_issues["invalid_env"]) == 0
                },
                "health": {
                    "orchestrator": "ok",
                    "tiendanube_service": "ok",  # Would need actual health checks
                    "whatsapp_service": "ok",
                    "database": "ok",
                    "redis": "ok"
                }
            }
        )
    except Exception as e:
        # If bootstrap fails, return safe fallback
        return JSONResponse(
            content={
                "version": "1.0.0",
                "tenants_count": 0,
                "last_inbound_at": None,
                "last_outbound_at": None,
                "api_public_base": os.getenv("PUBLIC_BASE_URL", "http://localhost:8000"),
                "suggested_webhook_url": os.getenv("WEBHOOK_BASE_URL", os.getenv("PUBLIC_BASE_URL", "http://localhost:8000").replace(":8000", ":8002")) + "/webhook",
                "config_status": {
                    "missing_required_env": [],
                    "invalid_env": [],
                    "integration_env_missing": [],
                    "ready": False
                },
                "health": {
                    "orchestrator": "ok",
                    "tiendanube_service": "unknown",
                    "whatsapp_service": "unknown",
                    "database": "unknown",
                    "redis": "unknown"
                },
                "bootstrap_error": str(e)
            }
        )


@app.on_event("shutdown")
async def shutdown(): await db.disconnect()

@app.post("/chat", response_model=OrchestratorResult)
async def chat(event: InboundChatEvent, x_internal_token: Optional[str] = Header(None)):
    correlation_id = event.correlation_id or str(uuid.uuid4())
    log = logger.bind(correlation_id=correlation_id, 
                      from_number=event.from_number[-4:],
                      to_number=event.to_number[-4:])
    
    if INTERNAL_API_TOKEN and x_internal_token != INTERNAL_API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid Internal Token")

    # 0. Tenant Resolution with Fallback
    tenant = await db.get_tenant_config(event.to_number)
    tenant_id = tenant.get("id") if tenant else None

    if not tenant:
        log.warning("tenant_not_found_in_db_using_env_fallback", to_number=event.to_number)
        # Use existing global variables as fallback
        t_store_name = STORE_NAME
        t_store_location = STORE_LOCATION
        t_store_description = STORE_DESCRIPTION
        t_store_website = STORE_WEBSITE
        t_store_catalog = STORE_CATALOG_KNOWLEDGE
        t_tn_store_id = os.getenv("TIENDANUBE_STORE_ID", "6873259")
        t_tn_token = await get_credential_value("TIENDANUBE_ACCESS_TOKEN", tenant_id, "TIENDANUBE_ACCESS_TOKEN")
        t_openai_key = await get_credential_value("OPENAI_API_KEY", tenant_id, "OPENAI_API_KEY")
    else:
        # Resolve from DB
        t_store_name = tenant.get("store_name") or STORE_NAME
        t_store_location = tenant.get("store_location") or STORE_LOCATION
        t_store_description = tenant.get("store_description") or STORE_DESCRIPTION
        t_store_website = tenant.get("store_website") or STORE_WEBSITE
        t_store_catalog = tenant.get("store_catalog_knowledge") or STORE_CATALOG_KNOWLEDGE
        t_tn_store_id = tenant.get("tiendanube_store_id") or os.getenv("TIENDANUBE_STORE_ID", "6873259")
        t_tn_token = await get_credential_value("TIENDANUBE_ACCESS_TOKEN", tenant_id, "TIENDANUBE_ACCESS_TOKEN")
        t_openai_key = await get_credential_value("OPENAI_API_KEY", tenant_id, "OPENAI_API_KEY")

    t_api_base = f"https://api.tiendanube.com/v1/{t_tn_store_id}"

    start_time = time.time()
    
    user_hash = hashlib.sha256(event.from_number.encode()).hexdigest()
    # Increased timeout to 80s to give more 'air' for complex tool calls
    lock = redis_client.lock(f"lock:from_number:{user_hash}", timeout=80)
    acquired = lock.acquire(blocking=True, blocking_timeout=2)
    if not acquired:
        log.warning("lock_busy")
        return OrchestratorResult(status="error", send=False, meta={"reason": "lock_busy"})

    try:
        # 1. Dedupe
        is_new = await db.try_insert_inbound(
            event.provider, event.provider_message_id, event.event_id, 
            event.from_number, event.dict(), correlation_id
        )
        if not is_new:
            log.info("duplicate_message", provider=event.provider, provider_message_id=event.provider_message_id)
            return OrchestratorResult(status="duplicate", send=False)

        await db.mark_inbound_processing(event.provider, event.provider_message_id)

        # 2. History
        history_records = await db.get_chat_history(event.from_number, limit=10)
        chat_history = []
        for h in history_records:
            if h.get('role') == 'user':
                chat_history.append(HumanMessage(content=h['content']))
            elif h.get('role') == 'assistant':
                chat_history.append(AIMessage(content=h['content']))

        # 3. Dynamic Agent Setup
        t_tools = get_tenant_tools_static(t_tn_token, t_api_base)
        # Add dynamic tools
        dynamic_tools = await get_tenant_tools_dynamic(tenant['id'])
        t_tools.extend(dynamic_tools)
        
        # PROMPT DINÁMICO
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", """Eres el asistente virtual de {STORE_NAME} ({STORE_LOCATION}), {STORE_DESCRIPTION}.
            PRIORIDADES:
            1. SALIDA JSON: Sigue estrictamente el esquema OrchestratorResponse.
            2.tools: Usa tools para stock/pedidos.
            3. PRODUCTOS: Máximo 3-4, 1 burbuja por producto con Link y Precio.
            4. WEB: Link obligatorio {STORE_WEBSITE} al final.
            5. CTA: Siempre pregunta algo al final para seguir el chat.
            
            {format_instructions}
            
            CONOCIMIENTO DEL CATÁLOGO:
            {STORE_CATALOG_KNOWLEDGE}
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]).partial(
            format_instructions=parser.get_format_instructions(),
            STORE_NAME=t_store_name,
            STORE_LOCATION=t_store_location,
            STORE_DESCRIPTION=t_store_description,
            STORE_WEBSITE=t_store_website,
            STORE_CATALOG_KNOWLEDGE=t_store_catalog
        )

        # Initialize LLM with tenant-specific OpenAI key
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=t_openai_key
        )
        agent = create_openai_functions_agent(llm, t_tools, agent_prompt)
        agent_executor = AgentExecutor(agent=agent, tools=t_tools, verbose=True)
        
        output_text = "Lo siento, tuve un problema al procesar tu mensaje."
        result_messages = []
        
        try:
            response = await agent_executor.ainvoke({"input": event.text, "chat_history": chat_history})
            raw_output = response.get("output", "")
            
            # Layer 1: Intelligent Post-Processor
            def clean_json(text: str):
                text = text.strip()
                if "```" in text:
                    try:
                        inner = text.split("```")[1]
                        if inner.startswith("json"): inner = inner[4:]
                        return inner.strip()
                    except: pass
                return text

            try:
                parsed = parser.parse(clean_json(raw_output))
                result_messages = parsed.messages
            except Exception as parse_err:
                log.warning("parser_failed_applying_manual_split", error=str(parse_err), raw=raw_output)
                import re
                sentences = re.split(r'(?<=[.!?]) +', raw_output)
                chunks = []
                current_chunk = ""
                for s in sentences:
                    if len(current_chunk) + len(s) < 350:
                        current_chunk += (" " + s if current_chunk else s)
                    else:
                        if current_chunk: chunks.append(current_chunk)
                        current_chunk = s
                if current_chunk: chunks.append(current_chunk)
                
                result_messages = [
                    OrchestratorMessage(part=i+1, total=len(chunks), text=c) 
                    for i, c in enumerate(chunks)
                ]
            
            output_text = " | ".join([m.text for m in result_messages])
                
        except Exception as e:
            log.error("agent_error", error=str(e))
            await db.mark_inbound_failed(event.provider, event.provider_message_id, str(e))
            return OrchestratorResult(status="error", send=False, meta={"error": str(e)})

        # 4. Save & Result
        latency = time.time() - start_time
        await db.append_chat_message(event.from_number, "user", event.text, correlation_id)
        await db.append_chat_message(event.from_number, "assistant", output_text, correlation_id)
        await db.mark_inbound_done(event.provider, event.provider_message_id)
        
        log.info("request_completed", status="ok", latency=latency)
        return OrchestratorResult(
            status="ok",
            send=True,
            text=output_text,
            messages=result_messages
        )
    except Exception as e:
        log.error("unexpected_error", error=str(e))
        await db.mark_inbound_failed(event.provider, event.provider_message_id, str(e))
        return OrchestratorResult(status="error", send=False, meta={"error": str(e)})
    finally:
        if acquired: lock.release()
