import os
import json
from typing import List, Optional, Any
from fastapi import APIRouter, Header, HTTPException, Depends, Request
from pydantic import BaseModel
from db import db

# Configuration
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "admin-secret-99")

router = APIRouter(prefix="/admin", tags=["admin"])

# --- Security ---
async def verify_admin_token(x_admin_token: str = Header(None)):
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid Admin Token")

# --- Models ---
class TenantModel(BaseModel):
    store_name: str
    bot_phone_number: str
    owner_email: Optional[str] = None
    store_location: Optional[str] = None
    store_website: Optional[str] = None
    store_description: Optional[str] = None
    store_catalog_knowledge: Optional[str] = None
    tiendanube_store_id: Optional[str] = None
    tiendanube_access_token: Optional[str] = None

class CredentialModel(BaseModel):
    name: str
    value: str
    category: str
    scope: str = "global"
    tenant_id: Optional[int] = None
    description: Optional[str] = None

# --- Endpoints ---

@router.get("/bootstrap", dependencies=[Depends(verify_admin_token)])
async def bootstrap():
    """Initial load for the dashboard."""
    # Get tenants count
    tenants = await db.pool.fetchval("SELECT COUNT(*) FROM tenants")
    
    # Get last activity
    last_inbound = await db.pool.fetchval("SELECT MAX(received_at) FROM inbound_messages")
    last_outbound = await db.pool.fetchval("SELECT MAX(created_at) FROM chat_messages WHERE role = 'assistant'")
    
    return {
        "version": "1.0.0",
        "tenants_count": tenants,
        "last_inbound_at": last_inbound,
        "last_outbound_at": last_outbound,
        "status": "ok"
    }

@router.get("/stats", dependencies=[Depends(verify_admin_token)])
async def get_stats():
    """Get dashboard statistics."""
    # Active tenants (with ID)
    active_tenants = await db.pool.fetchval("SELECT COUNT(*) FROM tenants WHERE tiendanube_store_id IS NOT NULL")
    
    # Message stats
    total_messages = await db.pool.fetchval("SELECT COUNT(*) FROM inbound_messages")
    processed_messages = await db.pool.fetchval("SELECT COUNT(*) FROM inbound_messages WHERE status = 'done'")
    
    return {
        "active_tenants": active_tenants,
        "total_messages": total_messages,
        "processed_messages": processed_messages
    }

@router.get("/logs", dependencies=[Depends(verify_admin_token)])
async def get_logs(limit: int = 50):
    """Fetch recent chat logs for the 'Live History' view."""
    # Using chat_messages as the source of truth for display
    # Join with inbound status if needed, but for now simple chat log is enough
    rows = await db.pool.fetch("""
        SELECT 
            cm.from_number,
            cm.role,
            cm.content,
            cm.created_at,
            im.status as inbound_status
        FROM chat_messages cm
        LEFT JOIN inbound_messages im ON cm.correlation_id = im.correlation_id AND cm.role = 'user'
        ORDER BY cm.created_at DESC
        LIMIT $1
    """, limit)
    
    # Adapt to frontend format (it expects 'payload' JSON usually, but we'll try to match app.js logic)
    # app.js expects: received_at, from_number, to_number, status, payload (json or string), ai_response
    
    logs = []
    # This is a bit tricky because the frontend expects a combined view (User + AI) in one row 
    # OR separate rows. app.js logic:
    # entry.innerHTML = `[${time}] User: ... | Status: ... <br> User: ... AI: ...`
    
    # Let's group by correlation_id to simulate sessions if possible, or just return flat list
    # For compatibility with the specific app.js provided which parses 'payload':
    
    for row in rows:
        logs.append({
            "received_at": row['created_at'].isoformat(),
            "from_number": row['from_number'],
            "to_number": "Bot",
            "status": row['inbound_status'] or "sent",
            "payload": json.dumps({"text": row['content']}), # Wrap in JSON as app.js tries to parse it
            "ai_response": None # We send individual messages, so AI response is its own row usually. 
                                # But app.js logic looks for ai_response property.
                                # To make it look nice, we might need a better query grouping user+assistant.
        })
    
    return logs

# --- Multi-Tenancy Routes ---

@router.get("/tenants", dependencies=[Depends(verify_admin_token)])
async def list_tenants():
    rows = await db.pool.fetch("SELECT * FROM tenants ORDER BY id DESC")
    return [dict(row) for row in rows]

@router.post("/tenants", dependencies=[Depends(verify_admin_token)])
async def create_tenant(tenant: TenantModel):
    # Upsert based on phone number
    q = """
        INSERT INTO tenants (store_name, bot_phone_number, owner_email, store_location, store_website, store_description, store_catalog_knowledge, tiendanube_store_id, tiendanube_access_token)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT (bot_phone_number) 
        DO UPDATE SET 
            store_name = EXCLUDED.store_name,
            owner_email = EXCLUDED.owner_email,
            store_location = EXCLUDED.store_location,
            store_website = EXCLUDED.store_website,
            store_description = EXCLUDED.store_description,
            store_catalog_knowledge = EXCLUDED.store_catalog_knowledge,
            tiendanube_store_id = EXCLUDED.tiendanube_store_id,
            tiendanube_access_token = EXCLUDED.tiendanube_access_token,
            updated_at = NOW()
        RETURNING id
    """
    row = await db.pool.fetchrow(q, tenant.store_name, tenant.bot_phone_number, tenant.owner_email, tenant.store_location, tenant.store_website, tenant.store_description, tenant.store_catalog_knowledge, tenant.tiendanube_store_id, tenant.tiendanube_access_token)
    return {"status": "ok", "id": row['id']}

@router.get("/tenants/{phone}", dependencies=[Depends(verify_admin_token)])
async def get_tenant(phone: str):
    row = await db.pool.fetchrow("SELECT * FROM tenants WHERE bot_phone_number = $1", phone)
    if not row:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return dict(row)

@router.delete("/tenants/{phone}", dependencies=[Depends(verify_admin_token)])
async def delete_tenant(phone: str):
    await db.pool.execute("DELETE FROM tenants WHERE bot_phone_number = $1", phone)
    return {"status": "ok"}

# --- Credentials Routes ---

@router.get("/credentials", dependencies=[Depends(verify_admin_token)])
async def list_credentials():
    # Join with tenants to get tenant name
    q = """
        SELECT c.*, t.store_name as tenant_name 
        FROM credentials c 
        LEFT JOIN tenants t ON c.tenant_id = t.id 
        ORDER BY c.id DESC
    """
    rows = await db.pool.fetch(q)
    return [dict(row) for row in rows]

@router.get("/credentials/{id}", dependencies=[Depends(verify_admin_token)])
async def get_credential(id: int):
    row = await db.pool.fetchrow("SELECT c.*, t.store_name as tenant_name FROM credentials c LEFT JOIN tenants t ON c.tenant_id = t.id WHERE c.id = $1", id)
    if not row:
        raise HTTPException(status_code=404, detail="Credential not found")
    return dict(row)

@router.post("/credentials", dependencies=[Depends(verify_admin_token)])
async def create_credential(cred: CredentialModel):
    # TODO: Encryption in real prod
    q = """
        INSERT INTO credentials (name, value, category, scope, tenant_id, description)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING id
    """
    row = await db.pool.fetchrow(q, cred.name, cred.value, cred.category, cred.scope, cred.tenant_id, cred.description)
    return {"status": "ok", "id": row['id']}

@router.delete("/credentials/{id}", dependencies=[Depends(verify_admin_token)])
async def delete_credential(id: int):
    await db.pool.execute("DELETE FROM credentials WHERE id = $1", id)
    return {"status": "ok"}
