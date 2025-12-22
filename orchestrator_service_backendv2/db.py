import asyncpg
import os
import json
from typing import List, Tuple, Optional

POSTGRES_DSN = os.getenv("POSTGRES_DSN")

class Database:
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        if not self.pool:
            self.pool = await asyncpg.create_pool(POSTGRES_DSN)

    async def disconnect(self):
        if self.pool:
            await self.pool.close()

    async def initialize(self):
        """Create tables if they don't exist."""
        queries = [
            # Tables first
            """
            CREATE TABLE IF NOT EXISTS tenants (
                id SERIAL PRIMARY KEY,
                bot_phone_number VARCHAR(50) UNIQUE NOT NULL,
                owner_email VARCHAR(255),
                store_name VARCHAR(255),
                store_description TEXT,
                store_location TEXT,
                store_website VARCHAR(255),
                store_catalog_knowledge TEXT,
                tiendanube_store_id VARCHAR(50),
                tiendanube_access_token TEXT,
                ycloud_api_key TEXT,
                ycloud_webhook_secret TEXT,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS tiendanube_stores (
                id SERIAL PRIMARY KEY,
                tenant_id INTEGER REFERENCES tenants(id),
                store_id VARCHAR(50) NOT NULL,
                access_token TEXT NOT NULL,
                store_name VARCHAR(255),
                status VARCHAR(20) DEFAULT 'inactive' CHECK (status IN ('active', 'inactive')),
                last_test_at TIMESTAMP,
                last_test_ok BOOLEAN,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(tenant_id, store_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS telemetry_events (
                id BIGSERIAL PRIMARY KEY,
                occurred_at TIMESTAMP DEFAULT NOW(),
                tenant_id INTEGER REFERENCES tenants(id),
                store_id VARCHAR(50),
                event_type VARCHAR(100) NOT NULL,
                severity VARCHAR(20) DEFAULT 'info' CHECK (severity IN ('debug', 'info', 'warn', 'error')),
                session_id VARCHAR(100),
                order_id VARCHAR(100),
                correlation_id VARCHAR(100),
                success BOOLEAN,
                duration_ms INTEGER,
                error_code VARCHAR(50),
                error_message TEXT,
                payload JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS inbound_messages (
                id SERIAL PRIMARY KEY,
                provider VARCHAR(50),
                provider_message_id VARCHAR(255),
                event_id VARCHAR(255),
                from_number VARCHAR(50),
                payload TEXT,
                status VARCHAR(50),
                error TEXT,
                correlation_id VARCHAR(255),
                created_at TIMESTAMP DEFAULT NOW(),
                processed_at TIMESTAMP,
                UNIQUE(provider, provider_message_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id SERIAL PRIMARY KEY,
                from_number VARCHAR(50),
                role VARCHAR(20),
                content TEXT,
                correlation_id VARCHAR(255),
                created_at TIMESTAMP DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS credentials (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                value TEXT NOT NULL,
                description TEXT,
                category VARCHAR(100),
                scope VARCHAR(50) NOT NULL CHECK (scope IN ('global', 'tenant')),
                tenant_id INTEGER REFERENCES tenants(id),
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(name, tenant_id)
            )
            """,
            # Add missing columns if they don't exist (for existing databases)
            """
            DO $$
            BEGIN
                -- Add description column if it doesn't exist
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'credentials' AND column_name = 'description') THEN
                    ALTER TABLE credentials ADD COLUMN description TEXT;
                END IF;

                -- Add category column if it doesn't exist
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'credentials' AND column_name = 'category') THEN
                    ALTER TABLE credentials ADD COLUMN category VARCHAR(100);
                END IF;

                -- Add updated_at column if it doesn't exist
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'credentials' AND column_name = 'updated_at') THEN
                    ALTER TABLE credentials ADD COLUMN updated_at TIMESTAMP DEFAULT NOW();
                END IF;
            END $$;
            """,
            """
            CREATE TABLE IF NOT EXISTS tools (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                type VARCHAR(50) NOT NULL,
                config JSONB NOT NULL,
                service_url VARCHAR(255),
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS tenant_tools (
                id SERIAL PRIMARY KEY,
                tenant_id INTEGER REFERENCES tenants(id),
                tool_id INTEGER REFERENCES tools(id),
                is_enabled BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(tenant_id, tool_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS setup_sessions (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(100) UNIQUE NOT NULL,
                current_step VARCHAR(50) DEFAULT 'preflight',
                status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'completed', 'failed', 'abandoned')),
                public_base_url VARCHAR(255),
                webhook_base_url VARCHAR(255),
                infrastructure_checks JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                completed_at TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS setup_events (
                id BIGSERIAL PRIMARY KEY,
                session_id INTEGER REFERENCES setup_sessions(id),
                event_type VARCHAR(100) NOT NULL,
                step VARCHAR(50),
                status VARCHAR(20) CHECK (status IN ('started', 'completed', 'failed', 'skipped')),
                message TEXT,
                details JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT NOW()
            )
            """,
            # Then indexes
            "CREATE INDEX IF NOT EXISTS idx_chat_messages_from_number ON chat_messages(from_number)",
            """
            CREATE INDEX IF NOT EXISTS idx_telemetry_events_tenant_occurred ON telemetry_events (tenant_id, occurred_at DESC);
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_telemetry_events_type_occurred ON telemetry_events (event_type, occurred_at DESC);
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_telemetry_events_session_occurred ON telemetry_events (session_id, occurred_at DESC);
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_telemetry_events_order ON telemetry_events (order_id);
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_telemetry_events_correlation ON telemetry_events (correlation_id);
            """
        ]
        async with self.pool.acquire() as conn:
            for q in queries:
                await conn.execute(q)

    async def get_tenant_config(self, bot_phone_number: str) -> Optional[dict]:
        """Fetch branding and credentials for a specific tenant bot number."""
        query = "SELECT * FROM tenants WHERE bot_phone_number = $1 AND is_active = TRUE"
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, bot_phone_number)
            return dict(row) if row else None

    async def get_tenant_config_by_id(self, tenant_id: int) -> Optional[dict]:
        """Fetch branding and credentials for a specific tenant by ID."""
        query = "SELECT * FROM tenants WHERE id = $1 AND is_active = TRUE"
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, tenant_id)
            return dict(row) if row else None

    async def delete_all_tenants(self):
        """Delete all tenants - USE WITH CAUTION"""
        query = "DELETE FROM tenants"
        async with self.pool.acquire() as conn:
            await conn.execute(query)

    # Tienda Nube Stores methods
    async def get_tiendanube_stores(self, tenant_id: int = None) -> List[dict]:
        """Get Tienda Nube stores for a tenant or all if no tenant specified"""
        query = """
        SELECT id, tenant_id, store_id, store_name, status, last_test_at, last_test_ok, error_message, created_at
        FROM tiendanube_stores
        WHERE tenant_id = $1 OR $1 IS NULL
        ORDER BY created_at DESC
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, tenant_id)
            return [dict(row) for row in rows]

    async def create_tiendanube_store(self, tenant_id: int, store_id: str, access_token: str, store_name: str = None) -> int:
        """Create a new Tienda Nube store"""
        query = """
        INSERT INTO tiendanube_stores (tenant_id, store_id, access_token, store_name)
        VALUES ($1, $2, $3, $4)
        RETURNING id
        """
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, tenant_id, store_id, access_token, store_name)

    async def update_tiendanube_store_status(self, store_id: int, status: str, last_test_ok: bool = None, error_message: str = None):
        """Update store status and test results"""
        query = """
        UPDATE tiendanube_stores
        SET status = $2, last_test_at = NOW(), last_test_ok = $3, error_message = $4, updated_at = NOW()
        WHERE id = $1
        """
        async with self.pool.acquire() as conn:
            await conn.execute(query, store_id, status, last_test_ok, error_message)

    # Telemetry methods
    async def insert_telemetry_event(self, event_data: dict):
        """Insert a telemetry event"""
        # Import here to avoid circular import
        from main import sanitize_payload

        query = """
        INSERT INTO telemetry_events (
            tenant_id, store_id, event_type, severity, session_id, order_id,
            correlation_id, success, duration_ms, error_code, error_message, payload
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        """
        async with self.pool.acquire() as conn:
            await conn.execute(query,
                event_data.get('tenant_id'),
                event_data.get('store_id'),
                event_data['event_type'],
                event_data.get('severity', 'info'),
                event_data.get('session_id'),
                event_data.get('order_id'),
                event_data.get('correlation_id'),
                event_data.get('success'),
                event_data.get('duration_ms'),
                event_data.get('error_code'),
                sanitize_payload(event_data.get('error_message')) if event_data.get('error_message') else None,
                sanitize_payload(event_data.get('payload', {}))
            )

    async def get_analytics_summary(self, tenant_id: int, store_id: str = None, from_date: str = None, to_date: str = None):
        """Get analytics summary KPIs"""
        # Build date filter
        date_filter = ""
        params = [tenant_id]
        if from_date:
            date_filter += " AND occurred_at >= $2"
            params.append(from_date)
        if to_date:
            date_filter += f" AND occurred_at <= ${len(params) + 1}"
            params.append(to_date)
        if store_id:
            date_filter += f" AND store_id = ${len(params) + 1}"
            params.append(store_id)

        # Calculate KPIs
        kpis = {}

        # Conversations
        conv_query = f"""
        SELECT COUNT(DISTINCT session_id) as value
        FROM telemetry_events
        WHERE tenant_id = $1 AND event_type = 'message.incoming'{date_filter}
        """
        conv_result = await self.pool.fetchval(conv_query, *params)
        kpis['conversations'] = {'value': conv_result or 0}

        # Assisted by intent
        intent_query = f"""
        SELECT payload->>'intent' as intent, COUNT(*) as value
        FROM telemetry_events
        WHERE tenant_id = $1 AND event_type = 'assistant.intent.detected'{date_filter}
        GROUP BY payload->>'intent'
        ORDER BY value DESC
        LIMIT 5
        """
        intent_results = await self.pool.fetch(intent_query, *params)
        kpis['assisted_by_intent'] = {
            'top': [{'key': r['intent'] or 'unknown', 'value': r['value']} for r in intent_results],
            'total': sum(r['value'] for r in intent_results)
        }

        # Orders lookup
        orders_req_query = f"""
        SELECT COUNT(*) as value
        FROM telemetry_events
        WHERE tenant_id = $1 AND event_type = 'order.lookup.requested'{date_filter}
        """
        orders_req = await self.pool.fetchval(orders_req_query, *params) or 0

        orders_succ_query = f"""
        SELECT COUNT(*) as value
        FROM telemetry_events
        WHERE tenant_id = $1 AND event_type = 'order.lookup.succeeded'{date_filter}
        """
        orders_succ = await self.pool.fetchval(orders_succ_query, *params) or 0

        kpis['orders_lookup'] = {
            'requested': orders_req,
            'succeeded': orders_succ,
            'success_rate': (orders_succ / orders_req * 100) if orders_req > 0 else 0
        }

        # Handoffs
        handoff_query = f"""
        SELECT COUNT(*) as value
        FROM telemetry_events
        WHERE tenant_id = $1 AND event_type = 'assistant.handoff.sent' AND success = true{date_filter}
        """
        handoffs = await self.pool.fetchval(handoff_query, *params) or 0

        handoff_reasons_query = f"""
        SELECT payload->>'reason' as reason, COUNT(*) as value
        FROM telemetry_events
        WHERE tenant_id = $1 AND event_type = 'assistant.handoff.sent'{date_filter}
        GROUP BY payload->>'reason'
        ORDER BY value DESC
        LIMIT 5
        """
        handoff_reasons = await self.pool.fetch(handoff_reasons_query, *params)

        kpis['handoffs'] = {
            'value': handoffs,
            'top_reasons': [{'key': r['reason'] or 'unknown', 'value': r['value']} for r in handoff_reasons]
        }

        # Errors
        errors_query = f"""
        SELECT COUNT(*) as value
        FROM telemetry_events
        WHERE tenant_id = $1 AND event_type IN ('api.error', 'setup.step.failed'){date_filter}
        """
        errors = await self.pool.fetchval(errors_query, *params) or 0

        error_codes_query = f"""
        SELECT error_code, COUNT(*) as value
        FROM telemetry_events
        WHERE tenant_id = $1 AND event_type IN ('api.error', 'setup.step.failed'){date_filter}
        GROUP BY error_code
        ORDER BY value DESC
        LIMIT 5
        """
        error_codes = await self.pool.fetch(error_codes_query, *params)

        kpis['errors'] = {
            'value': errors,
            'rate': (errors / (kpis['conversations']['value'] or 1) * 100),
            'top_error_codes': [{'key': r['error_code'] or 'unknown', 'value': r['value']} for r in error_codes]
        }

        # Sales confirmed (placeholder)
        kpis['sales_confirmed'] = {
            'value': 0,
            'amount': 0,
            'currency': 'ARS',
            'is_available': False
        }

        return {
            'range': {'from': from_date, 'to': to_date},
            'last_updated_at': '2025-01-01T00:00:00Z',  # Placeholder
            'kpis': kpis
        }

    async def get_analytics_timeseries(self, tenant_id: int, metric: str, store_id: str = None, from_date: str = None, to_date: str = None, bucket: str = 'day'):
        """Get timeseries data for a metric"""
        # This is a simplified implementation
        # In production, you'd use proper time bucketing
        return {
            'metric': metric,
            'bucket': bucket,
            'points': []  # Placeholder
        }

    async def get_analytics_breakdown(self, tenant_id: int, dimension: str, store_id: str = None, from_date: str = None, to_date: str = None, limit: int = 10):
        """Get breakdown by dimension"""
        # Simplified implementation
        return {
            'dimension': dimension,
            'items': []
        }

    async def get_telemetry_events(self, tenant_id: int, store_id: str = None, from_date: str = None, to_date: str = None,
                                  event_type: str = None, severity: str = None, session_id: str = None,
                                  order_id: str = None, page: int = 1, page_size: int = 50):
        """Get paginated telemetry events"""
        # Build query
        conditions = ["tenant_id = $1"]
        params = [tenant_id]
        param_count = 1

        if store_id:
            param_count += 1
            conditions.append(f"store_id = ${param_count}")
            params.append(store_id)
        if from_date:
            param_count += 1
            conditions.append(f"occurred_at >= ${param_count}")
            params.append(from_date)
        if to_date:
            param_count += 1
            conditions.append(f"occurred_at <= ${param_count}")
            params.append(to_date)
        if event_type:
            param_count += 1
            conditions.append(f"event_type = ${param_count}")
            params.append(event_type)
        if severity:
            param_count += 1
            conditions.append(f"severity = ${param_count}")
            params.append(severity)
        if session_id:
            param_count += 1
            conditions.append(f"session_id = ${param_count}")
            params.append(session_id)
        if order_id:
            param_count += 1
            conditions.append(f"order_id = ${param_count}")
            params.append(order_id)

        where_clause = " AND ".join(conditions)

        # Count total
        count_query = f"SELECT COUNT(*) FROM telemetry_events WHERE {where_clause}"
        total = await self.pool.fetchval(count_query, *params)

        # Get paginated results
        offset = (page - 1) * page_size
        data_query = f"""
        SELECT occurred_at, event_type, severity, session_id, order_id, success, duration_ms, error_code, error_message, payload
        FROM telemetry_events
        WHERE {where_clause}
        ORDER BY occurred_at DESC
        LIMIT ${param_count + 1} OFFSET ${param_count + 2}
        """
        params.extend([page_size, offset])
        rows = await self.pool.fetch(data_query, *params)

        return {
            'page': page,
            'page_size': page_size,
            'total': total,
            'items': [dict(row) for row in rows]
        }

    async def try_insert_inbound(self, provider: str, provider_message_id: str, event_id: str, from_number: str, payload: dict, correlation_id: str) -> bool:
        """Try to insert inbound message. Returns True if inserted, False if duplicate."""
        # Import here to avoid circular import
        from main import sanitize_payload

        # Sanitize payload before storing
        safe_payload = sanitize_payload(payload)

        query = """
        INSERT INTO inbound_messages (provider, provider_message_id, event_id, from_number, payload, status, correlation_id)
        VALUES ($1, $2, $3, $4, $5, 'received', $6)
        ON CONFLICT (provider, provider_message_id) DO NOTHING
        RETURNING id
        """
        async with self.pool.acquire() as conn:
            result = await conn.fetchval(query, provider, provider_message_id, event_id, from_number, json.dumps(safe_payload), correlation_id)
            return result is not None

    async def mark_inbound_processing(self, provider: str, provider_message_id: str):
        query = "UPDATE inbound_messages SET status = 'processing' WHERE provider = $1 AND provider_message_id = $2"
        async with self.pool.acquire() as conn:
            await conn.execute(query, provider, provider_message_id)

    async def mark_inbound_done(self, provider: str, provider_message_id: str):
        query = "UPDATE inbound_messages SET status = 'done', processed_at = NOW() WHERE provider = $1 AND provider_message_id = $2"
        async with self.pool.acquire() as conn:
            await conn.execute(query, provider, provider_message_id)

    async def mark_inbound_failed(self, provider: str, provider_message_id: str, error: str):
        query = "UPDATE inbound_messages SET status = 'failed', processed_at = NOW(), error = $3 WHERE provider = $1 AND provider_message_id = $2"
        async with self.pool.acquire() as conn:
            await conn.execute(query, provider, provider_message_id, error)

    async def append_chat_message(self, from_number: str, role: str, content: str, correlation_id: str):
        query = "INSERT INTO chat_messages (from_number, role, content, correlation_id) VALUES ($1, $2, $3, $4)"
        async with self.pool.acquire() as conn:
            await conn.execute(query, from_number, role, content, correlation_id)

    async def get_chat_history(self, from_number: str, limit: int = 15) -> List[dict]:
        """Returns list of {'role': ..., 'content': ...} in chronological order."""
        query = "SELECT role, content FROM chat_messages WHERE from_number = $1 ORDER BY created_at DESC LIMIT $2"
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, from_number, limit)
            return [dict(row) for row in reversed(rows)]

    async def get_all_tenants(self) -> List[dict]:
        """Fetch all tenants for the admin dashboard."""
        query = "SELECT * FROM tenants ORDER BY created_at DESC"
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query)
            return [dict(row) for row in rows]

    async def upsert_tenant(self, data: dict):
        """Create or update a tenant config."""
        query = """
        INSERT INTO tenants (
            bot_phone_number, owner_email, store_name, store_description,
            store_location, store_website, store_catalog_knowledge,
            tiendanube_store_id, tiendanube_access_token, ycloud_webhook_secret, is_active
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        ON CONFLICT (bot_phone_number) DO UPDATE SET
            owner_email = EXCLUDED.owner_email,
            store_name = EXCLUDED.store_name,
            store_description = EXCLUDED.store_description,
            store_location = EXCLUDED.store_location,
            store_website = EXCLUDED.store_website,
            store_catalog_knowledge = EXCLUDED.store_catalog_knowledge,
            tiendanube_store_id = EXCLUDED.tiendanube_store_id,
            tiendanube_access_token = EXCLUDED.tiendanube_access_token,
            ycloud_webhook_secret = EXCLUDED.ycloud_webhook_secret,
            is_active = EXCLUDED.is_active
        """
        async with self.pool.acquire() as conn:
            await conn.execute(query,
                data.get('bot_phone_number'), data.get('owner_email'), data.get('store_name'),
                data.get('store_description'), data.get('store_location'), data.get('store_website'),
                data.get('store_catalog_knowledge'), data.get('tiendanube_store_id'),
                data.get('tiendanube_access_token'), data.get('ycloud_webhook_secret'),
                data.get('is_active', True)
            )


    async def get_platform_stats(self):
        """Retrieve high-level metrics."""
        query = """
        SELECT 
            (SELECT COUNT(*) FROM tenants WHERE is_active = TRUE) as active_tenants,
            (SELECT COUNT(*) FROM inbound_messages) as total_messages,
            (SELECT COUNT(*) FROM inbound_messages WHERE status = 'done') as processed_messages,
            (SELECT COUNT(*) FROM inbound_messages WHERE status = 'failed') as failed_messages
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query)
            return dict(row)

    async def get_recent_logs(self, limit: int = 50):
        """Fetch recent processed messages for the live feed."""
        query = """
        SELECT im.received_at, im.from_number, im.to_number, im.status, im.payload,
               (SELECT content FROM chat_messages WHERE correlation_id = im.correlation_id AND role = 'assistant' LIMIT 1) as ai_response
        FROM inbound_messages im
        ORDER BY im.received_at DESC
        LIMIT $1
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, limit)
            return [dict(row) for row in rows]

    async def get_recent_logs_by_type(self, event_type: str, limit: int = 1):
        """Fetch recent logs by event type (inbound/outbound)."""
        # For now, assume inbound are from webhook, outbound are responses
        if event_type == 'inbound':
            # Check if received_at column exists for backward compatibility
            check_query = """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'inbound_messages' AND column_name = 'received_at'
            )
            """
            async with self.pool.acquire() as conn:
                has_received_at = await conn.fetchval(check_query)
                if has_received_at:
                    query = """
                    SELECT received_at FROM inbound_messages
                    ORDER BY received_at DESC LIMIT $1
                    """
                else:
                    query = """
                    SELECT created_at as received_at FROM inbound_messages
                    ORDER BY created_at DESC LIMIT $1
                    """
                rows = await conn.fetch(query, limit)
                return [dict(row) for row in rows]
        else:  # outbound
            query = """
            SELECT created_at FROM chat_messages WHERE role = 'assistant'
            ORDER BY created_at DESC LIMIT $1
            """
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, limit)
                return [dict(row) for row in rows]

    async def get_all_credentials(self) -> List[dict]:
        """Fetch all credentials for the admin dashboard."""
        query = """
        SELECT c.id, c.name, c.description, c.category, c.scope, c.tenant_id, c.created_at, c.updated_at,
               t.store_name as tenant_name
        FROM credentials c
        LEFT JOIN tenants t ON c.tenant_id = t.id
        ORDER BY c.created_at DESC
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query)
            return [dict(row) for row in rows]

    async def get_credentials_by_name(self, name: str, tenant_id: int = None) -> Optional[dict]:
        """Get a credential by name, optionally scoped to a tenant."""
        if tenant_id:
            # First try tenant-specific credential
            query = "SELECT * FROM credentials WHERE name = $1 AND tenant_id = $2"
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(query, name, tenant_id)
                if row:
                    return dict(row)
        # Fall back to global credential
        query = "SELECT * FROM credentials WHERE name = $1 AND (tenant_id IS NULL OR scope = 'global')"
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, name)
            return dict(row) if row else None

    async def get_credential_by_id(self, credential_id: int) -> Optional[dict]:
        """Get a credential by ID."""
        query = "SELECT * FROM credentials WHERE id = $1"
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, credential_id)
            return dict(row) if row else None

    async def delete_credential(self, credential_id: int):
        """Delete a credential by ID."""
        query = "DELETE FROM credentials WHERE id = $1"
        async with self.pool.acquire() as conn:
            await conn.execute(query, credential_id)

    async def get_credentials_by_tenant(self, tenant_id: int) -> List[dict]:
        """Get all credentials for a specific tenant."""
        query = "SELECT id, name, scope, created_at FROM credentials WHERE tenant_id = $1 ORDER BY created_at DESC"
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, tenant_id)
            return [dict(row) for row in rows]

    async def get_global_credentials(self) -> List[dict]:
        """Get all global credentials."""
        query = "SELECT id, name, scope, created_at FROM credentials WHERE tenant_id IS NULL AND scope = 'global' ORDER BY created_at DESC"
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query)
            return [dict(row) for row in rows]

    async def get_tenant_config_by_id(self, tenant_id: int) -> Optional[dict]:
        """Get tenant config by ID."""
        query = "SELECT * FROM tenants WHERE id = $1"
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, tenant_id)
            return dict(row) if row else None

    async def upsert_credential(self, data: dict):
        """Create or update a credential."""
        # Handle tenant_id conversion for global credentials
        tenant_id = data.get('tenant_id')
        if tenant_id is not None and tenant_id != '':
            try:
                tenant_id = int(tenant_id)
            except (ValueError, TypeError):
                tenant_id = None

        # Ensure description and category are strings or None
        description = data.get('description')
        if description == '':
            description = None
        category = data.get('category')
        if category == '':
            category = None

        query = """
        INSERT INTO credentials (name, value, description, category, scope, tenant_id)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (name, tenant_id) DO UPDATE SET
            value = EXCLUDED.value,
            description = EXCLUDED.description,
            category = EXCLUDED.category,
            scope = EXCLUDED.scope,
            updated_at = NOW()
        """
        async with self.pool.acquire() as conn:
            await conn.execute(query,
                             data.get('name'),
                             data.get('value'),
                             description,
                             category,
                             data.get('scope'),
                             tenant_id)

    async def delete_credential(self, name: str):
        """Delete a credential by name."""
        query = "DELETE FROM credentials WHERE name = $1"
        async with self.pool.acquire() as conn:
            await conn.execute(query, name)

    async def delete_tenant(self, bot_phone_number: str):
        """Delete a tenant by bot_phone_number."""
        query = "UPDATE tenants SET is_active = FALSE WHERE bot_phone_number = $1"
        async with self.pool.acquire() as conn:
            await conn.execute(query, bot_phone_number)

    async def get_all_tools(self) -> List[dict]:
        """Fetch all tools."""
        query = "SELECT * FROM tools WHERE is_active = TRUE ORDER BY created_at DESC"
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query)
            return [dict(row) for row in rows]

    async def create_tool(self, data: dict):
        """Create a new tool."""
        query = "INSERT INTO tools (name, type, config, service_url) VALUES ($1, $2, $3, $4) RETURNING id"
        async with self.pool.acquire() as conn:
            tool_id = await conn.fetchval(query, data['name'], data['type'], json.dumps(data['config']), data.get('service_url'))
            return tool_id

    async def get_tenant_tools(self, tenant_id: int) -> List[dict]:
        """Get tools enabled for a tenant."""
        query = """
        SELECT t.* FROM tools t
        JOIN tenant_tools tt ON t.id = tt.tool_id
        WHERE tt.tenant_id = $1 AND tt.is_enabled = TRUE AND t.is_active = TRUE
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, tenant_id)
            return [dict(row) for row in rows]

    async def enable_tenant_tool(self, tenant_id: int, tool_id: int):
        """Enable a tool for a tenant."""
        query = "INSERT INTO tenant_tools (tenant_id, tool_id) VALUES ($1, $2) ON CONFLICT DO NOTHING"
        async with self.pool.acquire() as conn:
            await conn.execute(query, tenant_id, tool_id)

    async def disable_tenant_tool(self, tenant_id: int, tool_id: int):
        """Disable a tool for a tenant."""
        query = "UPDATE tenant_tools SET is_enabled = FALSE WHERE tenant_id = $1 AND tool_id = $2"
        async with self.pool.acquire() as conn:
            await conn.execute(query, tenant_id, tool_id)

    # Setup Sessions methods
    async def create_setup_session(self, session_id: str, public_base_url: str = None, webhook_base_url: str = None) -> int:
        """Create a new setup session"""
        query = """
        INSERT INTO setup_sessions (session_id, public_base_url, webhook_base_url)
        VALUES ($1, $2, $3)
        RETURNING id
        """
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, session_id, public_base_url, webhook_base_url)

    async def get_setup_session(self, session_id: str) -> Optional[dict]:
        """Get setup session by session_id"""
        query = "SELECT * FROM setup_sessions WHERE session_id = $1"
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, session_id)
            return dict(row) if row else None

    async def update_setup_session(self, session_id: str, current_step: str = None, status: str = None,
                                 infrastructure_checks: dict = None):
        """Update setup session"""
        updates = []
        params = []
        param_count = 0

        if current_step:
            param_count += 1
            updates.append(f"current_step = ${param_count}")
            params.append(current_step)

        if status:
            param_count += 1
            updates.append(f"status = ${param_count}")
            params.append(status)

        if infrastructure_checks:
            param_count += 1
            updates.append(f"infrastructure_checks = ${param_count}")
            params.append(json.dumps(infrastructure_checks))

        if not updates:
            return

        updates.append("updated_at = NOW()")
        if status == 'completed':
            updates.append("completed_at = NOW()")

        query = f"UPDATE setup_sessions SET {', '.join(updates)} WHERE session_id = ${param_count + 1}"
        params.append(session_id)

        async with self.pool.acquire() as conn:
            await conn.execute(query, *params)

    async def insert_setup_event(self, session_id: int, event_type: str, step: str = None,
                               status: str = None, message: str = None, details: dict = None):
        """Insert a setup event"""
        query = """
        INSERT INTO setup_events (session_id, event_type, step, status, message, details)
        VALUES ($1, $2, $3, $4, $5, $6)
        """
        async with self.pool.acquire() as conn:
            await conn.execute(query, session_id, event_type, step, status, message,
                             json.dumps(details) if details else None)

    async def get_setup_events(self, session_id: int, limit: int = 50) -> List[dict]:
        """Get setup events for a session"""
        query = """
        SELECT event_type, step, status, message, details, created_at
        FROM setup_events
        WHERE session_id = $1
        ORDER BY created_at DESC
        LIMIT $2
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, session_id, limit)
            return [dict(row) for row in rows]

# Global instance
db = Database()

