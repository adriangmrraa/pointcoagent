-- Schema for Platform UI Support (Tenants & Credentials)

-- 1. Tenants Table (Multi-tenancy support / Store Configuration)
CREATE TABLE IF NOT EXISTS tenants (
    id SERIAL PRIMARY KEY,
    store_name TEXT NOT NULL,
    bot_phone_number TEXT UNIQUE NOT NULL, -- Acts as the main identifier for the bot
    owner_email TEXT,
    store_location TEXT,
    store_website TEXT,
    store_description TEXT, -- "Context" for the AI
    store_catalog_knowledge TEXT, -- "Knowledge" for the AI
    
    -- Tienda Nube Specifics
    tiendanube_store_id TEXT,
    tiendanube_access_token TEXT,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Credentials Table (Secrets Management)
CREATE TABLE IF NOT EXISTS credentials (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,          -- e.g., "OPENAI_API_KEY"
    value TEXT NOT NULL,         -- Encrypted or plain (depending on security reqs, plain for now per user context)
    category TEXT,               -- "OpenAI", "WhatsApp", "YCloud", etc.
    scope TEXT DEFAULT 'global', -- 'global' or 'tenant'
    tenant_id INTEGER REFERENCES tenants(id) ON DELETE CASCADE,
    description TEXT,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. System Events (For "Console" view)
CREATE TABLE IF NOT EXISTS system_events (
    id BIGSERIAL PRIMARY KEY,
    event_type TEXT NOT NULL,    -- "error", "info", "warning", "tool_call"
    severity TEXT DEFAULT 'info',
    message TEXT,
    payload JSONB,
    occurred_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Insert Default Tenant (Pointe Coach)
INSERT INTO tenants (
    store_name, 
    bot_phone_number, 
    store_location, 
    store_website,
    store_description,
    store_catalog_knowledge
) VALUES (
    'Pointe Coach',
    '5491100000000', -- Placeholder, user can update via UI
    'Paraná, Entre Ríos, Argentina',
    'https://www.pointecoach.shop/',
    'tienda de artículos de danza clásica y contemporánea',
    '- Accesorios: Metatarsianas, Bolsa de red, Elásticos, Cintas de satén y elastizadas, Endurecedor para puntas, Accesorios para el pie, Punteras, Protectores de puntas.
- Medias: Medias convertibles, Socks, Medias de contemporáneo, Medias poliamida, Medias de patín.
- Zapatillas: Zapatillas de punta, Zapatillas de media punta.
- Marcas: Pointe Coach, Grishko, Capezio, Sansha.'
) ON CONFLICT (bot_phone_number) DO NOTHING;
