-- Initial schema for Pointe Coach microservices

-- Table for inbound messages (dedupe/idempotency)
CREATE TABLE inbound_messages (
    id BIGSERIAL PRIMARY KEY,
    provider TEXT NOT NULL,
    provider_message_id TEXT NOT NULL,
    event_id TEXT NULL,
    from_number TEXT NOT NULL,
    payload JSONB NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('received', 'processing', 'done', 'failed')),
    received_at TIMESTAMPTZ DEFAULT NOW(),
    processed_at TIMESTAMPTZ NULL,
    error TEXT NULL,
    correlation_id TEXT NULL,
    UNIQUE (provider, provider_message_id)
);

-- Indexes for inbound_messages
CREATE INDEX idx_inbound_messages_from_number_received_at ON inbound_messages (from_number, received_at DESC);
CREATE INDEX idx_inbound_messages_status ON inbound_messages (status);

-- Table for chat messages (source-of-truth)
CREATE TABLE chat_messages (
    id BIGSERIAL PRIMARY KEY,
    from_number TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    correlation_id TEXT NULL
);

-- Indexes for chat_messages (assuming queries by from_number and recent messages)
CREATE INDEX idx_chat_messages_from_number_created_at ON chat_messages (from_number, created_at DESC);
