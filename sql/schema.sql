CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;

DROP TABLE IF EXISTS messages CASCADE;
DROP TABLE IF EXISTS sessions CASCADE;
DROP TABLE IF EXISTS chunks CASCADE;
DROP TABLE IF EXISTS documents CASCADE;
DROP INDEX IF EXISTS idx_chunks_embedding;
DROP INDEX IF EXISTS idx_chunks_document_id;
DROP INDEX IF EXISTS idx_documents_metadata;
DROP INDEX IF EXISTS idx_chunks_content_trgm;

CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_documents_metadata ON documents USING GIN (metadata);
CREATE INDEX idx_documents_created_at ON documents (created_at DESC);

CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(1536),
    chunk_index INTEGER NOT NULL,
    metadata JSONB DEFAULT '{}',
    token_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 1);
CREATE INDEX idx_chunks_document_id ON chunks (document_id);
CREATE INDEX idx_chunks_chunk_index ON chunks (document_id, chunk_index);
CREATE INDEX idx_chunks_content_trgm ON chunks USING GIN (content gin_trgm_ops);

CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_sessions_user_id ON sessions (user_id);
CREATE INDEX idx_sessions_expires_at ON sessions (expires_at);

CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_messages_session_id ON messages (session_id, created_at);

CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding vector(1536),
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    content TEXT,
    similarity FLOAT,
    metadata JSONB,
    document_title TEXT,
    document_source TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.id AS chunk_id,
        c.document_id,
        c.content,
        1 - (c.embedding <=> query_embedding) AS similarity,
        c.metadata,
        d.title AS document_title,
        d.source AS document_source
    FROM chunks c
    JOIN documents d ON c.document_id = d.id
    WHERE c.embedding IS NOT NULL
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

CREATE OR REPLACE FUNCTION hybrid_search(
    query_embedding vector(1536),
    query_text TEXT,
    match_count INT DEFAULT 10,
    text_weight FLOAT DEFAULT 0.3
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    content TEXT,
    combined_score FLOAT,
    vector_similarity FLOAT,
    text_similarity FLOAT,
    metadata JSONB,
    document_title TEXT,
    document_source TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH vector_results AS (
        SELECT 
            c.id AS chunk_id,
            c.document_id,
            c.content,
            1 - (c.embedding <=> query_embedding) AS vector_sim,
            c.metadata,
            d.title AS doc_title,
            d.source AS doc_source
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE c.embedding IS NOT NULL
    ),
    text_results AS (
        SELECT 
            c.id AS chunk_id,
            c.document_id,
            c.content,
            ts_rank_cd(to_tsvector('english', c.content), plainto_tsquery('english', query_text)) AS text_sim,
            c.metadata,
            d.title AS doc_title,
            d.source AS doc_source
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE to_tsvector('english', c.content) @@ plainto_tsquery('english', query_text)
    )
    SELECT 
        COALESCE(v.chunk_id, t.chunk_id) AS chunk_id,
        COALESCE(v.document_id, t.document_id) AS document_id,
        COALESCE(v.content, t.content) AS content,
        (COALESCE(v.vector_sim, 0) * (1 - text_weight) + COALESCE(t.text_sim, 0) * text_weight) AS combined_score,
        COALESCE(v.vector_sim, 0) AS vector_similarity,
        COALESCE(t.text_sim, 0) AS text_similarity,
        COALESCE(v.metadata, t.metadata) AS metadata,
        COALESCE(v.doc_title, t.doc_title) AS document_title,
        COALESCE(v.doc_source, t.doc_source) AS document_source
    FROM vector_results v
    FULL OUTER JOIN text_results t ON v.chunk_id = t.chunk_id
    ORDER BY combined_score DESC
    LIMIT match_count;
END;
$$;

CREATE OR REPLACE FUNCTION get_document_chunks(doc_id UUID)
RETURNS TABLE (
    chunk_id UUID,
    content TEXT,
    chunk_index INTEGER,
    metadata JSONB
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        id AS chunk_id,
        chunks.content,
        chunks.chunk_index,
        chunks.metadata
    FROM chunks
    WHERE document_id = doc_id
    ORDER BY chunk_index;
END;
$$;

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_sessions_updated_at BEFORE UPDATE ON sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Proposals table for human-in-the-loop approval workflow
CREATE TABLE proposals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    kind TEXT NOT NULL CHECK (kind IN ('character', 'relationship', 'location', 'event', 'mixed')),
    payload JSONB NOT NULL,
    source_doc TEXT,
    suggested_by TEXT,
    confidence NUMERIC(3,2) CHECK (confidence >= 0 AND confidence <= 1),
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected', 'failed')),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMPTZ,
    processed_by TEXT,
    neo4j_tx JSONB,
    errors JSONB,
    rejection_reason TEXT
);

CREATE INDEX idx_proposals_status ON proposals (status);
CREATE INDEX idx_proposals_created_at ON proposals (created_at DESC);
CREATE INDEX idx_proposals_kind ON proposals (kind);
CREATE INDEX idx_proposals_suggested_by ON proposals (suggested_by);

-- Validation results table for consistency checks
CREATE TABLE validation_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    proposal_id UUID REFERENCES proposals(id) ON DELETE CASCADE,
    validator_name TEXT NOT NULL,
    score NUMERIC(3,2) CHECK (score >= 0 AND score <= 1),
    violations JSONB DEFAULT '[]',
    suggestions JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_validation_results_proposal_id ON validation_results (proposal_id);
CREATE INDEX idx_validation_results_validator ON validation_results (validator_name);
CREATE INDEX idx_validation_results_score ON validation_results (score);

CREATE OR REPLACE VIEW document_summaries AS
SELECT 
    d.id,
    d.title,
    d.source,
    d.created_at,
    d.updated_at,
    d.metadata,
    COUNT(c.id) AS chunk_count,
    AVG(c.token_count) AS avg_tokens_per_chunk,
    SUM(c.token_count) AS total_tokens
FROM documents d
LEFT JOIN chunks c ON d.id = c.document_id
GROUP BY d.id, d.title, d.source, d.created_at, d.updated_at, d.metadata;

-- View for proposal summaries with validation scores
CREATE OR REPLACE VIEW proposal_summaries AS
SELECT 
    p.id,
    p.kind,
    p.status,
    p.confidence,
    p.created_at,
    p.processed_at,
    p.suggested_by,
    p.processed_by,
    COUNT(vr.id) AS validation_count,
    AVG(vr.score) AS avg_validation_score,
    MIN(vr.score) AS min_validation_score,
    CASE 
        WHEN MIN(vr.score) < 0.5 THEN 'high_risk'
        WHEN MIN(vr.score) < 0.7 THEN 'medium_risk'
        ELSE 'low_risk'
    END AS risk_level
FROM proposals p
LEFT JOIN validation_results vr ON p.id = vr.proposal_id
GROUP BY p.id, p.kind, p.status, p.confidence, p.created_at, p.processed_at, p.suggested_by, p.processed_by;

--
-- Emotional Memory System (v3 - Production Grade)
--

-- Tabel untuk melacak setiap proses analisis emosi (menjamin idempoten & re-run safety)
CREATE TABLE emotion_analysis_runs (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  method TEXT NOT NULL,
  model_name TEXT,
  model_version TEXT,
  started_at TIMESTAMPTZ DEFAULT now(),
  finished_at TIMESTAMPTZ,
  status TEXT CHECK (status IN ('pending','running','success','failed')),
  params JSONB,
  error TEXT
);

-- Tabel untuk data validasi/label dari manusia (untuk evaluasi kualitas)
CREATE TABLE emotion_labels (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  chunk_id UUID REFERENCES chunks(id) ON DELETE CASCADE,
  character_name TEXT,
  label_emotion TEXT NOT NULL,
  label_intensity REAL,
  annotator TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Tabel baru untuk mendefinisikan scene/adegan dalam cerita
CREATE TABLE scenes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    start_chunk_id UUID REFERENCES chunks(id),
    end_chunk_id UUID REFERENCES chunks(id),
    scene_summary TEXT,
    -- Menyimpan emosi bersama atau atmosfer umum dari sebuah scene
    scene_atmosphere JSONB, -- contoh: {"tension": 0.9, "mystery": 0.7}
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_scenes_document_id ON scenes (document_id);


-- Tabel utama untuk emosi karakter dengan semua penyempurnaan
CREATE TABLE character_emotions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID REFERENCES emotion_analysis_runs(id), -- Provenance
    scene_id UUID REFERENCES scenes(id) ON DELETE CASCADE, -- Konteks scene
    chunk_id UUID REFERENCES chunks(id) ON DELETE CASCADE,
    character_name TEXT NOT NULL,
    
    -- Data Emosi & Konteks
    emotion_space TEXT DEFAULT 'plutchik_8', -- Taksonomi Emosi
    emotion_vector JSONB NOT NULL,
    dominant_emotion TEXT NOT NULL,
    top_emotions TEXT[], -- Untuk emosi campuran (multi-label)
    intensity REAL NOT NULL CHECK (intensity >= 0 AND intensity <= 1),
    intensity_calibrated REAL, -- Untuk normalisasi intensitas
    emotion_category TEXT NOT NULL CHECK (emotion_category IN ('positive', 'negative', 'neutral')),
    
    -- Atribusi & Pemicu
    trigger_event TEXT,
    related_character TEXT, -- Karakter pemicu
    attribution_method TEXT, -- 'dialogue', 'narrative', 'coref'
    source_type TEXT NOT NULL CHECK (source_type IN ('dialogue', 'narrative', 'action')),
    
    -- Granularitas & Provenance
    span_start INT,
    span_end INT,
    sentence_index INT,
    source_text TEXT, -- Sebaiknya diredaksi atau dipisah
    
    -- Kualitas & Metadata
    method TEXT DEFAULT 'keyword',
    model_name TEXT,
    model_version TEXT,
    prompt_hash TEXT,
    confidence_score REAL CHECK (confidence_score >= 0 AND confidence_score <= 1),
    
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Index Unik untuk mencegah duplikasi saat ingest ulang
CREATE UNIQUE INDEX uq_emotion_once ON character_emotions (chunk_id, character_name, method, COALESCE(model_version,'v0'));

-- Indexes untuk performa query
CREATE INDEX idx_emotions_run_id ON character_emotions (run_id);
CREATE INDEX idx_emotions_character_name ON character_emotions (character_name, created_at DESC);
CREATE INDEX idx_emotions_dominant_emotion ON character_emotions (dominant_emotion);
CREATE INDEX idx_emotions_vector_gin ON character_emotions USING gin (emotion_vector);
CREATE INDEX idx_emotions_char_hi ON character_emotions (character_name) WHERE intensity >= 0.4;


-- Tabel baru untuk hubungan sebab-akibat antar emosi
CREATE TABLE emotion_causal_links (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_emotion_id UUID NOT NULL REFERENCES character_emotions(id) ON DELETE CASCADE,
    target_emotion_id UUID NOT NULL REFERENCES character_emotions(id) ON DELETE CASCADE,
    link_description TEXT, -- contoh: "kemarahan memicu penyesalan"
    confidence_score FLOAT
);

CREATE INDEX idx_emotion_causal_links_source ON emotion_causal_links (source_emotion_id);
CREATE INDEX idx_emotion_causal_links_target ON emotion_causal_links (target_emotion_id);


-- Tabel alur emosi (melacak evolusi emosi dari waktu ke waktu)
CREATE TABLE emotional_arcs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    arc_name TEXT NOT NULL,
    character_name TEXT,
    plot_thread TEXT,
    start_chunk_id UUID REFERENCES chunks(id),
    end_chunk_id UUID REFERENCES chunks(id),
    arc_summary TEXT,
    start_emotion_vector JSONB,
    end_emotion_vector JSONB,
    peak_emotion TEXT,
    peak_intensity FLOAT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_emotional_arcs_character_name_v2 ON emotional_arcs (character_name);
CREATE INDEX idx_emotional_arcs_plot_thread_v2 ON emotional_arcs (plot_thread);
CREATE INDEX idx_emotional_arcs_is_active_v2 ON emotional_arcs (is_active);

CREATE TRIGGER update_emotional_arcs_updated_at_v2 BEFORE UPDATE ON emotional_arcs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Materialized View untuk query cepat ke state emosi terakhir
-- CATATAN: View ini perlu di-refresh secara berkala (misalnya dengan TRIGGER atau CRON job).
-- CONTOH: REFRESH MATERIALIZED VIEW CONCURRENTLY character_latest_emotion;
CREATE MATERIALIZED VIEW character_latest_emotion AS
SELECT DISTINCT ON (ce.character_name, c.document_id)
  ce.character_name,
  c.document_id,
  ce.chunk_id,
  ce.dominant_emotion,
  ce.intensity,
  ce.created_at
FROM character_emotions ce
JOIN chunks c ON ce.chunk_id = c.id
ORDER BY ce.character_name, c.document_id, ce.created_at DESC;

CREATE INDEX idx_latest_emotion_character ON character_latest_emotion (character_name);
CREATE INDEX idx_latest_emotion_char_doc ON character_latest_emotion (character_name, document_id);
