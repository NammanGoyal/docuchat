-- Documents and their chunks. Vectors live next to the text they were computed
-- from; full-text search uses a generated tsvector column so it can never drift
-- out of sync with `content`.

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;

CREATE TABLE documents (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    filename        text        NOT NULL,
    kind            text        NOT NULL CHECK (kind IN ('pdf', 'markdown', 'text')),
    -- sha256 of the uploaded bytes: identical re-uploads are deduplicated.
    content_hash    text        NOT NULL UNIQUE,
    size_bytes      integer     NOT NULL,
    num_pages       integer,
    -- Which embedder produced this document's vectors. Search only compares a
    -- query with chunks embedded by the same model.
    embedding_model text        NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id uuid    NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    chunk_index integer NOT NULL,
    page        integer,
    content     text    NOT NULL,
    embedding   public.vector(512) NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    UNIQUE (document_id, chunk_index)
);

-- Approximate nearest-neighbour index for cosine distance (the <=> operator).
CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding public.vector_cosine_ops);
-- Inverted index for full-text search (the @@ operator).
CREATE INDEX chunks_tsv_gin ON chunks USING gin (tsv);
