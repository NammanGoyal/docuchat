"""Upload pipeline: hash -> dedupe -> extract -> chunk -> embed -> store (one transaction)."""

import hashlib
from dataclasses import dataclass
from typing import Any

from psycopg_pool import ConnectionPool

from app import repository
from app.config import Settings
from app.ingestion.chunking import chunk_pages
from app.ingestion.extract import ExtractionError, detect_kind, extract_pages
from app.providers import Embedder


@dataclass(frozen=True)
class IngestResult:
    document: dict[str, Any]
    duplicate: bool


def ingest(pool: ConnectionPool, settings: Settings, embedder: Embedder, filename: str, data: bytes) -> IngestResult:
    if not data:
        raise ExtractionError("file is empty")
    if len(data) > settings.max_upload_bytes:
        raise ExtractionError(f"file exceeds {settings.max_upload_bytes} bytes")
    kind = detect_kind(filename)
    content_hash = hashlib.sha256(data).hexdigest()

    # Cheap pre-check so re-uploads skip extraction and embedding entirely.
    with pool.connection() as conn:
        existing = repository.find_document_by_hash(conn, content_hash)
    if existing:
        return IngestResult(existing, duplicate=True)

    pages = extract_pages(filename, data)
    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    # Embed outside the DB transaction: it may be a slow network call.
    vectors = embedder.embed([c.text for c in chunks])

    with pool.connection() as conn, conn.transaction():
        doc = repository.insert_document(
            conn,
            filename=filename,
            kind=kind,
            content_hash=content_hash,
            size_bytes=len(data),
            num_pages=pages[-1].number if kind == "pdf" else None,
            embedding_model=embedder.name,
        )
        if doc is None:  # lost a race with an identical concurrent upload
            existing = repository.find_document_by_hash(conn, content_hash)
            assert existing is not None
            return IngestResult(existing, duplicate=True)
        repository.insert_chunks(conn, doc["id"], chunks, vectors)
    return IngestResult({**doc, "num_chunks": len(chunks)}, duplicate=False)
