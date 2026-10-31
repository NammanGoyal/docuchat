"""All SQL lives here. Functions take an open connection so callers control transactions."""

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import psycopg

from app.ingestion.chunking import Chunk


def vector_literal(vec: list[float]) -> str:
    """pgvector's text input format: '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{v:.7g}" for v in vec) + "]"


_DOC_COLUMNS = "id, filename, kind, content_hash, size_bytes, num_pages, embedding_model, created_at"


def find_document_by_hash(conn: psycopg.Connection, content_hash: str) -> dict[str, Any] | None:
    return conn.execute(
        f"""SELECT {_DOC_COLUMNS}, (SELECT count(*) FROM chunks c WHERE c.document_id = d.id) AS num_chunks
            FROM documents d WHERE content_hash = %s""",
        (content_hash,),
    ).fetchone()


def insert_document(
    conn: psycopg.Connection,
    *,
    filename: str,
    kind: str,
    content_hash: str,
    size_bytes: int,
    num_pages: int | None,
    embedding_model: str,
) -> dict[str, Any] | None:
    """Insert a document row; returns None if the same content already exists
    (a concurrent upload won the race on the UNIQUE(content_hash) constraint)."""
    return conn.execute(
        f"""INSERT INTO documents (filename, kind, content_hash, size_bytes, num_pages, embedding_model)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (content_hash) DO NOTHING
            RETURNING {_DOC_COLUMNS}""",
        (filename, kind, content_hash, size_bytes, num_pages, embedding_model),
    ).fetchone()


def insert_chunks(conn: psycopg.Connection, document_id: UUID, chunks: list[Chunk], vectors: list[list[float]]) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO chunks (document_id, chunk_index, page, content, embedding)
               VALUES (%s, %s, %s, %s, %s::vector)""",
            [(document_id, c.index, c.page, c.text, vector_literal(v)) for c, v in zip(chunks, vectors, strict=True)],
        )


def list_documents(conn: psycopg.Connection) -> list[dict[str, Any]]:
    return conn.execute(
        f"""SELECT {_DOC_COLUMNS}, (SELECT count(*) FROM chunks c WHERE c.document_id = d.id) AS num_chunks
            FROM documents d
            ORDER BY created_at DESC, id"""
    ).fetchall()


def delete_document(conn: psycopg.Connection, document_id: UUID) -> bool:
    cur = conn.execute("DELETE FROM documents WHERE id = %s", (document_id,))
    return cur.rowcount > 0


@dataclass(frozen=True)
class Candidate:
    chunk_id: int
    document_id: UUID
    filename: str
    chunk_index: int
    page: int | None
    content: str
    similarity: float  # cosine similarity between query and chunk vectors, in [-1, 1]


_CANDIDATE_SELECT = """
    SELECT c.id AS chunk_id, c.document_id, d.filename, c.chunk_index, c.page, c.content,
           1 - (c.embedding <=> %(qvec)s::vector) AS similarity
    FROM chunks c JOIN documents d ON d.id = c.document_id
"""


def _candidates(rows: list[dict[str, Any]]) -> list[Candidate]:
    return [Candidate(**r) for r in rows]


def vector_search(conn: psycopg.Connection, qvec: list[float], model: str, limit: int) -> list[Candidate]:
    """Nearest chunks by cosine distance (uses the HNSW index)."""
    rows = conn.execute(
        _CANDIDATE_SELECT
        + """WHERE d.embedding_model = %(model)s
             ORDER BY c.embedding <=> %(qvec)s::vector
             LIMIT %(limit)s""",
        {"qvec": vector_literal(qvec), "model": model, "limit": limit},
    ).fetchall()
    return _candidates(rows)


def text_search(conn: psycopg.Connection, question: str, qvec: list[float], model: str, limit: int) -> list[Candidate]:
    """Full-text matches ranked by ts_rank_cd (uses the GIN index).

    plainto_tsquery ANDs every word, which makes natural-language questions miss
    almost everything; we rewrite its `&`s into `|` so any stemmed, non-stop-word
    term can match and ranking decides relevance.
    """
    rows = conn.execute(
        """WITH q AS (
               SELECT NULLIF(replace(plainto_tsquery('english', %(question)s)::text, '&', '|'), '')::tsquery
                      AS query
           )"""
        + _CANDIDATE_SELECT
        + """, q
             WHERE d.embedding_model = %(model)s AND c.tsv @@ q.query
             ORDER BY ts_rank_cd(c.tsv, q.query) DESC, c.id
             LIMIT %(limit)s""",
        {"question": question, "qvec": vector_literal(qvec), "model": model, "limit": limit},
    ).fetchall()
    return _candidates(rows)
