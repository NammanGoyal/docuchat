"""HTTP API."""

import json
import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app import repository
from app.config import Settings, get_settings
from app.db import create_pool
from app.ingestion.extract import ExtractionError
from app.ingestion.service import ingest
from app.providers import ProviderError, build_answerer, build_embedder
from app.qa import answer_stream, prepare

log = logging.getLogger("docuchat")


class DocumentOut(BaseModel):
    id: UUID
    filename: str
    kind: str
    content_hash: str
    size_bytes: int
    num_pages: int | None
    num_chunks: int
    embedding_model: str
    created_at: datetime


class UploadOut(DocumentOut):
    duplicate: bool


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    stream: bool = False


class CitationOut(BaseModel):
    marker: int
    document_id: str
    filename: str
    chunk_index: int
    page: int | None
    text: str
    score: float
    similarity: float


class ConfidenceOut(BaseModel):
    similarity: float
    term_coverage: float


class AskOut(BaseModel):
    answer: str
    refused: bool
    confidence: ConfidenceOut
    citations: list[CitationOut]


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pool = create_pool(settings.database_url, settings.db_schema)
        app.state.embedder = build_embedder(settings)
        app.state.answerer = build_answerer(settings)
        log.info("embedder=%s answerer=%s", app.state.embedder.name, app.state.answerer.name)
        try:
            yield
        finally:
            app.state.pool.close()

    app = FastAPI(title="DocuChat", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    @app.exception_handler(ProviderError)
    def _provider_error(_: Request, exc: ProviderError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.get("/healthz")
    def healthz(request: Request) -> dict:
        try:
            with request.app.state.pool.connection(timeout=2) as conn:
                conn.execute("SELECT 1")
        except Exception as exc:  # report, don't crash, so orchestrators see 503
            raise HTTPException(503, f"database unavailable: {exc.__class__.__name__}") from exc
        return {
            "status": "ok",
            "embedder": request.app.state.embedder.name,
            "answerer": request.app.state.answerer.name,
        }

    @app.post("/documents", response_model=UploadOut, status_code=201)
    def upload_document(request: Request, response: Response, file: Annotated[UploadFile, File()]) -> dict:
        data = file.file.read(settings.max_upload_bytes + 1)
        try:
            result = ingest(request.app.state.pool, settings, request.app.state.embedder, file.filename or "", data)
        except ExtractionError as exc:
            raise HTTPException(422, str(exc)) from exc
        if result.duplicate:
            response.status_code = 200
        return {**result.document, "duplicate": result.duplicate}

    @app.get("/documents", response_model=list[DocumentOut])
    def list_documents(request: Request) -> list[dict]:
        with request.app.state.pool.connection() as conn:
            return repository.list_documents(conn)

    @app.delete("/documents/{document_id}", status_code=204)
    def delete_document(request: Request, document_id: UUID) -> Response:
        with request.app.state.pool.connection() as conn:
            if not repository.delete_document(conn, document_id):
                raise HTTPException(404, "document not found")
        return Response(status_code=204)

    @app.post("/ask", response_model=AskOut)
    def ask(request: Request, body: AskIn):
        state = request.app.state
        question = body.question.strip()
        prepared = prepare(state.pool, settings, state.embedder, question, body.top_k)
        citations = [asdict(c) for c in prepared.citations]
        conf = {k: round(v, 4) for k, v in asdict(prepared.confidence).items()}

        if not body.stream:
            answer = "".join(answer_stream(state.answerer, question, prepared))
            return AskOut(answer=answer, refused=prepared.refused, confidence=conf, citations=citations)

        def events() -> Iterator[str]:
            yield _sse("meta", {"refused": prepared.refused, "confidence": conf, "citations": citations})
            try:
                for delta in answer_stream(state.answerer, question, prepared):
                    yield _sse("token", {"text": delta})
            except ProviderError as exc:
                log.warning("generation failed: %s", exc)
                yield _sse("error", {"detail": str(exc)})
                return
            yield _sse("done", {})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
