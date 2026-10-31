"""Question answering: retrieve -> decide whether to answer -> generate with citations."""

from collections.abc import Iterator
from dataclasses import dataclass, field

from psycopg_pool import ConnectionPool

from app.config import Settings
from app.providers import AnswerGenerator, Embedder, Passage
from app.retrieval import Confidence, Mode, confidence, search, should_refuse
from app.text import content_terms

REFUSAL = (
    "I couldn't find anything in your documents that answers this confidently. "
    "Try rephrasing, or upload a document that covers it."
)


@dataclass(frozen=True)
class Citation:
    marker: int
    document_id: str
    filename: str
    chunk_index: int
    page: int | None
    text: str
    score: float
    similarity: float


@dataclass
class Prepared:
    """Everything known before generation starts (sent first when streaming)."""

    refused: bool
    confidence: Confidence
    citations: list[Citation] = field(default_factory=list)


def prepare(pool: ConnectionPool, settings: Settings, embedder: Embedder, question: str, top_k: int) -> Prepared:
    if not content_terms(question):  # only stop words / punctuation: nothing to search for
        return Prepared(refused=True, confidence=Confidence(0.0, 0.0))
    qvec = embedder.embed([question])[0]
    with pool.connection() as conn:
        results = search(
            conn,
            question,
            qvec,
            embedder.name,
            top_k,
            mode=Mode.HYBRID,
            candidate_pool=settings.candidate_pool,
            rrf_k=settings.rrf_k,
        )
    conf = confidence(question, results)
    if should_refuse(conf, settings.min_similarity, settings.min_term_coverage):
        return Prepared(refused=True, confidence=conf)
    citations = [
        Citation(
            marker=i,
            document_id=str(r.candidate.document_id),
            filename=r.candidate.filename,
            chunk_index=r.candidate.chunk_index,
            page=r.candidate.page,
            text=r.candidate.content,
            score=round(r.score, 6),
            similarity=round(r.candidate.similarity, 4),
        )
        for i, r in enumerate(results, start=1)
    ]
    return Prepared(refused=False, confidence=conf, citations=citations)


def answer_stream(answerer: AnswerGenerator, question: str, prepared: Prepared) -> Iterator[str]:
    if prepared.refused:
        yield REFUSAL
        return
    passages = [Passage(c.marker, c.filename, c.page, c.text) for c in prepared.citations]
    yield from answerer.stream_answer(question, passages)
