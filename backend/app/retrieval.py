"""Hybrid retrieval: vector search + full-text search, fused with Reciprocal Rank Fusion."""

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import psycopg

from app import repository
from app.repository import Candidate
from app.text import content_terms


class Mode(StrEnum):
    VECTOR = "vector"
    FULLTEXT = "fulltext"
    HYBRID = "hybrid"


def rrf_fuse[K: Hashable](rankings: Sequence[Sequence[K]], k: int = 60) -> list[tuple[K, float]]:
    """Reciprocal Rank Fusion (Cormack et al., 2009).

    score(item) = sum over rankings of 1 / (k + rank), rank starting at 1.
    Only ranks are used, so the two retrievers' raw scores (cosine similarity
    vs ts_rank) never need to be put on a common scale. `k` damps the
    advantage of the very top ranks. Ties keep first-seen order.
    """
    scores: dict[K, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


@dataclass(frozen=True)
class Retrieved:
    candidate: Candidate
    score: float  # RRF score in hybrid mode; raw rank-based score otherwise
    vector_rank: int | None
    text_rank: int | None


def search(
    conn: psycopg.Connection,
    question: str,
    qvec: list[float],
    model: str,
    top_k: int,
    *,
    mode: Mode = Mode.HYBRID,
    candidate_pool: int = 30,
    rrf_k: int = 60,
) -> list[Retrieved]:
    pool = max(candidate_pool, top_k)
    vec = repository.vector_search(conn, qvec, model, pool) if mode != Mode.FULLTEXT else []
    txt = repository.text_search(conn, question, qvec, model, pool) if mode != Mode.VECTOR else []

    by_id = {c.chunk_id: c for c in [*vec, *txt]}
    vrank = {c.chunk_id: i for i, c in enumerate(vec, start=1)}
    trank = {c.chunk_id: i for i, c in enumerate(txt, start=1)}
    rankings = [r for r in ([c.chunk_id for c in vec], [c.chunk_id for c in txt]) if r]
    fused = rrf_fuse(rankings, k=rrf_k)[:top_k]
    return [Retrieved(by_id[cid], score, vrank.get(cid), trank.get(cid)) for cid, score in fused]


@dataclass(frozen=True)
class Confidence:
    """Two independent signals about whether the results can answer the question.

    RRF scores only encode rank, so they can't tell "best of a bad bunch" from a
    genuine match; these absolute measures can.
    """

    similarity: float  # best query-chunk cosine similarity (semantic/lexical, from the embedder)
    term_coverage: float  # best fraction of the question's content terms found in one chunk


def confidence(question: str, results: Sequence[Retrieved]) -> Confidence:
    if not results:
        return Confidence(0.0, 0.0)
    q_terms = set(content_terms(question))
    coverage = (
        max(len(q_terms & set(content_terms(r.candidate.content))) for r in results) / len(q_terms) if q_terms else 0.0
    )
    return Confidence(max(r.candidate.similarity for r in results), coverage)


def should_refuse(conf: Confidence, min_similarity: float, min_term_coverage: float) -> bool:
    """Answer if EITHER signal is strong enough; refuse only when both are weak."""
    return conf.similarity < min_similarity and conf.term_coverage < min_term_coverage
