from uuid import uuid4

import pytest

from app.config import Settings
from app.qa import REFUSAL, Prepared, answer_stream, prepare
from app.repository import Candidate
from app.retrieval import Confidence, Retrieved, confidence, rrf_fuse, should_refuse


def test_rrf_scores_match_formula():
    fused = dict(rrf_fuse([["a", "b", "c"], ["b", "c", "d"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63 + 1 / 62)
    assert fused["d"] == pytest.approx(1 / 63)


def test_rrf_rewards_agreement_between_rankers():
    # "b" is never first, but both rankers like it, so it wins overall.
    order = [item for item, _ in rrf_fuse([["a", "b", "c"], ["d", "b", "e"]])]
    assert order[0] == "b"


def test_rrf_single_ranking_preserves_order():
    assert [i for i, _ in rrf_fuse([["x", "y", "z"]])] == ["x", "y", "z"]


def test_rrf_small_k_amplifies_top_rank():
    rankings = [["a", "b"], ["b", "c"], ["a", "c"]]
    # with any k, a (ranks 1,1) beats b (ranks 2,1)
    assert rrf_fuse(rankings, k=1)[0][0] == "a"
    big, small = dict(rrf_fuse(rankings, k=1000)), dict(rrf_fuse(rankings, k=1))
    assert small["a"] / small["c"] > big["a"] / big["c"]


def test_rrf_empty():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


def _retrieved(similarity: float, content: str = "text") -> Retrieved:
    cand = Candidate(1, uuid4(), "doc.md", 0, None, content, similarity)
    return Retrieved(cand, score=0.01, vector_rank=1, text_rank=None)


def test_confidence_of_nothing_is_zero_and_refused():
    conf = confidence("how are leaders elected", [])
    assert conf == Confidence(0.0, 0.0)
    assert should_refuse(conf, min_similarity=0.1, min_term_coverage=0.5)


def test_confidence_takes_best_similarity_and_best_single_chunk_coverage():
    results = [
        _retrieved(0.05, "Leaders send heartbeats."),  # covers "leader"
        _retrieved(0.12, "An election starts after a timeout; the leader wins."),  # "leader", "election"
    ]
    conf = confidence("How is a leader elected after a crash?", results)
    assert conf.similarity == pytest.approx(0.12)
    # content terms of the question: leader, elect, crash -> best chunk has 1 of 3
    # ("elected" stems to "elect", "election" stays "election")
    assert conf.term_coverage == pytest.approx(1 / 3)


def test_refuses_only_when_both_signals_are_weak():
    assert should_refuse(Confidence(0.10, 0.30), min_similarity=0.18, min_term_coverage=0.5)
    assert not should_refuse(Confidence(0.18, 0.00), min_similarity=0.18, min_term_coverage=0.5)
    assert not should_refuse(Confidence(0.00, 0.50), min_similarity=0.18, min_term_coverage=0.5)


def test_stopword_only_question_is_refused_without_touching_the_db():
    prepared = prepare(pool=None, settings=Settings(), embedder=None, question="what is the?", top_k=5)
    assert prepared.refused and prepared.citations == []


def test_refused_answer_streams_refusal_message_only():
    prepared = Prepared(refused=True, confidence=Confidence(0.0, 0.0))
    assert "".join(answer_stream(answerer=None, question="q", prepared=prepared)) == REFUSAL
