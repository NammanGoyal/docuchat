import json
import math

import httpx
import pytest

from app.config import EMBEDDING_DIM, Settings
from app.providers import ProviderError, build_answerer, build_embedder
from app.providers.base import Passage
from app.providers.local import ExtractiveAnswerer, HashingEmbedder, split_sentences
from app.providers.openai_compat import OpenAIChatAnswerer, OpenAIEmbedder, build_messages


def cos(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder(EMBEDDING_DIM)
    v1, v2 = e.embed(["Raft elects a leader", "Raft elects a leader"])
    assert v1 == v2
    assert len(v1) == EMBEDDING_DIM
    assert math.isclose(math.sqrt(sum(x * x for x in v1)), 1.0, rel_tol=1e-9)
    # a fresh instance produces the same vector (no per-process randomness)
    assert HashingEmbedder(EMBEDDING_DIM).embed_one("Raft elects a leader") == v1


def test_hashing_embedder_ranks_related_text_higher():
    e = HashingEmbedder(EMBEDDING_DIM)
    q = e.embed_one("how is a leader elected?")
    related = e.embed_one("Leader election starts when a follower's election timeout expires.")
    unrelated = e.embed_one("Water tomato plants early in the morning.")
    assert cos(q, related) > cos(q, unrelated) + 0.1


def test_hashing_embedder_handles_text_without_terms():
    v = HashingEmbedder(EMBEDDING_DIM).embed_one("the of and !!!")
    assert math.isclose(sum(x * x for x in v), 1.0)


def test_split_sentences_drops_fragments_and_joins_wrapped_lines():
    text = "First real sentence here. Ok.\n\n- a bullet point item\n- second bullet item\nwrapped onto next line"
    assert split_sentences(text) == [
        "First real sentence here.",
        "a bullet point item",
        "second bullet item wrapped onto next line",
    ]


def test_extractive_answer_quotes_best_sentence_with_citation():
    passages = [
        Passage(1, "garden.pdf", 2, "Water early in the morning. Mulch reduces evaporation a lot."),
        Passage(2, "raft.txt", None, "A candidate needs votes from a majority of servers to win."),
    ]
    answer = "".join(ExtractiveAnswerer().stream_answer("How many votes does a candidate need?", passages))
    lines = [line for line in answer.splitlines() if line.startswith(">")]
    assert lines[0] == "> A candidate needs votes from a majority of servers to win. [2]"
    assert "Water early" not in answer  # no shared terms with the question


def test_extractive_answer_falls_back_to_top_passage():
    passages = [Passage(1, "a.md", None, "Completely unrelated content sits here.")]
    picked = ExtractiveAnswerer().select("zebra quantum", passages)
    assert picked == [("Completely unrelated content sits here.", 1)]


def test_factory_defaults_to_offline_providers():
    s = Settings()
    assert build_embedder(s).name.startswith("local-hashing")
    assert build_answerer(s).name == "local-extractive-v1"


def test_settings_reject_overlap_not_smaller_than_size():
    with pytest.raises(ValueError):
        Settings(chunk_size=100, chunk_overlap=100)


# --- OpenAI-compatible providers, against an in-process fake server --------


def test_openai_embedder_sends_dimensions_and_orders_by_index():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        data = [{"index": 1, "embedding": [0.0] * 3}, {"index": 0, "embedding": [1.0] * 3}]
        return httpx.Response(200, json={"data": data})

    e = OpenAIEmbedder("http://fake/v1", "sk-test", "embed-model", 3, transport=httpx.MockTransport(handler))
    vecs = e.embed(["first", "second"])
    assert vecs == [[1.0] * 3, [0.0] * 3]
    assert seen["path"] == "/v1/embeddings"
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"] == {"model": "embed-model", "input": ["first", "second"], "dimensions": 3}


def test_openai_embedder_rejects_wrong_dimension():
    transport = httpx.MockTransport(
        lambda r: httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]})
    )
    with pytest.raises(ProviderError, match="expected 3"):
        OpenAIEmbedder("http://fake/v1", None, "m", 3, transport=transport).embed(["x"])


def test_openai_embedder_wraps_http_errors():
    transport = httpx.MockTransport(lambda r: httpx.Response(401, json={"error": "bad key"}))
    with pytest.raises(ProviderError):
        OpenAIEmbedder("http://fake/v1", "bad", "m", 3, transport=transport).embed(["x"])


def test_openai_chat_parses_streamed_deltas():
    chunks = [
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "Five servers "}}]},
        {"choices": [{"delta": {"content": "tolerate two failures [1]."}}]},
        {"choices": []},
    ]
    sse = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    llm = OpenAIChatAnswerer("http://fake/v1", "k", "chat-model", transport=httpx.MockTransport(handler))
    passages = [Passage(1, "raft.txt", None, "A five-server cluster tolerates two failures.")]
    out = "".join(llm.stream_answer("How many failures?", passages))
    assert out == "Five servers tolerate two failures [1]."
    assert captured["body"]["stream"] is True
    assert captured["body"]["model"] == "chat-model"


def test_prompt_numbers_passages_and_names_sources():
    msgs = build_messages("q?", [Passage(3, "guide.pdf", 2, "body text")])
    assert msgs[0]["role"] == "system"
    assert "[3] (source: guide.pdf, page 2)\nbody text" in msgs[1]["content"]
    assert msgs[1]["content"].endswith("Question: q?")
