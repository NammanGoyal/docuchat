"""End-to-end API tests against a real Postgres + pgvector (see conftest.py)."""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.db import connect, create_pool
from app.ingestion.service import ingest
from app.providers import build_embedder
from app.qa import REFUSAL

SAMPLES = Path(__file__).resolve().parents[2] / "sample_docs"


def upload(client, name: str, data: bytes | None = None):
    data = (SAMPLES / name).read_bytes() if data is None else data
    return client.post("/documents", files={"file": (name, data)})


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def test_healthz_reports_database_and_providers(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "embedder": "local-hashing-v1-512", "answerer": "local-extractive-v1"}


def test_upload_list_and_dedupe(client):
    r = upload(client, "raft-consensus.txt")
    assert r.status_code == 201
    doc = r.json()
    assert doc["kind"] == "text" and doc["duplicate"] is False and doc["num_chunks"] > 1
    assert re.fullmatch(r"[0-9a-f]{64}", doc["content_hash"])

    # Same bytes under another name: not re-ingested, the original is returned.
    data = (SAMPLES / "raft-consensus.txt").read_bytes()
    again = client.post("/documents", files={"file": ("copy-of-raft.txt", data)})
    assert again.status_code == 200
    assert again.json()["duplicate"] is True and again.json()["id"] == doc["id"]

    listed = client.get("/documents").json()
    assert [d["id"] for d in listed] == [doc["id"]]
    assert listed[0]["num_chunks"] == doc["num_chunks"]


def test_ask_returns_cited_answer(client):
    upload(client, "raft-consensus.txt")
    upload(client, "aurora-handbook.md")
    r = client.post("/ask", json={"question": "How does Raft avoid split votes in elections?", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["refused"] is False
    assert 1 <= len(body["citations"]) <= 3
    top = body["citations"][0]
    assert top["filename"] == "raft-consensus.txt" and top["marker"] == 1
    assert "election timeout randomly" in " ".join(top["text"].split())
    # every [n] in the answer points at a returned citation
    markers = {int(m) for m in re.findall(r"\[(\d+)\]", body["answer"])}
    assert markers and markers <= {c["marker"] for c in body["citations"]}


def test_ask_refuses_when_documents_do_not_cover_question(client):
    upload(client, "raft-consensus.txt")
    body = client.post("/ask", json={"question": "What is the capital of Australia?"}).json()
    assert body["refused"] is True
    assert body["citations"] == []
    assert body["answer"] == REFUSAL


def test_ask_with_no_documents_refuses(client):
    body = client.post("/ask", json={"question": "How are leaders elected?"}).json()
    assert body["refused"] is True
    assert body["confidence"] == {"similarity": 0.0, "term_coverage": 0.0}


def test_streaming_matches_non_streaming_answer(client):
    upload(client, "aurora-handbook.md")
    q = {"question": "How many days of paid annual leave do employees get?", "top_k": 4}
    plain = client.post("/ask", json=q).json()

    r = client.post("/ask", json={**q, "stream": True})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    assert events[0][0] == "meta" and events[-1][0] == "done"
    meta = events[0][1]
    assert meta["refused"] is False and meta["citations"] == plain["citations"]
    tokens = [data["text"] for name, data in events if name == "token"]
    assert len(tokens) > 1
    assert "".join(tokens) == plain["answer"]
    assert "24 days of paid annual leave" in plain["answer"]


def test_pdf_citations_carry_page_numbers(client):
    r = upload(client, "balcony-garden-guide.pdf")
    assert r.status_code == 201 and r.json()["kind"] == "pdf" and r.json()["num_pages"] == 3
    body = client.post("/ask", json={"question": "How should I treat aphids with neem oil?"}).json()
    assert body["refused"] is False
    top = body["citations"][0]
    assert top["filename"] == "balcony-garden-guide.pdf" and top["page"] == 3


def test_delete_removes_document_and_its_chunks(client, settings):
    doc = upload(client, "raft-consensus.txt").json()
    assert client.delete(f"/documents/{doc['id']}").status_code == 204
    assert client.get("/documents").json() == []
    with connect(settings.database_url, settings.db_schema) as conn:
        assert conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"] == 0
    assert client.post("/ask", json={"question": "How are Raft leaders elected?"}).json()["refused"] is True
    assert client.delete(f"/documents/{doc['id']}").status_code == 404


def test_rejects_bad_uploads(client):
    assert upload(client, "slides.pptx", b"not supported").status_code == 422
    assert upload(client, "empty.txt", b"").status_code == 422
    assert upload(client, "broken.pdf", b"%PDF-1.4 garbage").status_code == 422
    assert client.get("/documents").json() == []


def test_rejects_invalid_ask_requests(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "hi", "top_k": 0}).status_code == 422
    assert client.post("/ask", json={"question": "hi", "top_k": 21}).status_code == 422
    assert client.delete("/documents/not-a-uuid").status_code == 422


def test_chunks_from_a_different_embedding_model_are_not_searched(client, settings):
    upload(client, "raft-consensus.txt")
    q = {"question": "How does Raft avoid split votes in elections?"}
    assert client.post("/ask", json=q).json()["refused"] is False
    with connect(settings.database_url, settings.db_schema) as conn:
        conn.execute("UPDATE documents SET embedding_model = 'some-other-model'")
        conn.commit()
    assert client.post("/ask", json=q).json()["refused"] is True


def test_concurrent_identical_uploads_create_one_document(settings):
    data = (SAMPLES / "aurora-handbook.md").read_bytes()
    embedder = build_embedder(settings)
    pool = create_pool(settings.database_url, settings.db_schema, max_size=8)
    try:
        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(lambda i: ingest(pool, settings, embedder, f"h{i}.md", data), range(8)))
        assert sum(not r.duplicate for r in results) == 1
        assert len({str(r.document["id"]) for r in results}) == 1
        with pool.connection() as conn:
            assert conn.execute("SELECT count(*) AS n FROM documents").fetchone()["n"] == 1
    finally:
        pool.close()
