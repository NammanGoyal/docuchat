#!/usr/bin/env python3
"""End-to-end smoke test against a running stack (docker compose up).

Stdlib only. Exercises every API endpoint plus the web pages, through the
same HTTP interfaces a user hits, then deletes what it uploaded.

    python3 scripts/e2e_smoke.py [--api http://localhost:8000] [--web http://localhost:3000]
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

SAMPLES = Path(__file__).resolve().parents[1] / "sample_docs"


def call(method: str, url: str, body: bytes | None = None, headers: dict | None = None) -> tuple[int, bytes, dict]:
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as err:
        return err.code, err.read(), dict(err.headers)


def upload(api: str, path: Path) -> tuple[int, dict]:
    boundary = uuid.uuid4().hex
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    status, data, _ = call("POST", f"{api}/documents", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return status, json.loads(data)


def ask(api: str, question: str, stream: bool = False) -> tuple[int, bytes, dict]:
    payload = json.dumps({"question": question, "top_k": 3, "stream": stream}).encode()
    return call("POST", f"{api}/ask", payload, {"Content-Type": "application/json"})


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--web", default="http://localhost:3000")
    args = ap.parse_args()
    api, web = args.api.rstrip("/"), args.web.rstrip("/")

    status, body, _ = call("GET", f"{api}/healthz")
    check(status == 200 and json.loads(body)["status"] == "ok", f"GET /healthz -> {status} {body.decode()}")

    created: list[str] = []
    for path in sorted(SAMPLES.iterdir()):
        t0 = time.perf_counter()
        status, doc = upload(api, path)
        ms = (time.perf_counter() - t0) * 1000
        check(status in (200, 201), f"POST /documents {path.name} -> {status}, {doc.get('num_chunks')} chunks ({ms:.0f} ms)")
        if status == 201:
            created.append(doc["id"])

    status, doc = upload(api, SAMPLES / "raft-consensus.txt")
    check(status == 200 and doc["duplicate"] is True, "re-upload of identical bytes is deduplicated (200, duplicate)")

    status, body, _ = call("GET", f"{api}/documents")
    names = {d["filename"] for d in json.loads(body)}
    check(status == 200 and {p.name for p in SAMPLES.iterdir()} <= names, f"GET /documents lists {len(names)} documents")

    status, body, _ = ask(api, "How does Raft avoid split votes during elections?")
    res = json.loads(body)
    top = res["citations"][0] if res["citations"] else {}
    check(status == 200 and not res["refused"] and top.get("filename") == "raft-consensus.txt",
          f"POST /ask answered with citation [{top.get('marker')}] {top.get('filename')} chunk {top.get('chunk_index')}")
    print("     answer:", res["answer"].replace("\n", " ")[:200])

    status, body, _ = ask(api, "What size pot does a tomato plant need?")
    res = json.loads(body)
    pages = [c["page"] for c in res["citations"] if c["filename"] == "balcony-garden-guide.pdf"]
    check(status == 200 and not res["refused"] and pages and pages[0] is not None, f"PDF answer cites page {pages[:1]}")

    status, body, headers = ask(api, "How many days of paid annual leave do employees get?", stream=True)
    events = [blk for blk in body.decode().split("\n\n") if blk.strip()]
    kinds = [blk.splitlines()[0].removeprefix("event: ") for blk in events]
    check(status == 200 and headers.get("content-type", "").startswith("text/event-stream")
          and kinds[0] == "meta" and kinds[-1] == "done" and kinds.count("token") >= 2,
          f"POST /ask stream=true -> SSE events {kinds[0]}, {kinds.count('token')} x token, {kinds[-1]}")

    status, body, _ = ask(api, "What is the capital of Australia?")
    res = json.loads(body)
    check(status == 200 and res["refused"] and res["citations"] == [], f"off-topic question refused (confidence {res['confidence']})")

    for path in ("/", "/upload", "/documents"):
        status, body, _ = call("GET", f"{web}{path}")
        check(status == 200 and b"DocuChat" in body, f"web GET {path} -> {status}")

    for doc_id in created:
        status, _, _ = call("DELETE", f"{api}/documents/{doc_id}")
        check(status == 204, f"DELETE /documents/{doc_id} -> {status}")
    print("all checks passed")


if __name__ == "__main__":
    main()
