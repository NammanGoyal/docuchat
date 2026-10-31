#!/usr/bin/env python3
"""Measure /ask latency against a running stack. Stdlib only.

Uploads sample_docs/ (deduplicated if already present), then sends every
question in backend/eval/dataset.json `--rounds` times, sequentially, with
stream=false, and prints latency percentiles. Numbers depend on the machine
and the corpus size (4 small documents here) -- treat them as a local sanity
check, not a capacity claim.

    python3 scripts/bench_ask.py [--api http://localhost:8000] [--rounds 5]
"""

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path

from e2e_smoke import call, upload

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()
    api = args.api.rstrip("/")

    for path in sorted((ROOT / "sample_docs").iterdir()):
        upload(api, path)
    questions = [q["question"] for q in json.loads((ROOT / "backend/eval/dataset.json").read_text())["questions"]]

    # warm-up (connection pool, Python imports, Postgres caches)
    for q in questions[:5]:
        call("POST", f"{api}/ask", json.dumps({"question": q}).encode(), {"Content-Type": "application/json"})

    latencies = []
    started = time.perf_counter()
    for _ in range(args.rounds):
        for q in questions:
            req = urllib.request.Request(
                f"{api}/ask", data=json.dumps({"question": q, "top_k": 5}).encode(),
                headers={"Content-Type": "application/json"}, method="POST",
            )
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp.read()
                assert resp.status == 200
            latencies.append((time.perf_counter() - t0) * 1000)
    elapsed = time.perf_counter() - started

    q = statistics.quantiles(latencies, n=100)
    print(f"{len(latencies)} sequential POST /ask requests in {elapsed:.1f}s ({len(latencies) / elapsed:.0f} req/s)")
    print(f"latency ms: p50={q[49]:.1f} p95={q[94]:.1f} p99={q[98]:.1f} max={max(latencies):.1f}")


if __name__ == "__main__":
    main()
