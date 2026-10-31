"""Retrieval evaluation: vector-only vs full-text-only vs hybrid (RRF).

Ingests sample_docs/ into a throwaway Postgres schema, runs every question in
eval/dataset.json through each retrieval mode and prints hit@k and MRR, plus
how well the refusal threshold separates answerable from unanswerable questions.

Run from backend/ with Postgres up (docker compose up -d db):
    python -m eval.run_eval
"""

import argparse
import json
import uuid
from pathlib import Path

from app.config import Settings
from app.db import connect, create_pool, migrate
from app.ingestion.service import ingest
from app.providers import build_embedder
from app.retrieval import Confidence, Mode, confidence, search, should_refuse

HERE = Path(__file__).resolve().parent
DOCS = HERE.parents[1] / "sample_docs"
KS = (1, 3, 5)
MRR_DEPTH = 10


def _norm(s: str) -> str:
    return " ".join(s.split())


def first_relevant_rank(results, document: str, evidence: str) -> int | None:
    ev = _norm(evidence)
    for rank, r in enumerate(results, start=1):
        if r.candidate.filename == document and ev in _norm(r.candidate.content):
            return rank
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="print machine-readable results")
    args = ap.parse_args()

    base = Settings()
    schema = f"eval_{uuid.uuid4().hex[:8]}"
    settings = base.model_copy(update={"db_schema": schema})
    embedder = build_embedder(settings)
    dataset = json.loads((HERE / "dataset.json").read_text())["questions"]
    answerable = [q for q in dataset if q["document"]]
    unanswerable = [q for q in dataset if not q["document"]]

    with connect(settings.database_url, schema) as admin:
        migrate(admin, schema)
    pool = create_pool(settings.database_url, schema, max_size=2)
    try:
        n_chunks = 0
        for path in sorted(DOCS.iterdir()):
            if path.suffix.lower() in (".md", ".txt", ".pdf"):
                n_chunks += ingest(pool, settings, embedder, path.name, path.read_bytes()).document["num_chunks"]

        report: dict = {
            "embedder": embedder.name,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "chunks": n_chunks,
            "answerable": len(answerable),
            "unanswerable": len(unanswerable),
            "modes": {},
        }
        with pool.connection() as conn:
            for mode in Mode:
                ranks = []
                for q in answerable:
                    qvec = embedder.embed([q["question"]])[0]
                    res = search(
                        conn,
                        q["question"],
                        qvec,
                        embedder.name,
                        MRR_DEPTH,
                        mode=mode,
                        candidate_pool=settings.candidate_pool,
                        rrf_k=settings.rrf_k,
                    )
                    ranks.append(first_relevant_rank(res, q["document"], q["evidence"]))
                n = len(ranks)
                report["modes"][mode.value] = {
                    **{f"hit@{k}": sum(1 for r in ranks if r and r <= k) / n for k in KS},
                    f"mrr@{MRR_DEPTH}": sum(1 / r for r in ranks if r) / n,
                }

            # Refusal signals for the same top-5 hybrid results the API uses by default.
            def conf(q: dict) -> Confidence:
                qvec = embedder.embed([q["question"]])[0]
                res = search(
                    conn,
                    q["question"],
                    qvec,
                    embedder.name,
                    5,
                    candidate_pool=settings.candidate_pool,
                    rrf_k=settings.rrf_k,
                )
                return confidence(q["question"], res)

            pos = [conf(q) for q in answerable]
            neg = [conf(q) for q in unanswerable]

            def rule(name: str, min_sim: float, min_cov: float) -> dict:
                return {
                    "rule": name,
                    "min_similarity": min_sim,
                    "min_term_coverage": min_cov,
                    "answered_answerable": sum(not should_refuse(c, min_sim, min_cov) for c in pos) / len(pos),
                    "refused_unanswerable": sum(should_refuse(c, min_sim, min_cov) for c in neg) / len(neg),
                }

            never = 2.0  # a threshold no signal can reach, i.e. "ignore this signal"
            s, c = settings.min_similarity, settings.min_term_coverage
            report["refusal"] = [
                rule("similarity only", s, never),
                rule("term coverage only", never, c),
                rule("similarity OR coverage (configured)", s, c),
                *(rule("similarity OR coverage", t / 100, c) for t in range(10, 27, 4)),
            ]
    finally:
        pool.close()
        with connect(settings.database_url) as admin:
            admin.execute(f'DROP SCHEMA "{schema}" CASCADE')
            admin.commit()

    if args.json:
        print(json.dumps(report, indent=2))
        return
    print(
        f"embedder={report['embedder']} chunks={n_chunks} (size {settings.chunk_size}, "
        f"overlap {settings.chunk_overlap}); "
        f"{len(answerable)} answerable + {len(unanswerable)} unanswerable questions\n"
    )
    cols = [f"hit@{k}" for k in KS] + [f"mrr@{MRR_DEPTH}"]
    print("| mode | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for mode, m in report["modes"].items():
        print(f"| {mode} | " + " | ".join(f"{m[c]:.3f}" for c in cols) + " |")
    print(f"\nrefusal ({len(answerable)} answerable / {len(unanswerable)} unanswerable questions, hybrid top-5):\n")
    print("| rule | min_similarity | min_term_coverage | answered answerable | refused unanswerable |")
    print("|---|---|---|---|---|")
    for r in report["refusal"]:
        sim = "-" if r["min_similarity"] == never else r["min_similarity"]
        cov = "-" if r["min_term_coverage"] == never else r["min_term_coverage"]
        print(f"| {r['rule']} | {sim} | {cov} | {r['answered_answerable']:.1%} | {r['refused_unanswerable']:.1%} |")


if __name__ == "__main__":
    main()
