# LEARNING.md — how DocuChat works, and how to talk about it

This is a study guide for the codebase. Read it top to bottom once with the code open; every pointer is
`file:line` in this repository.

## 1. The problem in one paragraph

An LLM only knows what it was trained on. To answer questions about *your* files, you first **retrieve** the few
passages most likely to contain the answer and then **generate** an answer from only those passages, citing them.
That is RAG (retrieval-augmented generation). The hard parts are (a) finding the right passages and (b) knowing when
none of them actually answers the question. DocuChat's two interesting ideas are **hybrid retrieval** (two different
search methods fused with Reciprocal Rank Fusion) and an explicit **refusal rule**.

## 2. The life of an upload

`POST /documents` → `backend/app/main.py:116` → `ingest()` in `backend/app/ingestion/service.py:22`.

1. **Validate + hash** (`service.py:23-28`). Empty or > 10 MB files are rejected. The file extension decides the
   type (`extract.py:24`). We compute `sha256(bytes)`.
2. **Dedupe early** (`service.py:31-34`). If a document with that hash exists, return it with `duplicate: true` —
   no extraction, no embedding work. The HTTP status is 200 instead of 201 (`main.py:123-124`).
3. **Extract pages** (`extract.py:40`). PDFs go through `pypdf.PdfReader` page by page (`extract.py:45`), so we know
   page numbers; Markdown/TXT become one page with `number=None`. Whitespace is normalised, blank pages dropped. A
   PDF with no text layer (a scan) raises `ExtractionError` → HTTP 422.
4. **Chunk** (`chunking.py:38` `split_text`, `chunking.py:68` `chunk_pages`). Within a page, a chunk is the longest
   run of whole words that fits in `CHUNK_SIZE` characters (default 800, the loop at `chunking.py:51`). The next
   chunk starts at the first word that begins inside the last `CHUNK_OVERLAP` characters (default 150) of the
   previous one. Overlap means a sentence cut in half at a boundary still appears whole in one of the two chunks.
   Chunks never cross pages, so each chunk has exactly one page to cite.
5. **Embed** (`service.py:39`), *outside* the DB transaction, because with a real provider it's a slow network call
   and we don't want to hold a transaction open during it.
6. **Store atomically** (`service.py:41-55`): one transaction inserts the document row and all chunk rows. The insert
   uses `ON CONFLICT (content_hash) DO NOTHING RETURNING ...` (`repository.py:28-46`): if two identical uploads
   race past step 2, the database's `UNIQUE` constraint lets exactly one win and the loser returns the winner's row
   (`service.py:51-53`). `tests/test_api.py::test_concurrent_identical_uploads_create_one_document` fires 8 parallel
   uploads to prove there is only ever one document.

### The schema (`backend/migrations/001_init.sql`)

- `documents`: id (uuid), filename, kind, `content_hash UNIQUE` (line 12), size, pages, `embedding_model`, created_at.
- `chunks`: `document_id ... ON DELETE CASCADE` (line 23), chunk_index, page, content, `embedding vector(512)`, and
  `tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED` (line 28) — Postgres computes the
  full-text representation itself, so it can never drift out of sync with `content`.
- Indexes: **HNSW** on the embedding for approximate nearest-neighbour search by cosine distance (line 33), **GIN**
  on `tsv` for full-text search (line 35).

Migrations are plain SQL files applied in order by `migrate()` (`db.py:36`), which records applied versions in
`schema_migrations` and takes a Postgres advisory lock (`db.py:40`) so two API containers starting at once can't
both run the same migration. The API container runs `python -m app.db` before starting uvicorn
(`backend/Dockerfile`, last line).

## 3. The life of a question

`POST /ask` → `main.py:140` → `prepare()` in `backend/app/qa.py:40`.

1. **Nothing to search?** If the question is only stop words ("what is the?"), refuse immediately (`qa.py:41-42`).
2. **Embed the question** with the same embedder as the documents (`qa.py:43`).
3. **Hybrid search** — `search()` in `backend/app/retrieval.py:43`:
   - **Vector search** (`repository.py:93`): `ORDER BY embedding <=> query_vector LIMIT 30`. `<=>` is pgvector's
     cosine distance; `1 - distance` is cosine similarity. The HNSW index makes this fast without scanning every row.
   - **Full-text search** (`repository.py:105`): turns the question into a `tsquery`, matches with `@@` and ranks
     with `ts_rank_cd`. Important detail: `plainto_tsquery` joins words with AND, so "how does raft avoid split
     votes" would require *every* word in one chunk and usually match nothing. We rewrite `&` to `|` (OR) and let
     ranking sort it out (`repository.py:114`).
   - Both queries filter `d.embedding_model = <current embedder>` so we never compare vectors from different models.
   - **Fuse with RRF** (`retrieval.py:20`, applied at `retrieval.py:61-62`).
4. **Decide whether to answer** (`qa.py:55-57`, rule in `retrieval.py:78-90`). Two signals:
   - `similarity`: best cosine similarity among the top-k chunks.
   - `term_coverage`: the best fraction of the question's content words found in one chunk.
   Refuse only if **both** are below their thresholds (`MIN_SIMILARITY=0.18`, `MIN_TERM_COVERAGE=0.5`).
5. **Build citations** (`qa.py:58-71`): marker `[n]`, filename, chunk index, page, chunk text, RRF score, similarity.
6. **Generate** (`qa.py:74`): the configured `AnswerGenerator` streams text that cites `[n]` markers.
7. **Respond** (`main.py:147-166`): JSON, or Server-Sent Events: one `meta` event (citations + confidence) first,
   then `token` events, then `done` (or `error` if the LLM call fails mid-stream).

### Reciprocal Rank Fusion, concretely

`score(chunk) = Σ over retrievers of 1 / (k + rank)`, with `k = 60` and rank starting at 1.
A chunk ranked 2nd by vectors and 1st by full-text scores `1/62 + 1/61 ≈ 0.0325`; a chunk ranked 1st by vectors but
absent from full-text scores `1/61 ≈ 0.0164`. Agreement between retrievers wins. RRF uses only ranks, so we never
have to decide how a cosine similarity of 0.3 compares with a `ts_rank_cd` of 0.1. `k` damps the advantage of
being exactly first. Tests: `tests/test_retrieval.py` (formula, agreement, effect of k).

## 4. Providers (the pluggable parts)

`backend/app/providers/base.py` defines two small `Protocol`s: `Embedder.embed(texts) -> vectors` and
`AnswerGenerator.stream_answer(question, passages) -> iterator of text`. `providers/__init__.py:16,24` pick an
implementation from env vars. Nothing else in the app knows which one is active.

**Offline embedder — `HashingEmbedder`** (`providers/local.py:52`). The "hashing trick": instead of a vocabulary,
each feature string is hashed (`blake2b`, `local.py:30`) into one of 512 buckets with a +1/−1 sign. Features
(`local.py:35`) are stemmed words, adjacent word pairs, and character trigrams (for "index"/"indexes"-style
variation), weighted `1 + log(tf)` and scaled by feature family (`local.py:49`). The vector is L2-normalised so the
dot product equals cosine similarity. Deterministic and free, but **lexical, not semantic**.

**Offline answerer — `ExtractiveAnswerer`** (`local.py:82`). Splits the retrieved passages into sentences, scores
each by how many question terms it contains (prefers higher-ranked passages on ties, `local.py:98`), and quotes the
top 3 with their `[n]` marker. It cannot hallucinate: every sentence is copied from a cited chunk.

**OpenAI-compatible providers** (`providers/openai_compat.py`). Plain `httpx`, so the wire protocol is visible:
`POST /embeddings` with `dimensions: 512` (`openai_compat.py:46`), and `POST /chat/completions` with
`stream: true`, parsing `data: {...}` lines until `data: [DONE]` (`openai_compat.py:85-104`). The system prompt
(`openai_compat.py:11`) tells the model to answer only from numbered passages and cite them. Tests run these
against `httpx.MockTransport`, an in-process fake server (`tests/test_providers.py`).

## 5. The frontend (`web/`)

- `lib/api.ts` — typed client. `askStream()` (`api.ts:83`) POSTs to `/ask` and reads the response body as a stream
  (`api.ts:97`). We can't use the browser's `EventSource` because it only does GET.
- `lib/sse.ts:9` — `SseParser` buffers partial network chunks and emits complete `event:/data:` blocks. Tested in
  `lib/sse.test.ts`, including an event split across four chunks.
- `app/page.tsx` — chat. Each turn stores the question, the streamed answer (tokens appended in the `onToken`
  callback, `page.tsx:43-44`), and `meta` (citations). `components/Answer.tsx` turns `[n]` into buttons; clicking
  one (`page.tsx:55`) highlights and expands the matching source card. Refused answers get an amber style.
- `app/upload/page.tsx` — drag-and-drop or picker, uploads sequentially, shows added / duplicate / error per file.
- `app/documents/page.tsx` — table with pages, chunks, size; delete with confirm.
- The browser calls the API directly, so the API enables CORS for the web origin (`main.py`, `CORSMiddleware`).
  `NEXT_PUBLIC_API_URL` is inlined at build time, which is why the compose file passes it as a build arg.

## 6. Testing strategy

- **Unit tests** (no DB): chunking invariants (size limit, overlap, word coverage, page boundaries), RRF maths,
  refusal rule, extraction (PDFs are generated by the tiny writer in `tests/pdfgen.py`), embedder determinism,
  extractive answer selection, OpenAI wire format against a fake server.
- **Integration tests** (`tests/test_api.py`, real Postgres + pgvector): upload/list/dedupe/delete, cited answers,
  refusals, SSE stream equals the JSON answer, PDF page citations, bad input → 422, model isolation, concurrent
  duplicate uploads. Each test gets a fresh Postgres **schema** (`tests/conftest.py`): the connection's
  `search_path` points at it (`db.py:22`), so tests can't see each other's or the app's data.
- **End-to-end** (`scripts/e2e_smoke.py`): talks to the real Docker stack over HTTP, including the web pages.
- **Evaluation** (`backend/eval/run_eval.py`): measures retrieval quality, not correctness. A hit is a retrieved
  chunk from the right file that contains the evidence string (`run_eval.py:32`), so the dataset keeps working if
  you change the chunk size.

## 7. Trade-offs you should be able to defend

| Choice | Why | What it costs |
|---|---|---|
| pgvector instead of a vector DB (Pinecone, Qdrant) | one store, transactions, joins, full-text in the same query engine | fewer ANN tuning knobs; scaling is Postgres scaling |
| HNSW instead of IVFFlat | can be built on an empty table, better recall/speed | slower inserts, more memory |
| RRF instead of weighted score blending | no score normalisation, robust, one parameter | ignores *how much* better the top result was |
| OR-tsquery | natural questions actually match | more noisy matches (ranking handles it) |
| Refusal = similarity OR coverage | fewer wrong refusals than either alone on the eval set | thresholds tuned on the same small set; must re-tune per embedder |
| Synchronous ingestion in the request | simple, the user sees errors immediately | big files block a worker; a job queue would scale better |
| Raw SQL instead of an ORM | the vector/full-text SQL is the interesting part and stays visible | more hand-written mapping |
| Fixed `vector(512)` | HNSW indexes need a fixed dimension | changing models with other sizes needs a migration |

## 8. Numbers (all measured; commands in README)

- 56 backend tests pass, 96% line coverage of `app/` (`REQUIRE_DB=1 pytest --cov`).
- Retrieval eval on 41 answerable questions: hybrid hit@1 0.780 / MRR@10 0.873 vs vector 0.756 / 0.830 and full-text
  0.659 / 0.809 (`python -m eval.run_eval`).
- Refusal: 100% of the 8 unanswerable questions refused, 90.2% of answerable ones answered.
- `/ask` p50 ≈ 2 ms (warm) on a 28-chunk corpus with local providers (`scripts/bench_ask.py`) — a small-corpus
  number, don't present it as a scalability claim.

## 9. Likely interview questions (with answers grounded in this code)

**1. Walk me through what happens when a user asks a question.**
Embed the question; run a pgvector cosine search and a Postgres full-text search, 30 candidates each
(`retrieval.py:55-56`); fuse with RRF and keep top-k; compute two confidence signals; refuse if both are weak;
otherwise send citations first and stream the answer over SSE (`main.py:151-161`).

**2. Why hybrid search? Isn't vector search enough?**
They fail differently. Embeddings handle paraphrase; keyword search nails exact tokens like `hnsw.ef_search`, error
codes, names and numbers. On my eval set hybrid beat both: hit@1 0.780 vs 0.756 (vector) and 0.659 (full-text),
MRR 0.873 vs 0.830 / 0.809.

**3. Explain Reciprocal Rank Fusion. Why k = 60?**
Each list contributes `1/(k + rank)`; sum per item, sort. Only ranks matter, so the incomparable score scales don't
matter. k = 60 is the value from the original paper (Cormack et al., 2009); smaller k rewards being first more
strongly. `tests/test_retrieval.py::test_rrf_small_k_amplifies_top_rank` shows the effect.

**4. How do you decide to refuse? Why not use the RRF score?**
RRF scores are relative — there's always a "best" result even for "What is the capital of Australia?". I use
absolute signals: best cosine similarity and best term coverage, and refuse only if both are low
(`retrieval.py:88`). On the eval set, similarity alone answered 78% of answerable questions, the combined rule
90.2%, both refusing all 8 unanswerable ones. Caveat: thresholds were chosen on that same set.

**5. How does chunking work and why overlap?**
Word-boundary chunks up to 800 characters, next chunk starts inside the last 150 characters of the previous one, never
crossing a page (`chunking.py:38-65`). Overlap keeps facts that straddle a boundary retrievable; page-local chunks
give exact page citations. Trade-off: smaller chunks = precise retrieval but less context per chunk.

**6. How do you handle duplicate uploads — including two at the same time?**
SHA-256 of the bytes; a pre-check skips all work for known files; for true races, `UNIQUE(content_hash)` +
`INSERT ... ON CONFLICT DO NOTHING RETURNING` means exactly one insert succeeds and the other reads the winner
(`service.py:41-53`). Tested with 8 concurrent uploads.

**7. What is HNSW and what can go wrong with it?**
A layered proximity graph for approximate nearest-neighbour search. It's approximate: it can miss true neighbours,
and because filtering (`WHERE embedding_model = ...`) happens after the index scan, a very selective filter can
return fewer rows than requested (`hnsw.ef_search` default 40 limits candidates; I fetch 30).

**8. Why store `embedding_model` on each document?**
Vectors from different models live in different spaces; comparing them gives meaningless similarities. Search
filters on the current model; `test_chunks_from_a_different_embedding_model_are_not_searched` proves old documents
are ignored rather than silently mis-ranked.

**9. How does streaming work end to end?**
FastAPI returns a `StreamingResponse` over a generator that yields `event: ...\ndata: {json}\n\n` blocks
(`main.py:73`, `151-166`). The browser can't use `EventSource` for POST, so it reads `fetch().body` through
`TextDecoderStream` and my `SseParser`, which buffers partial chunks (`sse.ts:13`). Citations arrive in the first
event so the UI can render sources before the answer finishes.

**10. How would you make it work with a real LLM, and how did you test that without paying for one?**
Set `LLM_PROVIDER=openai` (and optionally `EMBEDDING_PROVIDER=openai`) with `OPENAI_BASE_URL`/`OPENAI_API_KEY` —
works with OpenAI or local servers like Ollama. The HTTP code is tested against `httpx.MockTransport`, which checks
the request body (model, `dimensions`, `stream: true`) and feeds back a recorded-style SSE stream.

**11. How do your integration tests stay isolated?**
Each test creates a random Postgres schema, runs migrations into it, and points the connection pool's `search_path`
at it (`db.py:16-23`, `tests/conftest.py`). The schema is dropped afterwards. Same trick isolates the eval run.

**12. How do you prevent the LLM from hallucinating?**
Three layers: refuse when retrieval is weak, give the model only numbered passages with a prompt that requires
`[n]` citations and an explicit "couldn't find" answer, and return the passages so users can verify. Offline, the
extractive answerer copies sentences verbatim, so it can't invent facts. With a real LLM, hallucination is reduced,
not eliminated.

**13. What would you change to handle 1 million documents?**
Move ingestion to a background queue (uploads return a job id), batch embedding calls, tune HNSW (`m`,
`ef_construction`, `ef_search`) and measure recall, partition or filter by tenant, add authentication and per-user
document scoping, and cache question embeddings. Measure with a larger eval set first.

**14. Why Postgres full-text instead of Elasticsearch?**
For this scale the generated `tsvector` column + GIN index is enough, and keeping everything in one database means
one transaction for writes and one system to operate. Elasticsearch gives BM25, better analyzers and horizontal
scaling — worth it when search is the product.

**15. What are the weaknesses of your offline embedder?**
It's lexical: hashed words, bigrams and character trigrams. It won't match synonyms ("car"/"automobile"), and hash
collisions add noise. Short questions with one rare keyword can be refused ("How do I get rid of aphids?" has
similarity 0.076 and 33% coverage). It exists so the app, CI and eval run offline and deterministically.
