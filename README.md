# DocQuery: citation-grounded hybrid-search RAG

A small full-stack retrieval-augmented question-answering app over contracts and policies.

- **Backend (Python, FastAPI):** heading-aware chunking, embeddings stored in **PostgreSQL + pgvector** (HNSW index),
  Postgres full-text search (GIN index), **hybrid retrieval with Reciprocal Rank Fusion**, a feature-based
  **re-ranker**, **citation-grounded answers** with a relevance **guardrail** (refuses when the documents do not
  contain the answer), and per-stage **request tracing** stored in Postgres.
- **Frontend (Next.js + TypeScript):** question box, retrieval-mode switch, cited answer, source snippets, stage timing chart.
- **Evals:** gold-section eval set, retrieval metrics per mode (hit@k, MRR, latency), answer/guardrail metrics,
  and pytest regression floors.

## Run it

```bash
# 1. Postgres 16 with the pgvector extension, then:
export DATABASE_URL="postgresql://postgres@localhost:5432/ragdb"   # default: local socket, port 5433

# 2. Backend
cd backend && python -m venv .venv && . .venv/bin/activate
pip install fastapi uvicorn "psycopg[binary]" scikit-learn numpy pypdf python-multipart pytest httpx openai anthropic
uvicorn app.main:app --port 8000
curl -X POST localhost:8000/ingest/sample          # index the 6 sample documents

# 3. Frontend
cd frontend && npm install && npm run dev           # http://localhost:3000

# 4. Tests and evals
cd backend && pytest -q
python -m eval.run_eval --json eval/results.json
```

API: `POST /query {question, mode, k}`, `POST /ingest`, `POST /upload` (.md/.txt/.pdf), `GET /documents`, `GET /traces`.
Modes: `vector`, `keyword`, `hybrid`, `hybrid_rerank`.

## Providers

| Part | Default (offline, tested) | Optional (written, **not tested live**) |
|---|---|---|
| Embeddings | 384-dim word + character n-gram feature hashing | Azure OpenAI (`EMBEDDINGS_PROVIDER=azure_openai`) |
| Answers | Extractive: quotes best sentences, tags `[n]` sources | Anthropic (`ANSWER_PROVIDER=anthropic`), output validated so every `[n]` maps to a retrieved source |
| Re-ranker | Feature-based (term coverage, bigrams, heading match) blended with fused score | Swap in a cross-encoder behind `rerank()` |

## Results (31-question eval set, 6 synthetic sample documents)

See `backend/eval/results.md`. Headline retrieval numbers (all 31 questions; 27 answerable):

| mode | hit@1 | hit@5 | MRR |
|---|---|---|---|
| vector only | 0.41 | 0.74 | 0.527 |
| keyword only | 0.56 | 0.85 | 0.651 |
| hybrid (RRF) | 0.56 | 0.89 | 0.667 |
| hybrid + re-rank | 0.63 | 0.93 | 0.720 |

## Known limitations (read before quoting any number)

- The corpus is six short **synthetic** documents and the 31 questions were written by the same author, so the
  numbers are indicative, not a benchmark. The guardrail threshold was chosen on the dev half; the held-out half is
  reported separately in `results.md` and is lower: vector-only MRR 0.48, hybrid 0.62, hybrid + re-rank 0.61. The re-ranker's gain over plain hybrid is small and not visible on the held-out half; the clear gain is hybrid over vector-only.
- The default embedder has no pretrained semantics, so paraphrases with different vocabulary can retrieve the wrong
  section (for example "data leak" vs "breach" returns an encryption clause). Real semantic embeddings are the intended fix.
- The guardrail uses term overlap, which is crude: it refuses some answerable questions (26% false refusals on all questions and 31% on the
  held-out half in the best mode). Only 4 unanswerable questions were tested.
- Extractive answers quote sentences; they do not synthesize. The LLM answer path and Azure embeddings are untested live.
- Tracing is a built-in Postgres table (stage timings), not an external observability product.
