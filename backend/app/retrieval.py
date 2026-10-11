"""Hybrid retrieval over Postgres: pgvector cosine search + full-text search, fused with Reciprocal Rank Fusion."""
import re

from .config import CANDIDATE_POOL, RRF_K
from .embeddings import get_embedder, to_pgvector
from .textutil import STOPWORDS, tokenize
from .tracing import Trace

MODES = ("vector", "keyword", "hybrid", "hybrid_rerank")


def vector_search(conn, query_vec, limit: int) -> list[int]:
    vec = to_pgvector(query_vec)
    rows = conn.execute(
        "select id from chunks order by embedding <=> %s::vector limit %s", (vec, limit)
    ).fetchall()
    return [r[0] for r in rows]


def keyword_search(conn, query: str, limit: int) -> list[int]:
    """OR-query full-text search (natural-language questions rarely match as an AND)."""
    terms = sorted({t for t in tokenize(query) if t not in STOPWORDS and len(t) > 1})
    if not terms:
        return []
    tsq = " | ".join(re.sub(r"[^a-z0-9]", "", t) for t in terms)
    rows = conn.execute(
        "select id from chunks, to_tsquery('english', %s) q where tsv @@ q "
        "order by ts_rank_cd(tsv, q, 32) desc limit %s",
        (tsq, limit),
    ).fetchall()
    return [r[0] for r in rows]


def rrf_fuse(rank_lists: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank)."""
    scores: dict[int, float] = {}
    for ranking in rank_lists:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def fetch_chunks(conn, ordered_ids: list[int]) -> list[dict]:
    if not ordered_ids:
        return []
    rows = conn.execute(
        "select c.id, d.title, c.heading, c.body from chunks c join documents d on d.id = c.doc_id "
        "where c.id = any(%s)",
        (ordered_ids,),
    ).fetchall()
    by_id = {r[0]: {"chunk_id": r[0], "doc": r[1], "heading": r[2], "body": r[3]} for r in rows}
    return [by_id[i] for i in ordered_ids if i in by_id]


def retrieve(conn, query: str, mode: str, k: int = 5, trace: Trace | None = None) -> list[dict]:
    """Return the top-k chunks for `query` using the given retrieval mode."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    trace = trace or Trace()
    pool = max(CANDIDATE_POOL, k)

    vec_ids: list[int] = []
    kw_ids: list[int] = []
    if mode in ("vector", "hybrid", "hybrid_rerank"):
        with trace.stage("embed"):
            qvec = get_embedder().embed([query])[0]
        with trace.stage("vector_search"):
            vec_ids = vector_search(conn, qvec, pool)
    if mode in ("keyword", "hybrid", "hybrid_rerank"):
        with trace.stage("keyword_search"):
            kw_ids = keyword_search(conn, query, pool)

    with trace.stage("fuse"):
        if mode == "vector":
            fused = [(i, 0.0) for i in vec_ids]
        elif mode == "keyword":
            fused = [(i, 0.0) for i in kw_ids]
        else:
            fused = rrf_fuse([vec_ids, kw_ids])
    fused_scores = dict(fused)
    candidate_ids = [i for i, _ in fused][: pool if mode == "hybrid_rerank" else k]

    with trace.stage("fetch"):
        hits = fetch_chunks(conn, candidate_ids)
    for h in hits:
        h["fused_score"] = fused_scores.get(h["chunk_id"], 0.0)

    if mode == "hybrid_rerank":
        from .rerank import rerank

        with trace.stage("rerank"):
            hits = rerank(query, hits)
    return hits[:k]
