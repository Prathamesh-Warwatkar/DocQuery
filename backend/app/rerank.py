"""Feature-based re-ranker applied to the fused candidate pool.

Scores each candidate on (1) share of question terms present, (2) share of question term-bigrams present
in order, (3) share of question terms found in the section heading, then blends with the fused retrieval
score. Deterministic and offline; swap for a cross-encoder or LLM reranker behind the same signature.
"""
from .textutil import content_stems

W_COVERAGE, W_BIGRAM, W_HEADING = 0.5, 0.2, 0.3
W_FEATURES, W_FUSED = 0.65, 0.35


def features(query: str, hit: dict) -> dict:
    q = content_stems(query)
    q_set = set(q)
    if not q_set:
        return {"coverage": 0.0, "bigram": 0.0, "heading_cov": 0.0}
    body = content_stems(hit["body"])
    head = content_stems(hit["heading"])
    doc_set = set(body) | set(head)
    coverage = len(q_set & doc_set) / len(q_set)
    q_bi = set(zip(q, q[1:]))
    d_seq = head + body
    d_bi = set(zip(d_seq, d_seq[1:]))
    bigram = len(q_bi & d_bi) / len(q_bi) if q_bi else 0.0
    heading = len(q_set & set(head)) / len(q_set)
    return {"coverage": coverage, "bigram": bigram, "heading_cov": heading}


def rerank(query: str, hits: list[dict]) -> list[dict]:
    if not hits:
        return hits
    max_fused = max(h.get("fused_score", 0.0) for h in hits) or 1.0
    for h in hits:
        f = features(query, h)
        h.update(f)
        feat = W_COVERAGE * f["coverage"] + W_BIGRAM * f["bigram"] + W_HEADING * f["heading_cov"]
        h["rerank_score"] = W_FEATURES * feat + W_FUSED * (h.get("fused_score", 0.0) / max_fused)
    return sorted(hits, key=lambda h: h["rerank_score"], reverse=True)
