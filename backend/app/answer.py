"""Citation-grounded answer generation with a relevance guardrail.

Default provider is extractive: it quotes the best-matching sentences from the retrieved chunks, each
tagged with the number of the source chunk it came from, so every statement is traceable. An LLM provider
(Anthropic) is available via ANSWER_PROVIDER=anthropic; its output is validated so that every [n] citation
maps to a retrieved source, and it falls back to the extractive answer otherwise.
NOTE: the Anthropic path is written against the SDK but was not exercised live in the build environment.
"""
import os
import re

from .config import ANSWER_PROVIDER, MIN_RELEVANCE
from .rerank import features
from .textutil import content_stems, split_sentences

NO_ANSWER = "I could not find this in the indexed documents."


def validate_citations(answer: str, n_sources: int) -> dict:
    cited = sorted({int(x) for x in re.findall(r"\[(\d+)\]", answer)})
    invalid = [c for c in cited if not 1 <= c <= n_sources]
    return {"cited": cited, "invalid": invalid, "grounded": bool(cited) and not invalid}


def extractive_answer(query: str, hits: list[dict], max_sentences: int = 2, max_sources: int = 3) -> str:
    q = set(content_stems(query))
    scored = []
    for n, hit in enumerate(hits[:max_sources], start=1):
        for pos, sent in enumerate(split_sentences(hit["body"])):
            cov = len(q & set(content_stems(sent))) / (len(q) or 1)
            scored.append((cov, -n, -pos, n, pos, sent))
    scored = [s for s in scored if s[0] > 0]
    scored.sort(reverse=True)
    chosen = sorted(scored[:max_sentences], key=lambda s: (s[3], s[4]))
    return " ".join(f"{s[5]} [{s[3]}]" for s in chosen)


def _llm_answer(query: str, hits: list[dict], max_sources: int = 4) -> str | None:
    if ANSWER_PROVIDER != "anthropic" or not os.getenv("ANTHROPIC_API_KEY"):
        return None
    import anthropic

    sources = "\n\n".join(f"[{i}] ({h['doc']} / {h['heading']}) {h['body']}" for i, h in enumerate(hits[:max_sources], 1))
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
        max_tokens=400,
        system=(
            "Answer using ONLY the numbered sources. Cite every claim with its source number like [1]. "
            f"If the sources do not contain the answer, reply exactly: {NO_ANSWER}"
        ),
        messages=[{"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {query}"}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()


def build_answer(query: str, hits: list[dict]) -> dict:
    """Returns {answer, answered, relevance, citations}."""
    if not hits:
        return {"answer": NO_ANSWER, "answered": False, "relevance": 0.0, "citations": []}
    relevance = features(query, hits[0])["coverage"]
    if relevance < MIN_RELEVANCE:
        return {"answer": NO_ANSWER, "answered": False, "relevance": round(relevance, 3), "citations": []}

    text = None
    try:
        text = _llm_answer(query, hits)
    except Exception:  # network/SDK failure -> fall back to extractive
        text = None
    if text is not None:
        check = validate_citations(text, min(len(hits), 4))
        if text == NO_ANSWER:
            return {"answer": NO_ANSWER, "answered": False, "relevance": round(relevance, 3), "citations": []}
        if not check["grounded"]:
            text = None
    if text is None:
        text = extractive_answer(query, hits)

    cited = sorted({int(x) for x in re.findall(r"\[(\d+)\]", text)})
    citations = [
        {"n": n, "chunk_id": hits[n - 1]["chunk_id"], "doc": hits[n - 1]["doc"],
         "heading": hits[n - 1]["heading"], "snippet": hits[n - 1]["body"]}
        for n in cited if 1 <= n <= len(hits)
    ]
    return {"answer": text, "answered": bool(text), "relevance": round(relevance, 3), "citations": citations}
