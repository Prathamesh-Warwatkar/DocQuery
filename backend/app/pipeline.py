from .answer import build_answer
from .retrieval import retrieve
from .tracing import Trace


def answer_question(conn, question: str, mode: str = "hybrid_rerank", k: int = 5, save_trace: bool = True) -> dict:
    trace = Trace()
    hits = retrieve(conn, question, mode, k, trace)
    with trace.stage("answer"):
        result = build_answer(question, hits)
    if save_trace:
        trace.save(conn, question, mode, len(hits), result["answered"], result["relevance"])
    result.update(
        mode=mode,
        retrieved=[{"chunk_id": h["chunk_id"], "doc": h["doc"], "heading": h["heading"]} for h in hits],
        trace={"stages_ms": trace.stages, "total_ms": trace.total_ms},
    )
    return result
