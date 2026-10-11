"""Evaluation harness: retrieval quality per mode + end-to-end answer quality and guardrail behaviour.

Usage (from backend/):  python -m eval.run_eval [--json out.json]
Metrics are computed on the gold section (document + heading) of each answerable question.
"""
import argparse
import json
import statistics
from pathlib import Path

from app.db import connect, init_schema
from app.ingest import ingest_directory
from app.pipeline import answer_question
from app.retrieval import MODES, retrieve
from app.tracing import Trace

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
K = 5


def is_gold(hit: dict, item: dict) -> bool:
    return hit["doc"].startswith(item["doc"]) and hit["heading"] == item["heading"]


def first_gold_rank(hits: list[dict], item: dict) -> int | None:
    for rank, h in enumerate(hits, start=1):
        if is_gold(h, item):
            return rank
    return None


def retrieval_metrics(conn, items: list[dict], mode: str) -> dict:
    ranks, lat = [], []
    for it in items:
        t = Trace()
        hits = retrieve(conn, it["question"], mode, K, t)
        lat.append(t.total_ms)
        ranks.append(first_gold_rank(hits, it))
    n = len(items)
    hit = lambda c: sum(1 for r in ranks if r is not None and r <= c) / n
    mrr = sum(1 / r for r in ranks if r) / n
    lat_sorted = sorted(lat)
    return {
        "n": n, "hit@1": hit(1), "hit@3": hit(3), "hit@5": hit(5), "mrr": mrr,
        "p50_ms": statistics.median(lat), "p95_ms": lat_sorted[max(0, int(0.95 * n) - 1)],
    }


def answer_metrics(conn, items: list[dict], mode: str) -> dict:
    ans = [i for i in items if i["answerable"]]
    unans = [i for i in items if not i["answerable"]]
    answered = correct = cited_gold = 0
    for it in ans:
        r = answer_question(conn, it["question"], mode, K, save_trace=False)
        if r["answered"]:
            answered += 1
            text = r["answer"].lower()
            correct += any(s in text for s in it["answer_any"])
            cited_gold += any(c["doc"].startswith(it["doc"]) and c["heading"] == it["heading"] for c in r["citations"])
    refused_unans = sum(
        1 for it in unans if not answer_question(conn, it["question"], mode, K, save_trace=False)["answered"]
    )
    return {
        "answerable": len(ans), "answered": answered, "false_refusal_rate": 1 - answered / len(ans),
        "answer_contains_key_fact": correct / len(ans), "citation_hits_gold": cited_gold / len(ans),
        "unanswerable": len(unans), "correctly_refused": refused_unans / len(unans) if unans else None,
    }


def run(conn, items: list[dict]) -> dict:
    answerable = [i for i in items if i["answerable"]]
    out = {"retrieval": {}, "answers": {}}
    for mode in MODES:
        out["retrieval"][mode] = retrieval_metrics(conn, answerable, mode)
        out["answers"][mode] = answer_metrics(conn, items, mode)
    return out


def fmt(report: dict, title: str) -> str:
    lines = [f"### {title}", "", "| mode | hit@1 | hit@3 | hit@5 | MRR | p50 ms | p95 ms |", "|---|---|---|---|---|---|---|"]
    for mode, m in report["retrieval"].items():
        lines.append(f"| {mode} | {m['hit@1']:.2f} | {m['hit@3']:.2f} | {m['hit@5']:.2f} | {m['mrr']:.3f} | {m['p50_ms']:.1f} | {m['p95_ms']:.1f} |")
    lines += ["", "| mode | false refusals | key fact in answer | citation hits gold section | unanswerable correctly refused |", "|---|---|---|---|---|"]
    for mode, a in report["answers"].items():
        lines.append(f"| {mode} | {a['false_refusal_rate']:.2f} | {a['answer_contains_key_fact']:.2f} | {a['citation_hits_gold']:.2f} | {a['correctly_refused']:.2f} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    items = json.loads((HERE / "eval_set.json").read_text())
    with connect() as conn:
        init_schema(conn)
        ingest_directory(conn, DATA)
        full = run(conn, items)
        test = run(conn, [i for i in items if i["split"] == "test"])
    print(fmt(full, f"All {len(items)} questions"))
    print()
    print(fmt(test, f"Held-out test split ({sum(1 for i in items if i['split'] == 'test')} questions)"))
    if args.json:
        Path(args.json).write_text(json.dumps({"all": full, "test": test}, indent=2))


if __name__ == "__main__":
    main()
