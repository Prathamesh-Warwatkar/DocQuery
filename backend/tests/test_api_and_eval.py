"""Integration tests: need the Postgres + pgvector database from DATABASE_URL."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.db import connect, init_schema
from app.ingest import ingest_directory
from app.main import DATA_DIR, app
from eval.run_eval import retrieval_metrics

ITEMS = json.loads((Path(__file__).resolve().parent.parent / "eval" / "eval_set.json").read_text())


@pytest.fixture(scope="module")
def conn():
    with connect() as c:
        init_schema(c)
        ingest_directory(c, DATA_DIR)
        yield c


def test_query_returns_cited_answer(conn):
    with TestClient(app) as client:
        r = client.post("/query", json={"question": "What is the initial term of the agreement?", "mode": "hybrid_rerank"})
    body = r.json()
    assert r.status_code == 200 and body["answered"]
    assert "thirty-six months" in body["answer"].lower()
    assert body["citations"] and body["citations"][0]["heading"] == "Term and Renewal"
    assert {"embed", "vector_search", "keyword_search", "rerank", "answer"} <= set(body["trace"]["stages_ms"])


def test_unanswerable_question_is_refused(conn):
    with TestClient(app) as client:
        body = client.post("/query", json={"question": "What is the parental leave policy?"}).json()
    assert body["answered"] is False and body["citations"] == []


def test_bad_mode_rejected(conn):
    with TestClient(app) as client:
        assert client.post("/query", json={"question": "anything here", "mode": "nope"}).status_code == 400


def test_traces_are_recorded(conn):
    with TestClient(app) as client:
        client.post("/query", json={"question": "How long is the notice for renewal?"})
        traces = client.get("/traces?limit=1").json()
    assert traces and traces[0]["total_ms"] > 0


def test_retrieval_quality_regression_floors(conn):
    """Floors sit a little below measured values (hybrid_rerank MRR ~0.72, hit@5 ~0.93)."""
    answerable = [i for i in ITEMS if i["answerable"]]
    vec = retrieval_metrics(conn, answerable, "vector")
    best = retrieval_metrics(conn, answerable, "hybrid_rerank")
    assert best["mrr"] >= 0.65
    assert best["hit@5"] >= 0.85
    assert best["mrr"] > vec["mrr"]
