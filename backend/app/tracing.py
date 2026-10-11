import json
import time
from contextlib import contextmanager


class Trace:
    """Collects per-stage wall-clock timings for one request."""

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self._t0 = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = round((time.perf_counter() - t) * 1000, 3)

    @property
    def total_ms(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000, 3)

    def save(self, conn, question: str, mode: str, n_hits: int, answered: bool, top_relevance: float) -> None:
        conn.execute(
            "insert into traces(question, mode, stages, total_ms, n_hits, answered, top_relevance) "
            "values (%s, %s, %s::jsonb, %s, %s, %s, %s)",
            (question, mode, json.dumps(self.stages), self.total_ms, n_hits, answered, top_relevance),
        )
