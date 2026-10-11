"use client";

import { useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Citation = { n: number; chunk_id: number; doc: string; heading: string; snippet: string };
type QueryResult = {
  answer: string;
  answered: boolean;
  relevance: number;
  mode: string;
  citations: Citation[];
  retrieved: { chunk_id: number; doc: string; heading: string }[];
  trace: { stages_ms: Record<string, number>; total_ms: number };
};

const MODES = [
  { value: "hybrid_rerank", label: "Hybrid + re-rank" },
  { value: "hybrid", label: "Hybrid (RRF)" },
  { value: "keyword", label: "Keyword only" },
  { value: "vector", label: "Vector only" },
];

const EXAMPLES = [
  "How quickly must a personal data breach be reported?",
  "What refund applies if uptime drops below 99 percent?",
  "What is the parental leave policy?",
];

export default function Home() {
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState("hybrid_rerank");
  const [result, setResult] = useState<QueryResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  async function ask(q: string = question) {
    if (q.trim().length < 3) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, mode, k: 5 }),
      });
      if (!res.ok) throw new Error(`API returned ${res.status}`);
      setResult(await res.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  async function loadSamples() {
    setStatus("Loading sample documents…");
    try {
      const res = await fetch(`${API}/ingest/sample`, { method: "POST" });
      const data = await res.json();
      setStatus(`Indexed ${Object.keys(data.ingested).length} sample documents.`);
    } catch {
      setStatus("Could not reach the API. Is the backend running?");
    }
  }

  const maxMs = result ? Math.max(...Object.values(result.trace.stages_ms), 0.001) : 1;

  return (
    <main>
      <h1>DocQuery</h1>
      <p className="sub">
        Ask questions over your documents. Answers cite the exact section they came from, and the system refuses
        when the documents do not contain the answer.
      </p>

      <div className="card">
        <form
          className="row"
          onSubmit={(e) => {
            e.preventDefault();
            void ask();
          }}
        >
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question about the indexed contracts and policies"
            aria-label="Question"
          />
          <select value={mode} onChange={(e) => setMode(e.target.value)} aria-label="Retrieval mode">
            {MODES.map((m) => (
              <option key={m.value} value={m.value}>
                {m.label}
              </option>
            ))}
          </select>
          <button className="primary" type="submit" disabled={loading}>
            {loading ? "Searching…" : "Ask"}
          </button>
        </form>
        <div className="chips">
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              className="chip"
              type="button"
              onClick={() => {
                setQuestion(ex);
                void ask(ex);
              }}
            >
              {ex}
            </button>
          ))}
        </div>
        <p className="meta" style={{ marginTop: 12 }}>
          <button type="button" onClick={loadSamples}>
            Load sample documents
          </button>{" "}
          {status}
        </p>
      </div>

      {error && <p className="err">{error}</p>}

      {result && (
        <>
          <div className="card">
            {result.answered ? (
              <p className="answer">{result.answer}</p>
            ) : (
              <p className="refused">{result.answer}</p>
            )}
            <p className="meta">
              mode: {result.mode} · relevance: {result.relevance} · {result.trace.total_ms.toFixed(1)} ms total
            </p>
            {result.citations.map((c) => (
              <div className="cite" key={c.chunk_id}>
                <b>[{c.n}]</b> {c.doc} — <i>{c.heading}</i>
                <div className="meta">{c.snippet}</div>
              </div>
            ))}
          </div>

          <div className="card">
            <strong>Pipeline trace</strong>
            {Object.entries(result.trace.stages_ms).map(([stage, ms]) => (
              <div className="stage" key={stage}>
                <span>{stage}</span>
                <div className="bar" style={{ width: `${Math.max((ms / maxMs) * 100, 1)}%` }} />
                <span>{ms.toFixed(1)} ms</span>
              </div>
            ))}
          </div>
        </>
      )}
    </main>
  );
}
