from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .db import connect, init_schema
from .ingest import extract_text, ingest_directory, ingest_document
from .pipeline import answer_question
from .retrieval import MODES

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@asynccontextmanager
async def lifespan(app: FastAPI):
    with connect() as conn:
        init_schema(conn)
    yield


app = FastAPI(title="DocQuery", description="Citation-grounded hybrid-search RAG over Postgres + pgvector", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["http://localhost:3000"], allow_methods=["*"], allow_headers=["*"]
)


class QueryIn(BaseModel):
    question: str = Field(min_length=3)
    mode: str = "hybrid_rerank"
    k: int = Field(default=5, ge=1, le=20)


class DocIn(BaseModel):
    title: str
    text: str


@app.get("/health")
def health():
    return {"status": "ok", "modes": MODES}


@app.post("/ingest/sample")
def ingest_sample():
    with connect() as conn:
        return {"ingested": ingest_directory(conn, DATA_DIR)}


@app.post("/ingest")
def ingest(doc: DocIn):
    with connect() as conn:
        try:
            return {"title": doc.title, "chunks": ingest_document(conn, doc.title, doc.text)}
        except ValueError as e:
            raise HTTPException(400, str(e))


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    text = extract_text(file.filename or "upload.txt", data)
    with connect() as conn:
        try:
            return {"title": file.filename, "chunks": ingest_document(conn, file.filename or "upload", text, "upload")}
        except ValueError as e:
            raise HTTPException(400, str(e))


@app.get("/documents")
def documents():
    with connect() as conn:
        rows = conn.execute(
            "select d.id, d.title, count(c.id) from documents d left join chunks c on c.doc_id = d.id "
            "group by d.id order by d.id"
        ).fetchall()
    return [{"id": r[0], "title": r[1], "chunks": r[2]} for r in rows]


@app.post("/query")
def query(q: QueryIn):
    if q.mode not in MODES:
        raise HTTPException(400, f"mode must be one of {MODES}")
    with connect() as conn:
        return answer_question(conn, q.question, q.mode, q.k)


@app.get("/traces")
def traces(limit: int = 20):
    with connect() as conn:
        rows = conn.execute(
            "select id, created_at, question, mode, stages, total_ms, n_hits, answered, top_relevance "
            "from traces order by id desc limit %s",
            (min(limit, 100),),
        ).fetchall()
    keys = ("id", "created_at", "question", "mode", "stages_ms", "total_ms", "n_hits", "answered", "top_relevance")
    return [dict(zip(keys, r)) for r in rows]
