import io
from pathlib import Path

from .chunking import chunk_text
from .embeddings import get_embedder, to_pgvector


def extract_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((p.extract_text() or "") for p in reader.pages)
    return data.decode("utf-8", errors="replace")


def ingest_document(conn, title: str, text: str, source: str = "upload") -> int:
    """Chunk, embed and store a document. Re-ingesting the same title replaces it."""
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("document has no extractable text")
    conn.execute("delete from documents where title = %s", (title,))
    doc_id = conn.execute(
        "insert into documents(title, source) values (%s, %s) returning id", (title, source)
    ).fetchone()[0]
    vectors = get_embedder().embed([f"{c.heading}. {c.body}" for c in chunks])
    with conn.cursor() as cur:
        cur.executemany(
            "insert into chunks(doc_id, ord, heading, body, embedding) values (%s, %s, %s, %s, %s::vector)",
            [(doc_id, c.ord, c.heading, c.body, to_pgvector(v)) for c, v in zip(chunks, vectors)],
        )
    return len(chunks)


def ingest_directory(conn, directory: Path) -> dict[str, int]:
    out = {}
    for path in sorted(directory.glob("*.md")):
        first = path.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()
        out[first] = ingest_document(conn, first, path.read_text(encoding="utf-8"), source=path.name)
    return out
