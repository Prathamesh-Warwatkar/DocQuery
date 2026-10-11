"""Heading-aware chunking: split on markdown headings, then pack sentences into
windows of at most `max_words` words with a sentence of overlap between windows."""
import re
from dataclasses import dataclass

from .textutil import split_sentences

_HEADING = re.compile(r"^#{1,6}\s+(.*)$")


@dataclass
class Chunk:
    ord: int
    heading: str
    body: str


def split_sections(text: str, default_heading: str = "Overview") -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = []
    heading, buf = default_heading, []
    for line in text.splitlines():
        m = _HEADING.match(line.strip())
        if m:
            if any(x.strip() for x in buf):
                sections.append((heading, buf))
            heading, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    if any(x.strip() for x in buf):
        sections.append((heading, buf))
    return [(h, " ".join(" ".join(b).split())) for h, b in sections]


def chunk_text(text: str, max_words: int = 110, overlap_sentences: int = 1) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading, body in split_sections(text):
        sentences = split_sentences(body)
        window: list[str] = []
        words = 0
        for sent in sentences:
            n = len(sent.split())
            if window and words + n > max_words:
                chunks.append(Chunk(len(chunks), heading, " ".join(window)))
                window = window[-overlap_sentences:] if overlap_sentences else []
                words = sum(len(s.split()) for s in window)
            window.append(sent)
            words += n
        if window:
            chunks.append(Chunk(len(chunks), heading, " ".join(window)))
    return chunks
