from app.answer import NO_ANSWER, extractive_answer, validate_citations
from app.chunking import chunk_text, split_sections
from app.retrieval import rrf_fuse
from app.rerank import features, rerank
from app.textutil import content_stems, stem


def test_stem_and_stopwords():
    assert stem("payments") == "payment"
    assert content_stems("How long does the contract last?") == ["long", "contract", "last"]


def test_split_sections_uses_headings():
    text = "# Title\nintro line.\n\n## A\nOne. Two.\n\n## B\nThree."
    assert [h for h, _ in split_sections(text)] == ["Title", "A", "B"]


def test_chunk_text_windows_with_overlap():
    body = " ".join(f"Sentence number {i} has exactly seven words." for i in range(30))
    chunks = chunk_text(f"## Long\n{body}", max_words=40, overlap_sentences=1)
    assert len(chunks) > 3
    assert all(c.heading == "Long" for c in chunks)
    # overlap: the last sentence of a chunk starts the next one
    last = chunks[0].body.split(". ")[-1].rstrip(".")
    assert chunks[1].body.startswith(last)
    assert all(len(c.body.split()) <= 40 + 8 for c in chunks)


def test_rrf_prefers_documents_ranked_well_by_both_lists():
    fused = rrf_fuse([[1, 2, 3], [9, 2, 4]], k=60)
    order = [i for i, _ in fused]
    assert order[0] == 2 and set(order) == {1, 2, 3, 4, 9}
    assert dict(fused)[2] > dict(fused)[1]


def test_rerank_moves_matching_chunk_up():
    hits = [
        {"chunk_id": 1, "doc": "d", "heading": "Fees", "body": "Pay invoices within forty-five days.", "fused_score": 0.03},
        {"chunk_id": 2, "doc": "d", "heading": "Term", "body": "The initial term is thirty-six months.", "fused_score": 0.033},
    ]
    out = rerank("how long is the initial term", hits)
    assert out[0]["chunk_id"] == 2
    assert features("how long is the initial term", out[0])["coverage"] > 0.5


def test_validate_citations():
    assert validate_citations("Pay in 45 days [1][2].", 3)["grounded"]
    assert not validate_citations("Pay in 45 days.", 3)["grounded"]
    assert validate_citations("Pay in 45 days [7].", 3)["invalid"] == [7]


def test_extractive_answer_cites_its_sources():
    hits = [{"chunk_id": 5, "doc": "d", "heading": "Term", "body": "The initial term is thirty-six months. It renews yearly."}]
    ans = extractive_answer("what is the initial term", hits)
    assert "thirty-six months" in ans and ans.rstrip().endswith("[1]")
    assert NO_ANSWER not in ans
