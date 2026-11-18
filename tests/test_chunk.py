"""Tests for ingest.chunk: window sizes, overlap, word boundaries, and metadata."""

import pytest

from ingest.chunk import chunk_pages, chunk_text, get_tokenizer


@pytest.fixture(scope="module")
def tok():
    return get_tokenizer()


def n_tokens(tok, text):
    return len(tok(text, add_special_tokens=False)["input_ids"])


LONG = " ".join(f"Segment{i} revenue grew to ₹{i},250 crore in FY2025." for i in range(300))


def test_chunks_respect_size_limit(tok):
    chunks = chunk_text(LONG, tok, size=100, overlap=20, min_tokens=5)
    assert len(chunks) > 1
    assert all(n_tokens(tok, c) <= 100 for c in chunks)


def test_consecutive_chunks_overlap(tok):
    chunks = chunk_text(LONG, tok, size=100, overlap=20, min_tokens=5)
    for a, b in zip(chunks, chunks[1:]):
        assert b[:30] in a  # the next chunk starts inside the previous one


def test_whole_text_is_covered(tok):
    chunks = chunk_text(LONG, tok, size=100, overlap=20, min_tokens=5)
    assert LONG.startswith(chunks[0]) and LONG.endswith(chunks[-1])


def test_no_word_is_cut_in_half(tok):
    words = set(LONG.split())
    for c in chunk_text(LONG, tok, size=100, overlap=20, min_tokens=5):
        assert c.split()[0] in words and c.split()[-1] in words


def test_original_casing_is_kept(tok):
    assert chunk_text("HDFC Bank Net Profit rose sharply this year", tok, min_tokens=1) == [
        "HDFC Bank Net Profit rose sharply this year"
    ]


def test_short_text_is_dropped(tok):
    assert chunk_text("Notes", tok, min_tokens=20) == []


def test_chunk_pages_copies_metadata_and_skips_garbled(tok):
    meta = {"source_file": "a.pdf", "company": "Acme", "year": "FY2025", "doc_type": "annual_report"}
    pages = [
        {**meta, "page": 7, "text": LONG, "garbled": False},
        {**meta, "page": 8, "text": LONG, "garbled": True},
    ]
    chunks = chunk_pages(pages, tok)
    assert chunks and all(c["page"] == 7 for c in chunks)
    assert chunks[0]["chunk_id"] == "a.pdf:p7:c0"
    for key in ("company", "year", "doc_type", "source_file"):
        assert all(c[key] == meta[key] for c in chunks)
