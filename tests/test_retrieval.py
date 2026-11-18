"""Tests for retrieval: embeddings, index build, search, filters and stale-index detection."""

import json

import numpy as np
import pytest

from retrieval.embed import embed_passages, embed_query
from retrieval.index import build_index
from retrieval.retrieve import Retriever

CHUNKS = [
    ("Acme", "FY2025", 3, "Net profit for the year rose 12% to ₹ 4,500 crore."),
    ("Acme", "FY2025", 9, "The Board recommended a final dividend of ₹ 20 per share."),
    ("Acme", "FY2026", 4, "Net profit for the year fell 5% to ₹ 4,275 crore."),
    ("Globex", "FY2025", 7, "Our data centres in Pune now run fully on renewable energy."),
]


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    d = tmp_path_factory.mktemp("index")
    chunks_path = d / "chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as f:
        for company, year, page, text in CHUNKS:
            f.write(json.dumps({
                "chunk_id": f"{company}.pdf:p{page}:c0", "source_file": f"{company}.pdf",
                "company": company, "year": year, "doc_type": "annual_report",
                "page": page, "text": text, "n_tokens": 0,
            }) + "\n")
    paths = {"index_path": d / "faiss.index", "chunks_path": chunks_path,
             "meta_path": d / "index_meta.json"}
    build_index(**paths)
    return paths


@pytest.fixture(scope="module")
def retriever(paths):
    return Retriever.load(**paths)


def test_embeddings_are_normalised():
    vecs = embed_passages(["revenue grew", "dividend declared"])
    assert vecs.dtype == np.float32
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
    assert embed_query("revenue").shape == (1, vecs.shape[1])


def test_relevant_chunk_ranks_first(retriever):
    top = retriever.search("How much dividend per share was recommended?", k=2)
    assert top[0]["page"] == 9
    assert [r["rank"] for r in top] == [1, 2]
    assert top[0]["score"] >= top[1]["score"]


def test_results_carry_citation_metadata(retriever):
    r = retriever.search("renewable energy data centres", k=1)[0]
    assert (r["company"], r["year"], r["page"], r["source_file"]) == ("Globex", "FY2025", 7, "Globex.pdf")
    assert r["source"] == "Globex FY2025 annual_report (Globex.pdf), p. 7"


def test_where_filter_restricts_results(retriever):
    results = retriever.search("net profit", k=5, where={"company": "Acme", "year": "FY2026"})
    assert [r["page"] for r in results] == [4]


def test_k_larger_than_index(retriever):
    assert len(retriever.search("profit", k=50)) == len(CHUNKS)


def test_with_chunks_adds_documents_without_changing_original(retriever):
    upload = {"chunk_id": "Initech.pdf:p2:c0", "source_file": "Initech.pdf", "company": "Initech",
              "year": "FY2025", "doc_type": "annual_report", "page": 2, "n_tokens": 0,
              "text": "Initech opened a new office in Hyderabad with 800 seats."}
    extended = retriever.with_chunks([upload])
    top = extended.search("Where did Initech open a new office?", k=1)[0]
    assert top["company"] == "Initech" and top["page"] == 2
    assert extended.index.ntotal == len(CHUNKS) + 1
    assert retriever.index.ntotal == len(CHUNKS)  # original untouched


def test_stale_index_is_rejected(paths, tmp_path):
    changed = tmp_path / "chunks.jsonl"
    changed.write_text(paths["chunks_path"].read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed since the index was built"):
        Retriever.load(paths["index_path"], changed, paths["meta_path"])
