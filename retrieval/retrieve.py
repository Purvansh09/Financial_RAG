"""retrieve(query, k) -> the k most similar chunks, each with its score and citation.

Retrieval knows nothing about the LLM, so it can be evaluated on its own (hit rate @k).
"""

import json
import sys
from functools import lru_cache
from pathlib import Path

import faiss

import config
from retrieval.embed import embed_passages, embed_query
from retrieval.index import file_sha256, load_chunks


def format_source(chunk: dict) -> str:
    """Human-readable citation, e.g. 'HDFC Bank FY2025 annual_report (hdfcbank_...pdf), p. 12'."""
    return (f"{chunk['company']} {chunk['year']} {chunk['doc_type']} "
            f"({chunk['source_file']}), p. {chunk['page']}")


class Retriever:
    """Holds a FAISS index and its chunks in memory and answers similarity searches.

    Row i of `index` must be the vector of `chunks[i]`.
    """

    def __init__(self, index: faiss.Index, chunks: list[dict]):
        if index.ntotal != len(chunks):
            raise ValueError(f"Index has {index.ntotal} vectors but there are {len(chunks)} chunks.")
        self.index = index
        self.chunks = chunks

    @classmethod
    def load(
        cls,
        index_path: Path = config.INDEX_PATH,
        chunks_path: Path = config.CHUNKS_PATH,
        meta_path: Path = config.INDEX_META_PATH,
    ) -> "Retriever":
        """Load the prebuilt index, refusing one that no longer matches its chunks or model."""
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["chunks_sha256"] != file_sha256(chunks_path):
            raise RuntimeError("chunks.jsonl changed since the index was built; "
                               "run `python -m retrieval.index`.")
        if meta["embedding_model"] != config.EMBEDDING_MODEL:
            raise RuntimeError(f"Index was built with {meta['embedding_model']}, "
                               f"but config says {config.EMBEDDING_MODEL}; rebuild it.")
        return cls(faiss.read_index(str(index_path)), load_chunks(chunks_path))

    def with_chunks(self, new_chunks: list[dict]) -> "Retriever":
        """A new Retriever over these chunks plus `new_chunks` (e.g. an uploaded document).

        Only the new chunks are embedded; existing vectors are copied, and this one is unchanged.
        """
        index = faiss.IndexFlatIP(self.index.d)
        index.add(self.index.reconstruct_n(0, self.index.ntotal))
        if new_chunks:
            index.add(embed_passages([c["text"] for c in new_chunks]))
        return Retriever(index, self.chunks + new_chunks)

    def search(self, query: str, k: int = config.TOP_K, where: dict | None = None) -> list[dict]:
        """Return the top-k chunks for `query`, best first.

        `where` optionally restricts results to chunks whose metadata matches every key,
        e.g. {"company": "TCS", "year": "FY2026"}. Filtering happens after scoring, over all
        chunks, so a filter never silently returns fewer than k results when k matches exist.
        """
        n = self.index.ntotal if where else min(k, self.index.ntotal)
        scores, ids = self.index.search(embed_query(query), n)

        results = []
        for score, i in zip(scores[0], ids[0]):
            chunk = self.chunks[i]
            if where and any(chunk.get(key) != value for key, value in where.items()):
                continue
            results.append({**chunk, "score": float(score), "rank": len(results) + 1,
                            "source": format_source(chunk)})
            if len(results) == k:
                break
        return results


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """Default retriever over data/processed, loaded once per process."""
    return Retriever.load()


def retrieve(query: str, k: int = config.TOP_K, where: dict | None = None) -> list[dict]:
    """Top-k chunks for a query from the default index. See Retriever.search."""
    return get_retriever().search(query, k, where)


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What was HDFC Bank's net profit in FY2025?"
    print(f"Q: {question}\n")
    for r in retrieve(question):
        preview = " ".join(r["text"].split())[:160]
        print(f"[{r['rank']}] {r['score']:.3f}  {r['source']}\n    {preview}...\n")
