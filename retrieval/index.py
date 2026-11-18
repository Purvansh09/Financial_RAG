"""chunks.jsonl -> FAISS index, written to data/processed/faiss.index.

Uses IndexFlatIP: exact search over all vectors (inner product = cosine, since vectors are
normalised). With ~4k chunks an exact search takes milliseconds, and it keeps approximation
error out of the eval numbers. Row i of the index is line i of chunks.jsonl; that ordering
is the only link between a vector and its text and metadata.

index_meta.json records what the index was built from, so retrieval can refuse to run
against an index that no longer matches chunks.jsonl.
"""

import hashlib
import json
from pathlib import Path

import faiss

import config
from retrieval.embed import embed_passages


def file_sha256(path: Path) -> str:
    """Fingerprint a file so a stale index can be detected."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_chunks(path: Path = config.CHUNKS_PATH) -> list[dict]:
    """Read chunks.jsonl in order (order = index row ids)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_index(
    chunks_path: Path = config.CHUNKS_PATH,
    index_path: Path = config.INDEX_PATH,
    meta_path: Path = config.INDEX_META_PATH,
    show_progress: bool = False,
) -> faiss.Index:
    """Embed every chunk, build an exact inner-product index, and save it with its metadata."""
    chunks = load_chunks(chunks_path)
    vectors = embed_passages([c["text"] for c in chunks], show_progress=show_progress)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(index_path))

    meta = {
        "embedding_model": config.EMBEDDING_MODEL,
        "dim": int(vectors.shape[1]),
        "n_chunks": len(chunks),
        "chunks_sha256": file_sha256(chunks_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return index


if __name__ == "__main__":
    index = build_index(show_progress=True)
    print(f"Indexed {index.ntotal} chunks ({index.d}-dim) -> "
          f"{config.INDEX_PATH.relative_to(config.ROOT)}")
