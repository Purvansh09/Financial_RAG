"""Page text -> overlapping chunks with metadata, written to data/processed/chunks.jsonl.

Chunks never cross a page boundary, so every chunk has exactly one page to cite. The cost is
that a sentence running over a page break is split; for page-accurate citations that's worth it.

Sizes are counted with the embedding model's own tokenizer, because that model's 512-token
limit is the one that matters. Chunk text is sliced from the original page text using the
tokenizer's character offsets, so casing and punctuation are preserved (bge's tokenizer
lowercases, so decoding tokens back to text would not be).
"""

import json
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase

import config


def get_tokenizer() -> PreTrainedTokenizerBase:
    """Load the tokenizer that belongs to the embedding model."""
    return AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)


def _word_start(offsets: list[tuple[int, int]], i: int, floor: int) -> int:
    """Move token index i back to the first sub-word token of its word, not below floor."""
    while i > floor and offsets[i][0] == offsets[i - 1][1]:
        i -= 1
    return i


def chunk_text(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
    min_tokens: int = config.MIN_CHUNK_TOKENS,
) -> list[str]:
    """Split text into windows of at most `size` tokens that share `overlap` tokens.

    Window edges are moved back to the start of a word so no word is cut in half.
    Text shorter than `min_tokens` yields no chunks.
    """
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                        verbose=False)["offset_mapping"]
    n = len(offsets)
    if n < min_tokens:
        return []

    chunks, start = [], 0
    while True:
        end = min(start + size, n)
        if end < n:
            # Cut before the word that token `end` belongs to. If that leaves less than half
            # a window (a very long unbroken string), cut mid-word instead.
            snapped = _word_start(offsets, end, start + 1)
            if snapped - start >= size // 2:
                end = snapped
        chunks.append(text[offsets[start][0]:offsets[end - 1][1]])
        if end == n:
            return chunks
        start = _word_start(offsets, max(end - overlap, start + 1), start + 1)


def chunk_pages(pages: list[dict], tokenizer: PreTrainedTokenizerBase) -> list[dict]:
    """Chunk every usable page, copying its metadata onto each chunk."""
    chunks = []
    for page in pages:
        if page["garbled"]:
            continue
        for i, text in enumerate(chunk_text(page["text"], tokenizer)):
            chunks.append({
                "chunk_id": f"{page['source_file']}:p{page['page']}:c{i}",
                "source_file": page["source_file"],
                "company": page["company"],
                "year": page["year"],
                "doc_type": page["doc_type"],
                "page": page["page"],
                "text": text,
                "n_tokens": len(tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]),
            })
    return chunks


def chunk_all(pages_path: Path = config.PAGES_PATH, out_path: Path = config.CHUNKS_PATH) -> list[dict]:
    """Read pages.jsonl, chunk it, and write chunks.jsonl."""
    with open(pages_path, encoding="utf-8") as f:
        pages = [json.loads(line) for line in f]
    chunks = chunk_pages(pages, get_tokenizer())
    # "\n" endings on every OS: index_meta.json fingerprints this file's exact bytes.
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for rec in chunks:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return chunks


if __name__ == "__main__":
    chunks = chunk_all()
    sizes = sorted(c["n_tokens"] for c in chunks)
    print(f"Wrote {len(chunks)} chunks -> {config.CHUNKS_PATH.relative_to(config.ROOT)}")
    print(f"Tokens per chunk: min {sizes[0]}, median {sizes[len(sizes) // 2]}, max {sizes[-1]}")
    for name in sorted({c["source_file"] for c in chunks}):
        print(f"  {name}: {sum(c['source_file'] == name for c in chunks)}")
