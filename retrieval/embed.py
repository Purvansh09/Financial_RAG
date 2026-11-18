"""Text -> embedding vectors with the bge model.

Vectors are L2-normalised, so an inner product between two of them is their cosine similarity.
Queries and passages are embedded differently: bge was trained with an instruction prefix on
the query side only.
"""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

import config


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the embedding model once per process."""
    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed_passages(texts: list[str], show_progress: bool = False) -> np.ndarray:
    """Embed document chunks. Returns a float32 array of shape (len(texts), dim)."""
    return get_model().encode(
        texts,
        batch_size=config.EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    ).astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed one search query (with bge's query prefix). Returns shape (1, dim)."""
    return embed_passages([config.QUERY_PREFIX + query])
