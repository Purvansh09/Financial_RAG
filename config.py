"""Central settings: model names, chunking, retrieval, and paths.

Every tunable lives here so experiments change one file and eval runs can log it.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MANIFEST_PATH = RAW_DIR / "manifest.csv"  # per-PDF metadata: company, year, doc_type
PAGES_PATH = PROCESSED_DIR / "pages.jsonl"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
INDEX_PATH = PROCESSED_DIR / "faiss.index"
INDEX_META_PATH = PROCESSED_DIR / "index_meta.json"  # what the index was built from
EVAL_DIR = ROOT / "eval"
EVAL_RESULTS_DIR = EVAL_DIR / "results"

# --- Embeddings ---
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# bge models retrieve better when queries (not passages) carry this prefix.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EMBED_BATCH_SIZE = 32

# --- Parsing ---
# Pages where more than this share of characters come from a broken font encoding
# (digits extracted as Greek/Latin-Extended glyphs) are flagged and not chunked.
GARBLED_PAGE_THRESHOLD = 0.02

# --- Chunking (measured in embedding-model tokens) ---
# bge-small accepts 512 tokens; 500 + [CLS]/[SEP] fits without truncation.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75
MIN_CHUNK_TOKENS = 20  # drop near-empty pages (covers, section dividers)

# --- Retrieval ---
TOP_K = 5

# --- Generation ---
LLM_MODEL = os.getenv("LLM_MODEL", "claude-opus-5-5")
# Opus 5.5 always thinks; effort sets how much. Its default is "medium"; set it explicitly so
# eval runs record what they used. (temperature is no longer accepted by the API, so answers
# can vary slightly between runs.)
LLM_EFFORT = os.getenv("LLM_EFFORT", "medium")
LLM_MAX_TOKENS = 16000  # ceiling, not a target: thinking tokens count toward it

# --- Evaluation ---
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-opus-5-5")
EVAL_KS = (1, 3, 5, 10)  # hit rate is reported at each of these cut-offs
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# --- App (a public deployment spends the owner's API key, so it is capped) ---
MAX_QUESTIONS_PER_SESSION = int(os.getenv("MAX_QUESTIONS_PER_SESSION", "20"))
APP_PASSWORD = os.getenv("APP_PASSWORD")  # if set, visitors must enter it before asking
MAX_UPLOAD_PAGES = 400  # embedding runs on CPU; ~400 pages takes about a minute
