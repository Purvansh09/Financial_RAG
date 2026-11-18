"""Upload the app and its prebuilt index to a Hugging Face Space (Docker SDK).

Usage (after `hf auth login`):
    python -m deploy.deploy_space <hf-username>/<space-name>

Uploads code, the Dockerfile and the processed index only. Never uploads .env, the PDFs,
tests or pages.jsonl. The API key is set separately as a Space secret in the Space settings.
"""

import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

import config
from retrieval.index import file_sha256

UPLOAD = [
    "Dockerfile", "requirements.txt", "config.py", ".streamlit/config.toml",
    "app/*.py", "ingest/*.py", "retrieval/*.py", "generation/*.py",
    "data/processed/chunks.jsonl", "data/processed/faiss.index", "data/processed/index_meta.json",
]

# Spaces read their settings from YAML front matter in the repo's README.
SPACE_README = """\
---
title: Financial Document Q&A
emoji: 📄
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
short_description: Cited answers from Indian company annual reports and calls
---

Ask questions about annual reports, earnings call transcripts and investor presentations from
HDFC Bank, ICICI Bank, Infosys and TCS. Answers come only from retrieved passages and cite the
document and page. Built with bge-small embeddings, FAISS and Claude.
"""


def check_index_is_current() -> None:
    """Refuse to deploy an index that doesn't match the chunks it will be served with."""
    import json
    meta = json.loads(config.INDEX_META_PATH.read_text(encoding="utf-8"))
    if meta["chunks_sha256"] != file_sha256(config.CHUNKS_PATH):
        sys.exit("The index is stale; run `python -m retrieval.index` before deploying.")


def deploy(repo_id: str) -> str:
    """Create the Space if needed and upload the app files. Returns the Space URL."""
    check_index_is_current()
    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id, repo_type="space", folder_path=config.ROOT,
        allow_patterns=UPLOAD, commit_message="Deploy app and prebuilt index",
    )
    with tempfile.TemporaryDirectory() as tmp:
        readme = Path(tmp) / "README.md"
        readme.write_text(SPACE_README, encoding="utf-8")
        api.upload_file(path_or_fileobj=readme, path_in_repo="README.md",
                        repo_id=repo_id, repo_type="space", commit_message="Space README")
    return f"https://huggingface.co/spaces/{repo_id}"


if __name__ == "__main__":
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        sys.exit("Usage: python -m deploy.deploy_space <hf-username>/<space-name>")
    url = deploy(sys.argv[1])
    print(f"Uploaded. The Space builds in a few minutes: {url}")
    print("Then add ANTHROPIC_API_KEY under Settings -> Variables and secrets -> New secret.")
