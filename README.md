# Financial Document RAG

Ask plain-English questions about Indian financial documents (annual reports, SEBI filings, earnings call transcripts). Answers come only from retrieved passages and cite the source document and page.

The bundled corpus is 11 filings (2,106 pages) from HDFC Bank, ICICI Bank, Infosys and TCS, indexed as 3,935 chunks. Users can also upload their own PDF for a session.

## Results (baseline)

Evaluated on 38 questions with gold answers and pages (facts, table values, earnings calls, FY2025 vs FY2026 comparisons, and 6 unanswerable questions). Details in [eval/README.md](eval/README.md).

| Metric | Result |
|---|---|
| Retrieval Hit@1 / @3 / @5 / @10 | 15.6% / 40.6% / 53.1% / 65.6% |
| MRR@10 | 0.31 |
| Answers correct | 53.1% (56.2% incl. partially correct) |
| Faithfulness (claims supported by retrieved passages) | 95.5% |
| Unanswerable questions declined | 100% (6/6) |

Retrieval is the bottleneck: when a correct page was in the top 5, 16 of 17 answers were correct; when it wasn't, 14 of 15 answers said the information wasn't found rather than guessing. Next: hybrid search (BM25 + vectors), a reranker, and table extraction, each measured against this baseline.

The evaluation questions were written with LLM assistance from randomly sampled pages and checked automatically; they have not yet been reviewed by hand.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # then add your ANTHROPIC_API_KEY
```

## Data

The prebuilt index in `data/processed/` is enough to run the app. To rebuild it, source PDFs go in `data/raw/`. Each one needs a row in `data/raw/manifest.csv` (`source_file, company, year, doc_type, source_url`); this is where chunk metadata comes from. PDFs themselves are not committed; use the URLs in the manifest to re-download them.

## Run the app

```bash
streamlit run app/streamlit_app.py
```

Ask a question, optionally narrow the search by company or year, and expand each cited source to read the exact passage and page it came from. A PDF uploaded in the sidebar is parsed, chunked and embedded for that session only.

## Deploy

The repo ships the prebuilt index, so a deployment doesn't need the PDFs or a rebuild.

**Streamlit Community Cloud:** create an app from this repo with main file `app/streamlit_app.py`, then add `ANTHROPIC_API_KEY` under the app's Secrets. `requirements.txt` pins CPU-only torch.

**Any Docker host:** the `Dockerfile` builds the same app (`deploy/deploy_space.py` uploads it to a Hugging Face Space, which now needs a PRO plan for Docker).

A public deployment spends the owner's API key, so the app caps questions per session (`MAX_QUESTIONS_PER_SESSION`) and can require a password (`APP_PASSWORD` secret).

## Evaluation

```bash
python -m eval.check_questions                              # validate the question set
python -m eval.run_eval --retrieval-only --label baseline   # free: retrieval metrics only
python -m eval.run_eval --label baseline                    # + answers judged by Claude (~$1.40)
```

## Pipeline

```bash
python -m ingest.parse    # PDFs -> data/processed/pages.jsonl (one record per page)
python -m ingest.chunk    # pages -> data/processed/chunks.jsonl (~500-token chunks)
python -m retrieval.index # embed chunks -> data/processed/faiss.index (~4 min on CPU)
python -m retrieval.retrieve "What was HDFC Bank's gross NPA ratio?"   # top-5 chunks
python -m generation.answer "What was HDFC Bank's gross NPA ratio?"    # cited answer (needs API key)
pytest
```

Parsing fixes known extraction problems: ligatures, letter-spaced text ("B usi ness"), and a rupee sign that several reports' fonts extract as a backtick or a capital letter ("H67,347 crore"). It also flags pages whose font encoding is broken (digits extracted as Greek letters); those pages are left out of chunking so wrong figures can't reach the answer step. Chunks never cross a page boundary, so each one cites exactly one PDF page.

Retrieval embeds chunks with `bge-small-en-v1.5` and searches them exactly (FAISS `IndexFlatIP`, cosine similarity). `retrieve(query, k, where=...)` can filter by metadata, e.g. `where={"company": "TCS"}`. Retrieval refuses to run if `chunks.jsonl` has changed since the index was built.

Generation passes the retrieved chunks to Claude as numbered sources and requires a `[n]` citation on every factual sentence, or the exact reply "Not found in the documents." Code maps each number back to its document and page, so a citation can only point at a passage that was actually retrieved. Answers with no citations, citations to non-existent sources, or figures without a citation are flagged in `warnings`.
