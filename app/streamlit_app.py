"""Streamlit UI: ask questions about the financial documents and get answers with page citations.

Run with: streamlit run app/streamlit_app.py
"""

import sys
from collections import Counter
from pathlib import Path

# `streamlit run` puts app/ on the import path, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic
import streamlit as st

import config
from generation.answer import generate
from ingest.chunk import chunk_pages, get_tokenizer
from ingest.parse import parse_pdf_bytes
from retrieval.embed import get_model
from retrieval.retrieve import Retriever

DOC_TYPES = ["annual_report", "earnings_call", "investor_presentation", "sebi_filing", "other"]
EXAMPLES = [
    "What final dividend per share did Infosys recommend for FY2025?",
    "What revenue growth guidance did Infosys give for FY26 on its Q4 FY25 call?",
    "How many employees did TCS have at the end of FY2026?",
]


@st.cache_resource(show_spinner="Loading the document index...")
def base_retriever() -> Retriever:
    """The prebuilt index over the bundled documents; shared by all sessions."""
    get_model()  # load the embedding model now, not on the first question
    return Retriever.load()


@st.cache_resource
def tokenizer():
    return get_tokenizer()


def current_retriever() -> Retriever:
    """This session's retriever: the bundled documents plus anything the visitor uploaded."""
    return st.session_state.get("retriever") or base_retriever()


def escape_markdown(text: str) -> str:
    """Stop Streamlit reading "$19.3 bn ... $40.7 bn" as a LaTeX formula."""
    return text.replace("$", r"\$")


def document_table(retriever: Retriever) -> list[dict]:
    """One row per document in the index, with how many chunks it contributes."""
    counts = Counter((c["company"], c["year"], c["doc_type"], c["source_file"])
                     for c in retriever.chunks)
    return [{"Company": co, "Year": yr, "Type": dt, "File": f, "Chunks": n}
            for (co, yr, dt, f), n in sorted(counts.items())]


def sidebar(retriever: Retriever) -> dict | None:
    """Documents list, search filters and upload form. Returns the metadata filter."""
    with st.sidebar:
        st.header("Documents")
        st.dataframe(document_table(retriever), hide_index=True, width="stretch")

        st.header("Search only")
        companies = sorted({c["company"] for c in retriever.chunks})
        years = sorted({c["year"] for c in retriever.chunks})
        company = st.selectbox("Company", ["All"] + companies)
        year = st.selectbox("Year", ["All"] + years)
        where = {k: v for k, v in (("company", company), ("year", year)) if v != "All"}

        st.header("Add a PDF")
        upload_form(retriever)
    return where or None


def upload_form(retriever: Retriever) -> None:
    """Parse, chunk and embed an uploaded PDF for this session only."""
    with st.form("upload", clear_on_submit=True):
        file = st.file_uploader("PDF", type="pdf")
        company = st.text_input("Company")
        year = st.text_input("Year", placeholder="FY2025")
        doc_type = st.selectbox("Type", DOC_TYPES)
        submitted = st.form_submit_button("Add to this session")
    if not submitted:
        return
    if not (file and company.strip() and year.strip()):
        st.error("Choose a PDF and fill in company and year; citations depend on them.")
        return
    if any(c["source_file"] == file.name for c in retriever.chunks):
        st.error(f"A document named {file.name} is already loaded.")
        return

    meta = {"company": company.strip(), "year": year.strip(), "doc_type": doc_type}
    with st.spinner(f"Reading and indexing {file.name}..."):
        try:
            pages = parse_pdf_bytes(file.getvalue(), file.name, meta)
        except Exception:
            st.error("That file could not be read as a PDF.")
            return
        if len(pages) > config.MAX_UPLOAD_PAGES:
            st.error(f"{len(pages)} pages is over the {config.MAX_UPLOAD_PAGES}-page limit.")
            return
        chunks = chunk_pages(pages, tokenizer())
        st.session_state["retriever"] = retriever.with_chunks(chunks)

    garbled = sum(p["garbled"] for p in pages)
    st.success(f"Added {file.name}: {len(pages)} pages, {len(chunks)} chunks. "
               "It stays only for this session.")
    if garbled:
        st.warning(f"{garbled} page(s) have broken text encoding and were skipped.")
    if not chunks:
        st.warning("No text was found. Scanned PDFs need OCR, which this app doesn't do.")


def password_ok() -> bool:
    """Gate the app behind APP_PASSWORD when one is configured."""
    if not config.APP_PASSWORD or st.session_state.get("authorised"):
        return True
    entered = st.text_input("Password", type="password")
    if entered and entered == config.APP_PASSWORD:
        st.session_state["authorised"] = True
        st.rerun()
    elif entered:
        st.error("Wrong password.")
    return False


def show_result(result: dict) -> None:
    """Answer, warnings, cited passages, and everything that was retrieved."""
    if result["not_found"]:
        st.info("Not found in the documents. Try rephrasing, or narrow the search in the sidebar.")
    elif result["answer"]:
        st.markdown(escape_markdown(result["answer"]))

    for warning in result["warnings"]:
        st.warning(warning)

    if result["citations"]:
        st.subheader("Sources")
        by_id = {c["chunk_id"]: c for c in result["sources"]}
        for cite in sorted(result["citations"], key=lambda c: c["n"]):
            label = (f"[{cite['n']}] {cite['company']} {cite['year']} "
                     f"{cite['doc_type'].replace('_', ' ')}, page {cite['page']}")
            with st.expander(label):
                st.caption(cite["source_file"])
                st.text(by_id[cite["chunk_id"]]["text"])

    with st.expander(f"All {len(result['sources'])} retrieved passages"):
        for n, s in enumerate(result["sources"], start=1):
            st.markdown(f"**[{n}]** {s['source']} (similarity {s['score']:.3f})")
    u = result["usage"]
    st.caption(f"{result['model']} · {u['input_tokens']:,} input / {u['output_tokens']:,} output tokens")


def main() -> None:
    st.set_page_config(page_title="Financial Document Q&A", page_icon="📄")
    st.title("Financial Document Q&A")
    st.caption("Answers come only from the documents listed in the sidebar, with the page each "
               "fact came from. If the documents don't say, it tells you.")

    if not password_ok():
        return
    retriever = current_retriever()
    where = sidebar(retriever)

    with st.form("ask"):
        question = st.text_input("Question", placeholder=EXAMPLES[0])
        asked = st.form_submit_button("Ask")
    st.caption("Try: " + " · ".join(EXAMPLES[1:]))

    if asked and question.strip():
        used = st.session_state.get("questions_asked", 0)
        if used >= config.MAX_QUESTIONS_PER_SESSION:
            st.error(f"This demo allows {config.MAX_QUESTIONS_PER_SESSION} questions per session.")
        elif not config.ANTHROPIC_API_KEY:
            st.error("No API key is configured. Set ANTHROPIC_API_KEY in .env (or the Space's secrets).")
        else:
            with st.spinner("Searching the documents and writing an answer..."):
                try:
                    chunks = retriever.search(question.strip(), config.TOP_K, where)
                    st.session_state["result"] = generate(question.strip(), chunks)
                    st.session_state["questions_asked"] = used + 1
                except anthropic.RateLimitError:
                    st.error("The model is busy right now. Please try again in a minute.")
                except anthropic.APIError as e:
                    st.error(f"The model request failed: {e.__class__.__name__}.")

    if result := st.session_state.get("result"):
        st.divider()
        st.markdown(f"**Q: {escape_markdown(result['question'])}**")
        show_result(result)


main()
