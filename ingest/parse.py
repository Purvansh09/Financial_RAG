"""PDF -> page-level text records, written to data/processed/pages.jsonl.

One record per PDF page. `page` is the 1-based PDF page number (what a PDF viewer's page box
shows and what `file.pdf#page=N` opens), not the number printed on the page: most of these
PDFs carry no page labels, and printed numbering often skips covers and restarts per section.
"""

import csv
import json
import re
from pathlib import Path

import pymupdf

import config

# Characters these PDFs extract wrongly, mapped to plain equivalents. Kept narrow on purpose:
# full NFKC normalisation would also turn superscript footnote markers (¹) into digits and
# silently change numbers.
CHAR_FIXES = str.maketrans({
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "\xa0": " ", " ": " ", " ": " ", "‑": "-",
})

# Several reports use a font that maps the rupee glyph to a backtick ("` 1,68,302 crore",
# "Amount (in `)"). Treat a backtick as a quote mark only when a closing quote follows within
# a few words and no amount does ("` Ind AS’"); every other backtick becomes ₹.
RUPEE_BACKTICK = re.compile(
    r"`(?=\s*(?:[\d.]|in\b|crore|lakh|billion|million|mn\b|bn\b|'000|‘000))"
    r"|`(?![^`’'\n]{1,30}[’'])",
    re.IGNORECASE,
)

# HDFC and TCS fonts also extract ₹ as a capital letter: "H67,347.4 crore", "K2,51,150",
# "(J in crore)", "C in crore". Replace one only directly before a unit or an amount (a number
# followed by a unit, or one with Indian digit grouping). "H1 FY25", "H2H" and "H000052115"
# are left alone.
RUPEE_LETTER = re.compile(
    r"(?<![A-Za-z0-9])[CEHIJK](?= ?(?:(?:in )?(?:crore|lakh)\b"
    r"|\d[\d,]*(?:\.\d+)?\s*(?:crore|lakh|cr\b|bn\b|billion|mn\b|million|per\b)"
    r"|\d{1,3}(?:,\d{2})+,\d{3}\b))"
)

# Some PDFs space out letters ("B usi ness", "D I R E C T O R S"). PyMuPDF can be told not to
# add gap-based spaces, but on other PDFs that glues real words together, so it's chosen per page.
TIGHT_FLAGS = pymupdf.TEXTFLAGS_TEXT | pymupdf.TEXT_INHIBIT_SPACES
STRAY_LETTER = re.compile(r"(?<!\S)[B-HJ-Zb-z](?!\S)")  # single letters other than a / A / I
RUN_ON_WORD = re.compile(r"[A-Za-z]{22,}")

# Some PDFs use control characters as bullets or spacers (ICICI: \x07, TCS: \x03).
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

# Glyph ranges that show up when a font's encoding is broken (TCS FY26 prints digits as
# Greek letters, e.g. "26ϳ,021", and words as Latin-Extended, e.g. "ƌevenue").
GARBLED_CHARS = re.compile(r"[Ā-ɏͰ-Ͽༀ-࿿]")

# A number with a broken digit in it. One of these is enough to flag a page: a wrong figure
# reaching the LLM is worse than losing the page.
GARBLED_NUMBER = re.compile(r"[\d,.₹][Ͱ-Ͽ]|[Ͱ-Ͽ][\d,.]|[Ͱ-Ͽ]{2,}")


def clean_text(text: str) -> str:
    """Fix known extraction artefacts and tidy whitespace, keeping line breaks."""
    text = text.translate(CHAR_FIXES)
    text = CONTROL_CHARS.sub(" ", text)
    text = RUPEE_BACKTICK.sub("₹", text)
    text = RUPEE_LETTER.sub("₹", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def garbled_ratio(text: str) -> float:
    """Share of non-space characters that look like broken-encoding glyphs."""
    visible = re.sub(r"\s", "", text)
    if not visible:
        return 0.0
    return len(GARBLED_CHARS.findall(visible)) / len(visible)


def is_garbled(text: str) -> bool:
    """True if the page has a broken number or a high share of broken glyphs."""
    return bool(GARBLED_NUMBER.search(text)) or garbled_ratio(text) > config.GARBLED_PAGE_THRESHOLD


def spacing_errors(text: str) -> int:
    """Count signs of bad word spacing: stray single letters and run-on words."""
    return len(STRAY_LETTER.findall(text)) + len(RUN_ON_WORD.findall(text))


def extract_page_text(page: pymupdf.Page) -> str:
    """Page text from whichever extraction mode spaces words better (default wins ties)."""
    default = page.get_text()
    default_errors = spacing_errors(default)
    if default_errors == 0:
        return default
    tight = page.get_text(flags=TIGHT_FLAGS)
    return tight if spacing_errors(tight) < default_errors else default


def load_manifest(path: Path = config.MANIFEST_PATH) -> dict[str, dict]:
    """Read manifest.csv into {source_file: {company, year, doc_type, source_url}}."""
    with open(path, encoding="utf-8", newline="") as f:
        return {row["source_file"]: row for row in csv.DictReader(f)}


def parse_document(doc: pymupdf.Document, source_file: str, meta: dict) -> list[dict]:
    """Extract one cleaned record per page of an open PDF, carrying the document's metadata."""
    records = []
    for i, page in enumerate(doc):
        raw = extract_page_text(page)
        records.append({
            "source_file": source_file,
            "company": meta["company"],
            "year": meta["year"],
            "doc_type": meta["doc_type"],
            "page": i + 1,
            "text": clean_text(raw),
            "garbled": is_garbled(raw),
        })
    return records


def parse_pdf(pdf_path: Path, meta: dict) -> list[dict]:
    """Parse a PDF file on disk."""
    with pymupdf.open(pdf_path) as doc:
        return parse_document(doc, pdf_path.name, meta)


def parse_pdf_bytes(data: bytes, source_file: str, meta: dict) -> list[dict]:
    """Parse a PDF held in memory (e.g. a file uploaded in the app)."""
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        return parse_document(doc, source_file, meta)


def parse_all(raw_dir: Path = config.RAW_DIR, out_path: Path = config.PAGES_PATH) -> list[dict]:
    """Parse every PDF listed in the manifest and write pages.jsonl."""
    manifest = load_manifest(raw_dir / "manifest.csv")
    on_disk = {p.name for p in raw_dir.glob("*.pdf")}
    if missing := set(manifest) - on_disk:
        raise FileNotFoundError(f"In manifest but not in {raw_dir}: {sorted(missing)}")
    if unlisted := on_disk - set(manifest):
        raise ValueError(f"PDFs with no manifest row (metadata unknown): {sorted(unlisted)}")

    pages = []
    for name, meta in manifest.items():
        pages.extend(parse_pdf(raw_dir / name, meta))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:  # same bytes on every OS
        for rec in pages:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return pages


if __name__ == "__main__":
    pages = parse_all()
    garbled = [p for p in pages if p["garbled"]]
    print(f"Parsed {len(pages)} pages from {len({p['source_file'] for p in pages})} PDFs "
          f"-> {config.PAGES_PATH.relative_to(config.ROOT)}")
    print(f"Flagged {len(garbled)} garbled pages (excluded from chunking):")
    for name in sorted({p["source_file"] for p in garbled}):
        nums = [p["page"] for p in garbled if p["source_file"] == name]
        print(f"  {name}: {nums}")
