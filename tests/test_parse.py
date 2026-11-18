"""Tests for ingest.parse: text cleanup, garbled-page detection, and page records."""

import pymupdf
import pytest

from ingest.parse import clean_text, is_garbled, parse_all, parse_pdf_bytes, spacing_errors


@pytest.mark.parametrize("raw, expected", [
    ("reached ` 1,68,302.4 crore", "reached ₹ 1,68,302.4 crore"),
    ("cash flow of `34,549 cr", "cash flow of ₹34,549 cr"),
    ("Amount (in ` Cr)", "Amount (in ₹ Cr)"),
    ("Basic (`) 67.01", "Basic (₹) 67.01"),
    ("referred to as ` Ind AS’) prescribed", "referred to as ` Ind AS’) prescribed"),
])
def test_rupee_backtick(raw, expected):
    assert clean_text(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("PAT of H67,347.4 crore", "PAT of ₹67,347.4 crore"),
    ("K26,19,609 crore Advances", "₹26,19,609 crore Advances"),
    ("dividend of H22.0 per share", "dividend of ₹22.0 per share"),
    ("salary of K2,51,150 in FY25", "salary of ₹2,51,150 in FY25"),
    ("Dividend amount (J in crore)", "Dividend amount (₹ in crore)"),
    ("Revenue Trend (H crore)", "Revenue Trend (₹ crore)"),
    ("AUM of I3,36,282 crore", "AUM of ₹3,36,282 crore"),
    ("I think margins will improve", "I think margins will improve"),
    ("growth in H1 for FY25", "growth in H1 for FY25"),
    ("H2H payments", "H2H payments"),
    ("CIN H000052115", "CIN H000052115"),
])
def test_rupee_letter(raw, expected):
    assert clean_text(raw) == expected


def test_spacing_errors_counts_stray_letters_and_run_ons():
    assert spacing_errors("B usi ness Responsi b i l i ty") == 5  # B, b, i, l, i
    assert spacing_errors("Certainstatementsareincludedinthisrelease") == 1
    assert spacing_errors("A normal sentence about a bank and I") == 0


def test_ligatures_and_spaces():
    assert clean_text("proﬁt   after\xa0tax") == "profit after tax"


def test_superscripts_are_not_turned_into_digits():
    # NFKC would make this "101", silently changing the number.
    assert clean_text("10¹") == "10¹"


def test_control_chars_and_blank_lines():
    assert clean_text("a\x07b\n\n\n\nc  \n") == "a b\n\nc"


@pytest.mark.parametrize("text, expected", [
    ("Revenue of ₹ 26ϳ,021 crore", True),     # broken digit inside a number
    ("₹ 66,ϴ3ϴ Đƌoƌe", True),
    ("Revenue of ₹ 2,67,021 crore", False),
    # One stray symbol on an otherwise normal page, numbers intact.
    ("Scope 1 emissions (tCOφ) fell this year. " + "Net profit was ₹ 2,67,021 crore. " * 20, False),
])
def test_is_garbled(text, expected):
    assert is_garbled(text) is expected


def _make_pdf(path, pages):
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_text((72, 72), text)
    doc.save(path)


def test_parse_all_keeps_metadata_and_page_numbers(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _make_pdf(raw / "acme.pdf", ["first page", "second page"])
    (raw / "manifest.csv").write_text(
        "source_file,company,year,doc_type,source_url\nacme.pdf,Acme,FY2025,annual_report,\n",
        encoding="utf-8",
    )
    pages = parse_all(raw, tmp_path / "pages.jsonl")

    assert [p["page"] for p in pages] == [1, 2]
    assert pages[1]["text"] == "second page"
    assert pages[0]["company"] == "Acme" and pages[0]["year"] == "FY2025"
    assert pages[0]["doc_type"] == "annual_report" and pages[0]["source_file"] == "acme.pdf"
    assert (tmp_path / "pages.jsonl").exists()


def test_parse_pdf_bytes(tmp_path):
    _make_pdf(tmp_path / "up.pdf", ["uploaded page one", "uploaded page two"])
    meta = {"company": "Acme", "year": "FY2026", "doc_type": "earnings_call"}
    pages = parse_pdf_bytes((tmp_path / "up.pdf").read_bytes(), "up.pdf", meta)
    assert [(p["page"], p["text"], p["source_file"]) for p in pages] == [
        (1, "uploaded page one", "up.pdf"), (2, "uploaded page two", "up.pdf")]


def test_parse_all_rejects_pdf_without_manifest_row(tmp_path):
    _make_pdf(tmp_path / "orphan.pdf", ["x"])
    (tmp_path / "manifest.csv").write_text("source_file,company,year,doc_type,source_url\n",
                                          encoding="utf-8")
    with pytest.raises(ValueError, match="orphan.pdf"):
        parse_all(tmp_path, tmp_path / "pages.jsonl")
