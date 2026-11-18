"""Tests for generation: prompt building and citation checks. Uses a fake client, no API calls."""

from types import SimpleNamespace

import pytest

import config
from generation.answer import generate, parse_citations, uncited_figures
from generation.prompts import NOT_FOUND, build_user_message

CHUNKS = [
    {"chunk_id": "hdfc.pdf:p12:c0", "source_file": "hdfc.pdf", "company": "HDFC Bank",
     "year": "FY2025", "doc_type": "annual_report", "page": 12,
     "source": "HDFC Bank FY2025 annual_report (hdfc.pdf), p. 12",
     "text": "Gross Non-Performing Assets stood at 1.33 per cent."},
    {"chunk_id": "hdfc.pdf:p38:c0", "source_file": "hdfc.pdf", "company": "HDFC Bank",
     "year": "FY2025", "doc_type": "annual_report", "page": 38,
     "source": "HDFC Bank FY2025 annual_report (hdfc.pdf), p. 38",
     "text": "Profit After Tax grew by 10.7 per cent."},
]


class FakeClient:
    """Stands in for anthropic.Anthropic; records the request and returns a canned reply."""

    def __init__(self, text: str, stop_reason: str = "end_turn"):
        self.request = None
        reply = SimpleNamespace(
            content=[SimpleNamespace(type="thinking", thinking=""),
                     SimpleNamespace(type="text", text=text)],
            stop_reason=stop_reason, model=config.LLM_MODEL,
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        )

        def create(**kwargs):
            self.request = kwargs
            return reply

        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))


def test_sources_are_numbered_with_metadata():
    msg = build_user_message("What is the GNPA?", CHUNKS)
    assert '<source id="1" company="HDFC Bank" year="FY2025" doc_type="annual_report" page="12">' in msg
    assert '<source id="2"' in msg and msg.endswith("Question: What is the GNPA?")


@pytest.mark.parametrize("text, valid, invalid", [
    ("GNPA was 1.33% [1].", [1], []),
    ("A [2]. B [1][2]. C [1, 2].", [2, 1], []),
    ("Made up [3].", [], [3]),
    ("No citations here.", [], []),
])
def test_parse_citations(text, valid, invalid):
    assert parse_citations(text, n_sources=2) == (valid, invalid)


def test_uncited_figures():
    text = "GNPA was 1.33% [1]. PAT grew 10.7%.\n1. The bank is large. [2]"
    assert uncited_figures(text) == ["PAT grew 10.7%."]


def test_table_followed_by_citation_is_cited():
    cited = "Revenue rose [2].\n\n| Item | FY26 |\n|---|---|\n| Services | 1,47,806 |\n\n[2]"
    uncited = "Revenue rose [2].\n\n| Item | FY26 |\n|---|---|\n| Services | 1,47,806 |"
    assert uncited_figures(cited) == []
    assert len(uncited_figures(uncited)) == 1


@pytest.mark.parametrize("sentence, flagged", [
    ("The sources do not state the ratio at March 31, 2025.", False),  # a date, not a figure
    ("This covers FY2026 and Q4 FY25.", False),
    ("Deposits were ₹27,14,715 crore.", True),
    ("Margins rose 20 bps.", True),
    ("Headcount was 5,84,519.", True),
])
def test_dates_are_not_figures(sentence, flagged):
    assert bool(uncited_figures(sentence)) is flagged


def test_generate_maps_citations_to_pages():
    client = FakeClient("Gross NPA was 1.33 per cent [1], and PAT grew 10.7 per cent [2].")
    result = generate("GNPA and PAT growth?", CHUNKS, client)
    assert [(c["n"], c["page"]) for c in result["citations"]] == [(1, 12), (2, 38)]
    assert result["warnings"] == [] and not result["not_found"]
    assert result["answer"].startswith("Gross NPA")  # thinking block is not part of the answer


def test_request_settings():
    client = FakeClient("x [1]")
    generate("q", CHUNKS, client)
    req = client.request
    assert req["model"] == config.LLM_MODEL
    assert req["output_config"] == {"effort": config.LLM_EFFORT}
    assert "temperature" not in req  # rejected by the current API
    assert "1.33 per cent" in req["messages"][0]["content"]


def test_not_found_answer():
    result = generate("What is TCS's market cap?", CHUNKS, FakeClient(NOT_FOUND))
    assert result["not_found"] and result["citations"] == [] and result["warnings"] == []


@pytest.mark.parametrize("text, expected", [
    ("GNPA was 1.33 per cent.", "cites no sources"),
    ("GNPA was 1.33 per cent [5].", "not provided: [5]"),
    ("GNPA was 1.33 per cent [1]. PAT grew 10.7 per cent.", "figures without a citation"),
])
def test_ungrounded_answers_are_flagged(text, expected):
    result = generate("q", CHUNKS, FakeClient(text))
    assert any(expected in w for w in result["warnings"])


def test_refusal_and_truncation():
    refused = generate("q", CHUNKS, FakeClient("partial", stop_reason="refusal"))
    assert refused["answer"] == "" and "declined" in refused["warnings"][0]
    cut = generate("q", CHUNKS, FakeClient("GNPA was [1]", stop_reason="max_tokens"))
    assert "cut off" in cut["warnings"][0]
