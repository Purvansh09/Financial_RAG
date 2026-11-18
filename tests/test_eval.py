"""Tests for the eval harness: ranking, metrics, question validation. No API calls."""

import json
from types import SimpleNamespace

from eval.dataset import validate
from eval.run_eval import figures_grounded, first_gold_rank, score_question, summarize


def chunk(f, p, text="x"):
    return {"chunk_id": f"{f}:p{p}:c0", "source_file": f, "page": p, "company": "Acme",
            "year": "FY2025", "doc_type": "annual_report", "text": text, "score": 0.5,
            "source": f"Acme FY2025 annual_report ({f}), p. {p}"}


def test_first_gold_rank():
    retrieved = [chunk("a.pdf", 1), chunk("a.pdf", 7), chunk("b.pdf", 3)]
    assert first_gold_rank(retrieved, {("a.pdf", 7), ("b.pdf", 3)}) == 2
    assert first_gold_rank(retrieved, {("a.pdf", 99)}) is None


def test_figures_grounded():
    cited = [chunk("a.pdf", 1, "Revenue was 1,62,990 crore, up 6.1%.")]
    assert figures_grounded("Revenue was ₹162,990 crore [1], up 6.1% [1].", cited)
    assert not figures_grounded("Revenue was ₹1,70,000 crore [1].", cited)
    assert figures_grounded("In FY2025 and 2024 it changed [1].", cited)  # years ignored


def row(qtype, rank, correctness="correct", faithful=True, not_found=False):
    return {"type": qtype, "gold_rank": rank, "correctness": correctness, "faithful": faithful,
            "not_found": not_found, "figures_grounded": True, "cites_gold": rank is not None,
            "warnings": [], "unsupported_claims": []}


def test_summarize_metrics():
    rows = [row("fact", 1), row("fact", 4, "partially_correct"), row("fact", None, "incorrect", not_found=True),
            row("unanswerable", None, "correct", not_found=True)]
    s = summarize(rows)
    assert s["n_answerable"] == 3
    assert (s["hit@1"], s["hit@3"], s["hit@5"]) == (33.3, 33.3, 66.7)
    assert s["mrr@10"] == round((1 + 0.25) / 3, 3)
    assert (s["correct"], s["correct_or_partial"]) == (33.3, 66.7)
    assert s["unanswerable_declined"] == 100.0
    assert s["faithful"] == 100.0  # not-found answers are excluded


def test_validate_catches_mistakes():
    pages = {("a.pdf", 1): {"text": "Revenue 1,62,990", "garbled": False},
             ("a.pdf", 2): {"text": "broken", "garbled": True}}
    questions = [
        {"id": "q1", "type": "fact", "question": "?", "answer": "₹1,62,990", "gold_pages": [{"source_file": "a.pdf", "page": 1}]},
        {"id": "q2", "type": "fact", "question": "?", "answer": "₹5,000", "gold_pages": [{"source_file": "a.pdf", "page": 1}]},
        {"id": "q3", "type": "fact", "question": "?", "answer": "x", "gold_pages": [{"source_file": "a.pdf", "page": 9}]},
        {"id": "q4", "type": "table", "question": "?", "answer": "x", "gold_pages": [{"source_file": "a.pdf", "page": 2}]},
        {"id": "q4", "type": "guess", "question": "?", "answer": "", "gold_pages": []},
        {"id": "q6", "type": "unanswerable", "question": "?", "answer": "", "gold_pages": []},
    ]
    errors, warnings = validate(questions, pages)
    joined = " ".join(errors + warnings)
    assert "q1" not in joined and "q6" not in joined
    assert "q2: answer figure(s) ['5,000']" in joined
    assert "q3: a.pdf has pages 1-2, not 9" in joined
    assert "q4: a.pdf p.2 has broken encoding" in joined
    assert "q4: duplicate id" in joined and "type 'guess'" in joined


class FakeClient:
    """Answers through beta.messages (generation) and grades through messages (judge)."""

    def __init__(self, answer, verdict):
        def reply(text):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn",
                                   model="fake", usage=SimpleNamespace(input_tokens=10, output_tokens=5))
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: reply(answer)))
        self.messages = SimpleNamespace(create=lambda **kw: reply(json.dumps(verdict)))


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, k):
        return self.chunks[:k]


def test_score_question_end_to_end():
    retriever = FakeRetriever([chunk("a.pdf", 5, "Dividend of 22 per share."), chunk("a.pdf", 36, "Final dividend 22")])
    q = {"id": "q1", "type": "fact", "question": "Final dividend?", "answer": "₹22",
         "gold_pages": [{"source_file": "a.pdf", "page": 36}]}
    client = FakeClient("The final dividend was ₹22 per share [2].",
                        {"correctness": "correct", "unsupported_claims": [], "reason": "ok"})
    r = score_question(q, retriever, client, retrieval_only=False)
    assert r["gold_rank"] == 2 and r["cites_gold"] and r["figures_grounded"]
    assert r["correctness"] == "correct" and r["faithful"]


def test_exact_not_found_skips_judge():
    retriever = FakeRetriever([chunk("a.pdf", 5)])
    q = {"id": "u1", "type": "unanswerable", "question": "Reliance profit?", "answer": "", "gold_pages": []}
    client = FakeClient("Not found in the documents.", verdict=None)  # judge would fail if called
    client.messages = None
    r = score_question(q, retriever, client, retrieval_only=False)
    assert r["correctness"] == "correct" and r["gold_rank"] is None
