# Evaluation

**Provenance of the current `questions.jsonl` (38 questions):** written by Claude, not by hand.
Pages were drawn at random (seed 2026) from each PDF, without looking at retrieval results;
pages with nothing answerable (legal boilerplate, e-voting instructions) were skipped. Each
answer was located on every page that contains it with `find_page`, unanswerable questions were
checked to have no answer in any document, and the set passes `check_questions`. It has not yet
been reviewed by a person; spot-checking some questions against the PDFs is recommended.

## Writing the questions

Write ~40 questions by hand, from the PDFs themselves, into `eval/questions.jsonl`: one JSON
object per line. `questions.example.jsonl` shows the format; its rows are never scored.

```json
{"id": "q01", "type": "fact", "question": "...", "answer": "...", "gold_pages": [{"source_file": "infosys_fy25_annual_report.pdf", "page": 36}], "notes": ""}
```

| Field | What to put |
|---|---|
| `id` | Unique, e.g. `q01` |
| `type` | `fact`, `table`, `earnings_call`, `cross_year` or `unanswerable` |
| `question` | Phrased the way a user would ask it |
| `answer` | The correct answer, short, with units (`₹1,62,990 crore`, `35%`). Empty for `unanswerable` |
| `gold_pages` | Every page that answers it. Empty list for `unanswerable` |
| `notes` | Optional: anything odd (standalone vs consolidated, which table) |

**`page` is the PDF page number** (the number in your PDF viewer's page box, starting at 1),
not the number printed on the page. They usually differ.

Suggested mix:

| Type | Count | Examples |
|---|---|---|
| `fact` | ~18 | A figure or statement from a report, spread over all four companies |
| `table` | ~6 | A value you have to read from a table or chart |
| `earnings_call` | ~6 | What management said on a call (guidance, reasons, outlook) |
| `cross_year` | ~4 | Infosys FY2025 vs FY2026 (answer needs both reports, or the right one) |
| `unanswerable` | ~6 | Plausible but not in the documents (other companies, future targets, things never disclosed) |

Tips:
- If a fact appears on several pages (highlights page, MD&A, financial statements), list all of
  them in `gold_pages`; retrieval counts as a hit if it finds any one.
- Avoid copying sentences from the PDF into the question. Real users paraphrase, and copied
  wording makes retrieval look better than it is.
- Make unanswerable questions tempting: close to real content, so a careless system would guess.

## Helpers

```bash
python -m eval.find_page "1,62,990"            # which PDF page(s) contain this text?
python -m eval.find_page "CASA" --file hdfcbank
python -m eval.check_questions                 # validate questions.jsonl before running
```

`check_questions` catches wrong file names, page numbers past the end of a PDF, pages the index
excludes (broken font encoding: retrieval can never find them), and answer figures that don't
appear on any gold page (usually a page-number typo).

## Running

```bash
python -m eval.run_eval --retrieval-only --label baseline   # free, local
python -m eval.run_eval --label baseline                    # full: answers + judge (~$0.06/question)
```

Each run writes `eval/results/<timestamp>_<label>.json` (every question, every score) and adds
one row to `eval/results/summary.csv`, so each change gets a before/after row.

## Metrics

| Metric | Meaning |
|---|---|
| Hit@k | Share of answerable questions where a gold page is in the top k retrieved chunks |
| MRR@10 | Mean of 1/rank of the first gold page (0 if not in the top 10) |
| Answer correctness | Judge compares the answer with your gold answer: correct / partially correct / incorrect |
| Faithfulness | Judge checks that every claim in the answer is supported by the retrieved passages |
| Figure grounding | Automatic: every figure in the answer appears in a passage it cites |
| Citation accuracy | The answer cites at least one gold page |
| Unanswerable declined | Share of unanswerable questions where the system says the documents don't contain it |

The judge is Claude, the same model family that writes the answers, so it may be lenient toward
its own style. Correctness is judged against your hand-written answer, which limits that.
