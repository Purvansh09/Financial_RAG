"""Load and validate the hand-written question set (eval/questions.jsonl)."""

import json
import re
from pathlib import Path

import config

QUESTIONS_PATH = config.EVAL_DIR / "questions.jsonl"
TYPES = {"fact", "table", "earnings_call", "cross_year", "unanswerable"}
FIGURE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    """Read questions.jsonl, skipping blank lines. Raises with the line number on bad JSON."""
    questions = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                questions.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path.name} line {n}: invalid JSON ({e.msg})") from e
    return questions


def gold_set(question: dict) -> set[tuple[str, int]]:
    """The (source_file, page) pairs that answer a question."""
    return {(g["source_file"], int(g["page"])) for g in question.get("gold_pages", [])}


def load_pages(path: Path = config.PAGES_PATH) -> dict[tuple[str, int], dict]:
    """Parsed pages keyed by (source_file, page)."""
    with open(path, encoding="utf-8") as f:
        return {(p["source_file"], p["page"]): p for p in map(json.loads, f)}


def validate(questions: list[dict], pages: dict[tuple[str, int], dict]) -> tuple[list[str], list[str]]:
    """Check the question set. Returns (errors, warnings); errors must be fixed before a run."""
    errors, warnings = [], []
    files = {f for f, _ in pages}
    page_counts = {f: max(p for g, p in pages if g == f) for f in files}
    seen = set()

    for q in questions:
        qid = q.get("id", "?")
        missing = [k for k in ("id", "type", "question", "answer", "gold_pages") if k not in q]
        if missing:
            errors.append(f"{qid}: missing field(s) {missing}")
            continue
        if qid in seen:
            errors.append(f"{qid}: duplicate id")
        seen.add(qid)
        if q["type"] not in TYPES:
            errors.append(f"{qid}: type '{q['type']}' is not one of {sorted(TYPES)}")
        if q["type"] == "unanswerable":
            if q["gold_pages"]:
                errors.append(f"{qid}: unanswerable questions should have no gold_pages")
            continue
        if not q["gold_pages"]:
            errors.append(f"{qid}: needs at least one gold page")
        if not q["answer"].strip():
            errors.append(f"{qid}: needs a gold answer")

        gold_text = ""
        for source_file, page in sorted(gold_set(q)):
            if source_file not in files:
                errors.append(f"{qid}: unknown source_file '{source_file}'")
            elif not 1 <= page <= page_counts[source_file]:
                errors.append(f"{qid}: {source_file} has pages 1-{page_counts[source_file]}, not {page}")
            else:
                p = pages[(source_file, page)]
                gold_text += p["text"].replace(",", "")
                if p["garbled"]:
                    warnings.append(f"{qid}: {source_file} p.{page} has broken encoding and is "
                                    "not in the index, so retrieval can never find it")

        figures = [f for f in FIGURE.findall(q["answer"]) if not re.fullmatch(r"(19|20)\d\d", f)]
        absent = [f for f in figures if f.replace(",", "") not in gold_text]
        if gold_text and absent:
            warnings.append(f"{qid}: answer figure(s) {absent} not found on any gold page "
                            "(check the page numbers)")
    return errors, warnings


def main() -> None:
    import sys
    questions = load_questions(Path(sys.argv[1]) if len(sys.argv) > 1 else QUESTIONS_PATH)
    errors, warnings = validate(questions, load_pages())
    counts = {t: sum(q.get("type") == t for q in questions) for t in sorted(TYPES)}
    print(f"{len(questions)} questions: " + ", ".join(f"{t} {n}" for t, n in counts.items()))
    for w in warnings:
        print(f"WARNING {w}")
    for e in errors:
        print(f"ERROR   {e}")
    print("OK" if not errors else f"{len(errors)} error(s) to fix before running the eval.")


if __name__ == "__main__":
    main()
