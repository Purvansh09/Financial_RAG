"""Score the system on the hand-written question set.

    python -m eval.run_eval --retrieval-only --label baseline   # free: retrieval metrics only
    python -m eval.run_eval --label baseline                    # + answers, judged by Claude

Retrieval and generation are scored separately, so a bad answer can be traced to either
"the right page was never retrieved" or "it was retrieved but the answer was wrong".
Every run saves all per-question details plus the settings used, and appends one summary row
to eval/results/summary.csv.
"""

import argparse
import csv
import json
import re
import time
from datetime import datetime

import anthropic

import config
from eval.dataset import gold_set, load_questions
from generation.answer import generate, get_client
from retrieval.retrieve import get_retriever

NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
CITATION = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")

JUDGE_SYSTEM = """\
You grade answers from a question-answering system over company financial documents. \
Be strict and literal. Judge only what is written; do not reward effort or style."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "correctness": {"type": "string", "enum": ["correct", "partially_correct", "incorrect"]},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": ["correctness", "unsupported_claims", "reason"],
    "additionalProperties": False,
}


def first_gold_rank(retrieved: list[dict], gold: set[tuple[str, int]]) -> int | None:
    """1-based rank of the first retrieved chunk from a gold page, or None if absent."""
    for rank, chunk in enumerate(retrieved, start=1):
        if (chunk["source_file"], chunk["page"]) in gold:
            return rank
    return None


def figures_grounded(answer: str, cited_chunks: list[dict]) -> bool:
    """Every figure in the answer appears (ignoring commas) in a passage it cites."""
    body = CITATION.sub("", answer)
    figures = [n for n in NUM.findall(body) if not re.fullmatch(r"(19|20)\d\d|\d", n)]
    pool = {n.replace(",", "") for c in cited_chunks for n in NUM.findall(c["text"])}
    return all(f.replace(",", "") in pool for f in figures)


def judge_prompt(q: dict, result: dict) -> str:
    passages = "\n\n".join(f'<passage id="{n}" page="{c["page"]}" file="{c["source_file"]}">\n'
                           f'{c["text"]}\n</passage>' for n, c in enumerate(result["sources"], 1))
    if q["type"] == "unanswerable":
        task = ("The documents do NOT contain the answer to this question. Grade correctness as "
                "'correct' if the system said the information is not available (giving clearly "
                "labelled related context is fine), 'incorrect' if it presented an answer to the "
                "question as if the documents contained one.")
        reference = "(none: the question is unanswerable from the documents)"
    else:
        task = ("Grade correctness against the reference answer: 'correct' if it states the "
                "reference answer (extra accurate detail is fine), 'partially_correct' if it gets "
                "only part of it or hedges between the right value and a wrong one, 'incorrect' "
                "if it is wrong, missing, or says the information is not available.")
        reference = q["answer"]
    return (f"<question>{q['question']}</question>\n<reference_answer>{reference}</reference_answer>\n"
            f"<system_answer>{result['answer']}</system_answer>\n<retrieved_passages>\n{passages}\n"
            f"</retrieved_passages>\n\n1. {task}\n2. List every factual claim in the system answer "
            "that is NOT supported by the retrieved passages (not the reference answer; the "
            "passages). Claims that the documents do not contain something count as supported. "
            "Empty list if all claims are supported.\n3. Give a one-sentence reason.")


def judge(q: dict, result: dict, client: anthropic.Anthropic) -> dict:
    """Ask Claude to grade correctness and faithfulness; returns the parsed JSON verdict."""
    response = client.messages.create(
        model=config.JUDGE_MODEL,
        max_tokens=config.LLM_MAX_TOKENS,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": judge_prompt(q, result)}],
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
    )
    verdict = json.loads(next(b.text for b in response.content if b.type == "text"))
    verdict["usage"] = {"input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens}
    return verdict


def score_question(q: dict, retriever, client, retrieval_only: bool) -> dict:
    """Retrieve, optionally answer and judge, and return one result row."""
    gold = gold_set(q)
    retrieved = retriever.search(q["question"], max(config.EVAL_KS))
    rank = first_gold_rank(retrieved, gold) if gold else None
    row = {"id": q["id"], "type": q["type"], "question": q["question"], "gold_answer": q["answer"],
           "gold_pages": sorted(f"{f} p.{p}" for f, p in gold), "gold_rank": rank,
           "retrieved": [f"{c['source_file']} p.{c['page']}" for c in retrieved]}
    if retrieval_only:
        return row

    start = time.time()
    result = generate(q["question"], retrieved[:config.TOP_K], client)
    row["latency_s"] = round(time.time() - start, 1)
    by_id = {c["chunk_id"]: c for c in result["sources"]}
    cited = [by_id[c["chunk_id"]] for c in result["citations"]]
    row.update({
        "answer": result["answer"], "not_found": result["not_found"], "warnings": result["warnings"],
        "cited_pages": [f"{c['source_file']} p.{c['page']}" for c in cited],
        "cites_gold": any((c["source_file"], c["page"]) in gold for c in cited) if gold else None,
        "figures_grounded": figures_grounded(result["answer"], cited),
        "usage": result["usage"],
    })
    if result["not_found"]:  # exact decline: no claims to check, correct only if unanswerable
        row.update(correctness="correct" if q["type"] == "unanswerable" else "incorrect",
                   unsupported_claims=[], reason="Exact 'not found' reply.")
    else:
        verdict = judge(q, result, client)
        row.update(correctness=verdict["correctness"], unsupported_claims=verdict["unsupported_claims"],
                   reason=verdict["reason"], judge_usage=verdict["usage"])
    row["faithful"] = not row["unsupported_claims"]
    return row


def pct(values: list[bool]) -> float | None:
    return round(100 * sum(values) / len(values), 1) if values else None


def summarize(rows: list[dict]) -> dict:
    """Headline metrics over a set of result rows (any subset, e.g. one question type)."""
    answerable = [r for r in rows if r["type"] != "unanswerable"]
    out = {"n": len(rows), "n_answerable": len(answerable)}
    for k in config.EVAL_KS:
        out[f"hit@{k}"] = pct([r["gold_rank"] is not None and r["gold_rank"] <= k for r in answerable])
    out["mrr@10"] = (round(sum(1 / r["gold_rank"] for r in answerable if r["gold_rank"]) / len(answerable), 3)
                     if answerable else None)
    if rows and "correctness" in rows[0]:
        out["correct"] = pct([r["correctness"] == "correct" for r in answerable])
        out["correct_or_partial"] = pct([r["correctness"] != "incorrect" for r in answerable])
        out["faithful"] = pct([r["faithful"] for r in rows if not r["not_found"]])
        out["figures_grounded"] = pct([r["figures_grounded"] for r in rows if not r["not_found"]])
        out["cites_gold"] = pct([r["cites_gold"] for r in answerable if not r["not_found"]])
        out["unanswerable_declined"] = pct([r["correctness"] == "correct"
                                            for r in rows if r["type"] == "unanswerable"])
        out["answers_with_warnings"] = pct([bool(r["warnings"]) for r in rows])
    return out


def settings() -> dict:
    """What this run used, so results can be compared across experiments."""
    meta = json.loads(config.INDEX_META_PATH.read_text(encoding="utf-8"))
    return {"embedding_model": config.EMBEDDING_MODEL, "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP, "top_k": config.TOP_K, "n_chunks": meta["n_chunks"],
            "llm_model": config.LLM_MODEL, "llm_effort": config.LLM_EFFORT, "judge_model": config.JUDGE_MODEL}


def print_table(overall: dict, by_type: dict[str, dict]) -> None:
    """Markdown table: one row per question type plus the overall row."""
    cols = [c for c in overall if c not in ("n_answerable",)]
    print("| Set | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for name, s in [*by_type.items(), ("**overall**", overall)]:
        print(f"| {name} | " + " | ".join("" if s.get(c) is None else str(s[c]) for c in cols) + " |")


def save(label: str, rows: list[dict], overall: dict, by_type: dict, run_settings: dict) -> None:
    """Write the full run to JSON and append the headline row to summary.csv."""
    config.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = config.EVAL_RESULTS_DIR / f"{stamp}_{label}.json"
    path.write_text(json.dumps({"label": label, "timestamp": stamp, "settings": run_settings,
                                "overall": overall, "by_type": by_type, "questions": rows},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    summary = config.EVAL_RESULTS_DIR / "summary.csv"
    record = {"timestamp": stamp, "label": label, **run_settings, **overall}
    new = not summary.exists()
    with open(summary, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(record))
        if new:
            writer.writeheader()
        writer.writerow(record)
    print(f"\nSaved {path.relative_to(config.ROOT)} and a row in {summary.relative_to(config.ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="run", help="name for this experiment, e.g. baseline")
    parser.add_argument("--questions", default=str(config.EVAL_DIR / "questions.jsonl"))
    parser.add_argument("--retrieval-only", action="store_true", help="skip answers and judging (free)")
    parser.add_argument("--limit", type=int, help="only the first N questions (for a trial run)")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    from pathlib import Path
    questions = load_questions(Path(args.questions))[:args.limit]
    retriever = get_retriever()
    client = None if args.retrieval_only else get_client()
    if client:
        print(f"{len(questions)} questions; estimated cost ~${0.06 * len(questions):.2f} "
              f"({config.LLM_MODEL} answers, {config.JUDGE_MODEL} judge)")

    rows = []
    for i, q in enumerate(questions, start=1):
        row = score_question(q, retriever, client, args.retrieval_only)
        rows.append(row)
        status = f"rank {row['gold_rank']}" if q["type"] != "unanswerable" else "unanswerable"
        print(f"[{i}/{len(questions)}] {q['id']} ({q['type']}): {status}"
              + (f", {row['correctness']}, faithful={row['faithful']}" if "correctness" in row else ""))

    overall = summarize(rows)
    by_type = {t: summarize([r for r in rows if r["type"] == t])
               for t in sorted({r["type"] for r in rows})}
    print()
    print_table(overall, by_type)
    if not args.retrieval_only:
        tokens = [r.get("usage", {}) for r in rows] + [r.get("judge_usage", {}) for r in rows]
        cost = sum(t.get("input_tokens", 0) * 4 + t.get("output_tokens", 0) * 20 for t in tokens) / 1e6
        print(f"\nAPI cost ~${cost:.2f} at {config.LLM_MODEL} list prices")
    if not args.no_save:
        save(args.label, rows, overall, by_type, settings())


if __name__ == "__main__":
    main()
