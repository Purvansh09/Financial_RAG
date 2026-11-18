"""ask(question) -> an answer grounded in retrieved passages, with checked citations.

ask() = retrieve() + generate(). generate() takes the passages as an argument, so it can also
be run on hand-picked passages to evaluate generation separately from retrieval.

The model cites sources by number. Code, not the model, turns those numbers into documents
and pages, and flags answers that could be ungrounded: no citations, citation numbers that
don't exist, or figures stated without a citation. Flags are reported, not hidden, so the UI
can warn and the eval can count them.
"""

import os
import re
import sys
from functools import lru_cache

import anthropic

import config
from generation.prompts import NOT_FOUND, SYSTEM_PROMPT, build_user_message
from retrieval.retrieve import retrieve

CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
# A financial figure, as opposed to a date or year: a decimal, a comma-grouped number, a
# currency amount, or a number with a unit. "March 31, 2025" and "FY2026" don't match.
FIGURE = re.compile(
    r"\d+\.\d+|\d{1,3}(?:,\d{2,3})+|₹\s*\d"
    r"|\d+\s*(?:%|per ?cent|crore|lakh|bn\b|billion|mn\b|million|bps\b|basis points)",
    re.IGNORECASE,
)

# Server-side refusal fallback: if a safety classifier declines the request, the API re-runs
# it on a recommended model instead of returning a refusal. Only these models support it.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}


@lru_cache(maxsize=1)
def get_client() -> anthropic.Anthropic:
    """API client; reads ANTHROPIC_API_KEY, which config.py loads from .env."""
    return anthropic.Anthropic()


def parse_citations(text: str, n_sources: int) -> tuple[list[int], list[int]]:
    """Cited source numbers in order of first use, split into (valid, out of range)."""
    seen: list[int] = []
    for match in CITATION.finditer(text):
        for num in match.group(1).split(","):
            if (n := int(num)) not in seen:
                seen.append(n)
    valid = [n for n in seen if 1 <= n <= n_sources]
    return valid, [n for n in seen if n not in valid]


def split_sentences(text: str) -> list[str]:
    """Split into sentences, keeping a citation that follows the full stop ("... crore. [2]").

    List markers ("1. ", "- ") are removed first so a list number isn't read as a sentence.
    A markdown table is kept as one block, so a citation on the line after it covers it.
    """
    parts = []
    for line in text.splitlines():
        if line.lstrip().startswith("|"):
            parts.append(line)
        else:
            parts += re.split(r"(?<=[.!?])\s+", LIST_MARKER.sub("", line))
    sentences: list[str] = []
    for part in (p for p in parts if p.strip()):
        is_table_row = part.lstrip().startswith("|")
        if sentences and (CITATION.match(part.strip())
                          or (is_table_row and sentences[-1].lstrip().startswith("|"))):
            sentences[-1] += "\n" + part
        else:
            sentences.append(part)
    return sentences


def uncited_figures(text: str) -> list[str]:
    """Sentences that state a number but cite nothing; the riskiest kind of claim."""
    flagged = []
    for sentence in split_sentences(text):
        if not CITATION.search(sentence) and FIGURE.search(sentence):
            flagged.append(sentence.strip())
    return flagged


def is_not_found(text: str) -> bool:
    """True if the model gave the exact not-found reply."""
    return text.strip().rstrip(".").lower() == NOT_FOUND.rstrip(".").lower()


def _citation_record(n: int, chunk: dict) -> dict:
    keys = ("chunk_id", "source", "source_file", "company", "year", "doc_type", "page")
    return {"n": n, **{k: chunk[k] for k in keys if k in chunk}}


def generate(question: str, chunks: list[dict], client: anthropic.Anthropic | None = None) -> dict:
    """Answer `question` from `chunks` only, and check the answer's citations."""
    client = client or get_client()
    fallback = {"betas": [FALLBACK_BETA], "fallbacks": "default"} \
        if config.LLM_MODEL in FALLBACK_MODELS else {}
    response = client.beta.messages.create(
        model=config.LLM_MODEL,
        max_tokens=config.LLM_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_message(question, chunks)}],
        output_config={"effort": config.LLM_EFFORT},
        **fallback,
    )

    text = "".join(b.text for b in response.content if b.type == "text").strip()
    warnings = []
    if response.stop_reason == "refusal":
        text = ""
        warnings.append("The model declined to answer this question.")
    elif response.stop_reason == "max_tokens":
        warnings.append("The answer was cut off (max_tokens reached).")

    not_found = is_not_found(text)
    valid, invalid = parse_citations(text, len(chunks))
    figures = [] if not_found else uncited_figures(text)
    if text and not not_found and not valid:
        warnings.append("The answer cites no sources.")
    if invalid:
        warnings.append(f"The answer cites sources that were not provided: {invalid}.")
    if figures:
        warnings.append(f"{len(figures)} sentence(s) state figures without a citation.")

    return {
        "question": question,
        "answer": text,
        "not_found": not_found,
        "citations": [_citation_record(n, chunks[n - 1]) for n in valid],
        "invalid_citations": invalid,
        "uncited_figures": figures,
        "warnings": warnings,
        "sources": chunks,
        "model": response.model,
        "stop_reason": response.stop_reason,
        "usage": {"input_tokens": response.usage.input_tokens,
                  "output_tokens": response.usage.output_tokens},
    }


def ask(question: str, k: int = config.TOP_K, where: dict | None = None,
        client: anthropic.Anthropic | None = None) -> dict:
    """Retrieve the top-k passages for `question` and answer from them."""
    return generate(question, retrieve(question, k, where), client)


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What was HDFC Bank's gross NPA ratio at March 31, 2025?"
    if not (config.ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        sys.exit("No API key found. Copy .env.example to .env and set ANTHROPIC_API_KEY.")
    try:
        result = ask(question)
    except anthropic.AuthenticationError:
        sys.exit("No valid API key. Copy .env.example to .env and set ANTHROPIC_API_KEY.")
    except anthropic.APIConnectionError as e:
        sys.exit(f"Could not reach the Anthropic API: {e}")

    print(f"Q: {result['question']}\n\n{result['answer']}\n")
    if result["citations"]:
        print("Sources:")
        for c in result["citations"]:
            print(f"  [{c['n']}] {c['source']}")
    for w in result["warnings"]:
        print(f"WARNING: {w}")
    u = result["usage"]
    print(f"\n({result['model']}, {u['input_tokens']} in / {u['output_tokens']} out tokens)")
