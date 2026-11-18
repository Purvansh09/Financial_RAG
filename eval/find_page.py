"""Find which PDF pages contain some text, to fill in gold_pages.

    python -m eval.find_page "1,62,990"
    python -m eval.find_page "CASA ratio" --file hdfcbank

Matching ignores case, and ignores commas when the search text is a number, so "162990"
also finds "1,62,990". Page numbers printed are PDF page numbers (what gold_pages needs).
"""

import argparse
import re

from eval.dataset import load_pages


def find(text: str, pages: dict, file_filter: str = "") -> list[tuple[str, int, str]]:
    """(source_file, page, snippet) for every page containing `text`."""
    numeric = bool(re.fullmatch(r"[\d,.\s]+", text.strip()))
    needle = text.replace(",", "").lower() if numeric else text.lower()
    hits = []
    for (source_file, page), p in sorted(pages.items()):
        if file_filter.lower() not in source_file.lower():
            continue
        body = " ".join(p["text"].split())
        haystack = body.replace(",", "").lower() if numeric else body.lower()
        if (i := haystack.find(needle)) >= 0:
            snippet = haystack[max(0, i - 60):i + len(needle) + 60] if numeric else body[max(0, i - 60):i + len(text) + 60]
            hits.append((source_file, page, snippet + ("  [NOT INDEXED: broken encoding]" if p["garbled"] else "")))
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text")
    parser.add_argument("--file", default="", help="only files whose name contains this")
    args = parser.parse_args()
    hits = find(args.text, load_pages(), args.file)
    for source_file, page, snippet in hits[:40]:
        print(f"{source_file} p.{page}: ...{snippet}...")
    print(f"{len(hits)} page(s)" + (" (showing 40)" if len(hits) > 40 else ""))


if __name__ == "__main__":
    main()
