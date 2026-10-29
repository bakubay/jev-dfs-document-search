#!/usr/bin/env python3
"""Turn Datalab JSON into page records and a physically grounded ToC."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


class TableRows(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._cell is not None and self._row is not None:
            # Preserve leading spaces/non-breaking spaces: ToCs frequently encode
            # hierarchy visually even when numbering is absent or inconsistent.
            self._row.append("".join(self._cell))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None


def normalize(text: str) -> str:
    text = re.sub(r"^#+\s*", "", text)
    text = text.replace("–", "-").replace("—", "-")
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def block_text(block: dict[str, Any]) -> str:
    return (block.get("markdown") or block.get("html") or "").strip()


def parse_toc_rows(html: str, source_page_id: int) -> list[dict[str, Any]]:
    parser = TableRows()
    parser.feed(html)
    entries = []
    for row in parser.rows:
        if len(row) < 2:
            continue
        printed = re.fullmatch(r"\s*(\d+)\s*", row[-1])
        raw_title = " ".join(row[:-1])
        leading = re.match(r"^[\s\u00a0]*", raw_title).group(0)
        title = raw_title.strip()
        if not printed or not title:
            continue
        entries.append(
            {
                "title": title,
                "printed_page": int(printed.group(1)),
                "toc_page_id": source_page_id,
                "toc_indent": len(leading) // 4,
                "toc_leading_characters": len(leading),
            }
        )
    return entries


def extract_toc(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for page_id, page in enumerate(pages):
        active_heading = ""
        for block in page.get("children", []):
            if block.get("block_type") == "SectionHeader":
                active_heading = normalize(block_text(block))
            elif (
                block.get("block_type") == "TableOfContents"
                and active_heading == "table of contents"
            ):
                entries.extend(parse_toc_rows(block.get("html", ""), page_id))
    return entries


def extract_headings(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    headings = []
    for page_id, page in enumerate(pages):
        for block in page.get("children", []):
            if block.get("block_type") != "SectionHeader":
                continue
            text = block_text(block)
            headings.append(
                {
                    "title": re.sub(r"^#+\s*", "", text).strip(),
                    "normalized": normalize(text),
                    "page_id": page_id,
                    "pdf_page_number": page_id + 1,
                    "block_id": block.get("id"),
                    "bbox": block.get("bbox"),
                }
            )
    return headings


def best_heading(title: str, headings: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float]:
    needle = normalize(title)
    if not needle:
        return None, 0.0
    best = None
    best_score = 0.0
    for heading in headings:
        score = SequenceMatcher(None, needle, heading["normalized"]).ratio()
        if score > best_score:
            best, best_score = heading, score
    return best, best_score


def infer_offset(entries: list[dict[str, Any]], headings: list[dict[str, Any]]) -> tuple[int, Counter[int]]:
    votes: Counter[int] = Counter()
    for entry in entries:
        heading, score = best_heading(entry["title"], headings)
        if heading and score >= 0.90:
            offset = heading["pdf_page_number"] - entry["printed_page"]
            votes[offset] += 1
    if not votes:
        raise ValueError("Could not infer a printed-page to PDF-page offset")
    return votes.most_common(1)[0][0], votes


def page_markdown(page: dict[str, Any]) -> str:
    ignored = {"PageHeader", "PageFooter"}
    return "\n\n".join(
        block_text(block)
        for block in page.get("children", [])
        if block.get("block_type") not in ignored and block_text(block)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datalab_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    document = json.loads(args.datalab_json.read_text(encoding="utf-8"))
    pages = document["children"]
    entries = extract_toc(pages)
    headings = extract_headings(pages)
    offset, votes = infer_offset(entries, headings)

    grounded = []
    for entry in entries:
        expected_pdf_page = entry["printed_page"] + offset
        heading, score = best_heading(entry["title"], headings)
        actual_pdf_page = heading["pdf_page_number"] if heading and score >= 0.75 else None
        if actual_pdf_page == expected_pdf_page:
            status = "verified"
        elif actual_pdf_page and abs(actual_pdf_page - expected_pdf_page) <= 1:
            status = "verified_nearby"
        else:
            status = "unverified"
        grounded.append(
            {
                **entry,
                "pdf_page_number": expected_pdf_page,
                "page_id": expected_pdf_page - 1,
                "verification_status": status,
                "matched_heading": heading["title"] if heading else None,
                "matched_pdf_page_number": actual_pdf_page,
                "heading_similarity": round(score, 4),
                "matched_block_id": heading["block_id"] if heading else None,
                "matched_bbox": heading["bbox"] if heading else None,
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "toc_entries.json").write_text(
        json.dumps(grounded, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (args.output / "pages.jsonl").open("w", encoding="utf-8") as handle:
        for page_id, page in enumerate(pages):
            handle.write(
                json.dumps(
                    {
                        "page_id": page_id,
                        "pdf_page_number": page_id + 1,
                        "source_block_id": page.get("id"),
                        "bbox": page.get("bbox"),
                        "markdown": page_markdown(page),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    counts = Counter(item["verification_status"] for item in grounded)
    report = {
        "page_count": len(pages),
        "toc_entry_count": len(grounded),
        "printed_to_pdf_page_offset": offset,
        "offset_votes": dict(votes),
        "verification_counts": dict(counts),
        "all_pages_have_physical_ids": all(
            page.get("id") == f"/page/{i}/Page/{i}" for i, page in enumerate(pages)
        ),
        "failed_page_ids": document.get("metadata", {}).get("failed_pages", []),
    }
    (args.output / "grounding_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
