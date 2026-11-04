#!/usr/bin/env python3
"""Prepare the public PageIndex OSS benchmark for this ToC-first experiment.

The upstream benchmark supplies PDFs, answers, and evidence pages. This module
does not vendor those files. It reads a local checkout, performs a cheap local
text screen for a visible table of contents, and writes a normalized manifest.
The Datalab parse/grounding step remains separate because it consumes credits.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SOURCE_URL = "https://github.com/VectifyAI/PageIndex-OSS-Benchmark"


@dataclass(frozen=True)
class TocScreen:
    doc_id: str
    page_count: int
    contents_heading_count: int
    entry_like_line_count: int
    likely_real_toc: bool
    reason: str


def parse_evidence_pages(value: Any) -> list[int]:
    """Normalize the upstream stringified-list field without using eval."""
    if isinstance(value, list):
        raw = value
    elif isinstance(value, str):
        raw = ast.literal_eval(value)
    else:
        raise ValueError(f"Unsupported evidence_pages value: {value!r}")
    pages = sorted({int(page) for page in raw})
    if not pages or pages[0] < 1:
        raise ValueError(f"Invalid evidence pages: {value!r}")
    return pages


def question_id(doc_id: str, question: str) -> str:
    digest = hashlib.sha256(f"{doc_id}\0{question}".encode()).hexdigest()[:12]
    return f"pi-oss-{digest}"


def score_toc_text(text: str) -> tuple[int, int]:
    """Return visible ToC heading and entry-like line counts."""
    heading = re.compile(r"^\s*(?:table\s+of\s+)?contents\s*$", re.IGNORECASE)
    dotted = re.compile(r"\.{2,}\s*(?:[ivxlcdm]+|\d{1,4})\s*$", re.IGNORECASE)
    numbered = re.compile(
        r"^\s*(?:\d+(?:\.\d+)*[.)]?\s+)?"
        r"[A-Za-z][A-Za-z0-9 ,:;()/'&+\-]{4,}\s+"
        r"(?:[ivxlcdm]+|\d{1,4})\s*$",
        re.IGNORECASE,
    )
    lines = [line.rstrip() for line in text.splitlines()]
    return (
        sum(bool(heading.match(line)) for line in lines),
        sum(bool(dotted.search(line) or numbered.match(line)) for line in lines),
    )


def _pdf_text(pdf: Path, start: int, end: int) -> str:
    result = subprocess.run(
        ["pdftotext", "-f", str(start), "-l", str(end), "-layout", str(pdf), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def screen_pdf(pdf: Path, page_count: int, window: int = 20) -> TocScreen:
    first_end = min(window, page_count)
    chunks = [_pdf_text(pdf, 1, first_end)]
    if page_count > window:
        chunks.append(_pdf_text(pdf, max(1, page_count - window + 1), page_count))
    headings, entries = score_toc_text("\n".join(chunks))
    likely = headings > 0 and entries >= 3
    if likely:
        reason = f"visible contents heading plus {entries} entry-like lines"
    elif headings == 0:
        reason = "no visible contents heading in the first/last page windows"
    else:
        reason = f"contents heading found, but only {entries} entry-like lines"
    return TocScreen(
        doc_id=pdf.name,
        page_count=page_count,
        contents_heading_count=headings,
        entry_like_line_count=entries,
        likely_real_toc=likely,
        reason=reason,
    )


def _source_commit(directory: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def prepare(
    benchmark_dir: Path,
    output: Path,
    window: int = 20,
    min_pages: int = 70,
) -> dict[str, Any]:
    questions_path = benchmark_dir / "questions.json"
    documents_path = benchmark_dir / "documents.json"
    pdf_dir = benchmark_dir / "documents"
    if not questions_path.is_file() or not documents_path.is_file() or not pdf_dir.is_dir():
        raise ValueError(f"Not a PageIndex OSS benchmark checkout: {benchmark_dir}")

    raw_questions = json.loads(questions_path.read_text(encoding="utf-8"))
    raw_documents = json.loads(documents_path.read_text(encoding="utf-8"))
    questions_by_doc: dict[str, list[dict[str, Any]]] = {}
    normalized_questions = []
    for row in raw_questions:
        normalized = {
            "id": question_id(row["doc_id"], row["question"]),
            "doc_id": row["doc_id"],
            "question": row["question"],
            "answer": row["answer"],
            "answer_format": row.get("answer_format"),
            "task_type": row.get("task_type"),
            "doc_type": row.get("doc_type"),
            "evidence_pages": parse_evidence_pages(row["evidence_pages"]),
            "answerable": True,
            "source": "PageIndex-OSS-Benchmark / MMLongBench-Doc-V2",
        }
        normalized_questions.append(normalized)
        questions_by_doc.setdefault(row["doc_id"], []).append(normalized)

    screens = []
    for document in raw_documents:
        pdf = pdf_dir / document["doc_id"]
        if not pdf.is_file():
            raise FileNotFoundError(pdf)
        screen = screen_pdf(pdf, int(document["pages"]), window=window)
        screens.append(
            {
                **asdict(screen),
                "eligible_for_initial_eval": screen.likely_real_toc and screen.page_count >= min_pages,
                "question_count": len(questions_by_doc.get(document["doc_id"], [])),
                "source_pdf": str(pdf.resolve()),
                "upstream_metadata": document,
            }
        )

    selected_ids = {row["doc_id"] for row in screens if row["eligible_for_initial_eval"]}
    manifest = {
        "schema_version": 1,
        "source": {
            "name": "PageIndex-OSS-Benchmark",
            "url": SOURCE_URL,
            "commit": _source_commit(benchmark_dir),
            "note": (
                "Upstream files are referenced from a local checkout and are not "
                "vendored into this repository. Confirm upstream licensing before redistribution."
            ),
        },
        "screening": {
            "method": "pdftotext over first/last page windows; visible Contents heading plus >=3 entry-like lines",
            "window_pages": window,
            "minimum_document_pages": min_pages,
            "screened_documents": len(screens),
            "selected_documents": len(selected_ids),
            "selected_questions": sum(q["doc_id"] in selected_ids for q in normalized_questions),
            "warning": "This is a cheap candidate screen, not proof that Datalab can extract and ground the ToC.",
        },
        "documents": screens,
        "questions": normalized_questions,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--min-pages", type=int, default=70)
    args = parser.parse_args()
    if args.window < 1 or args.min_pages < 1:
        parser.error("--window and --min-pages must be positive")
    manifest = prepare(args.benchmark_dir, args.output, args.window, args.min_pages)
    selected = [row for row in manifest["documents"] if row["eligible_for_initial_eval"]]
    print(json.dumps(manifest["screening"], indent=2))
    for row in selected:
        print(f"  {row['doc_id']}: {row['page_count']} pages, {row['question_count']} questions")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

