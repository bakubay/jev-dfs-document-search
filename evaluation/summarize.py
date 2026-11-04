#!/usr/bin/env python3
"""Aggregate completed case JSON files into a comparison table."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def mean(rows: list[dict[str, Any]], path: tuple[str, ...]) -> float | None:
    values = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return statistics.mean(values) if values else None


def mean_length(rows: list[dict[str, Any]], path: tuple[str, ...]) -> float | None:
    values = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, list):
            values.append(len(value))
    return statistics.mean(values) if values else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["strategy"]].append(row)
    strategies = []
    for strategy, items in sorted(groups.items()):
        judged = [row for row in items if isinstance(row.get("answer_correct"), bool)]
        strategies.append({
            "strategy": strategy,
            "cases": len(items),
            "failures": sum(row.get("failure") is not None for row in items),
            "answer_accuracy": (
                sum(row["answer_correct"] for row in judged) / len(judged) if judged else None
            ),
            "mean_gold_section_recall": mean(items, ("retrieval_metrics", "gold_section_recall")),
            "mean_gold_page_recall": mean(items, ("retrieval_metrics", "gold_page_recall")),
            "mean_pages_screened": mean_length(items, ("coverage", "screened_pages")),
            "mean_pages_verified": mean(items, ("coverage", "checked_count")),
            "mean_latency_seconds": mean(items, ("elapsed_seconds",)),
            "mean_jev_requests": mean(items, ("usage", "jev_requests")),
            "mean_jev_questions": mean(items, ("usage", "jev_questions")),
            "mean_sol_calls": mean(items, ("usage", "writer_calls")),
            "mean_query_cost_usd": mean(items, ("cost", "estimated_query_cost_usd")),
        })
    return {"case_count": len(rows), "strategies": strategies}


def markdown(summary: dict[str, Any]) -> str:
    headers = [
        "strategy", "cases", "failures", "answer accuracy", "section recall",
        "pages screened", "pages verified", "JEV requests", "Sol calls", "latency (s)", "query cost ($)",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in summary["strategies"]:
        def show(value: Any, digits: int = 3) -> str:
            return "—" if value is None else f"{value:.{digits}f}" if isinstance(value, float) else str(value)
        lines.append("| " + " | ".join([
            row["strategy"], str(row["cases"]), str(row["failures"]),
            show(row["answer_accuracy"]), show(row["mean_gold_section_recall"]),
            show(row["mean_pages_screened"], 1), show(row["mean_pages_verified"], 1),
            show(row["mean_jev_requests"], 1),
            show(row["mean_sol_calls"], 1), show(row["mean_latency_seconds"], 2),
            show(row["mean_query_cost_usd"], 6),
        ]) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    result = summarize(rows)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.markdown_output.write_text(markdown(result), encoding="utf-8")
    print(markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

