#!/usr/bin/env python3
"""Run one gold question through a selected traversal strategy."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from typesafe_sdk import RetryPolicy, TypeSafeClient

from datalab_dfs_search import MODEL, load_grounded_document
from datalab_parse import load_local_env
from evaluated_dfs import EvaluatedDFSSearcher
from evaluation.metrics import map_gold_pages, score_trace
from evaluation.pricing import estimate_query_cost
from evaluation.strategies import ExhaustiveSectionSearcher, LLMSelectedSectionSearcher
from openai_grounded_answer import DEFAULT_OPENAI_MODEL


STRATEGIES = {
    "jev_dfs": EvaluatedDFSSearcher,
    "exhaustive": ExhaustiveSectionSearcher,
    "llm_toc": LLMSelectedSectionSearcher,
}


def _load_question(manifest: Path, identifier: str) -> dict[str, Any]:
    rows = json.loads(manifest.read_text(encoding="utf-8"))["questions"]
    matches = [row for row in rows if row["id"] == identifier]
    if len(matches) != 1:
        raise ValueError(f"Expected one question {identifier!r}, found {len(matches)}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grounded_dir", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("question_id")
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--env-file", type=Path, default=Path(__file__).parents[1] / ".env")
    parser.add_argument("--openai-model", default=None)
    args = parser.parse_args()

    load_local_env(args.env_file)
    if not os.environ.get("TYPESAFE_API_KEY") or not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("TYPESAFE_API_KEY and OPENAI_API_KEY are required")
    question = _load_question(args.manifest, args.question_id)
    pages, tree = load_grounded_document(args.grounded_dir)
    gold = map_gold_pages(tree, question["evidence_pages"])
    client = TypeSafeClient(
        api_key=os.environ["TYPESAFE_API_KEY"],
        model=MODEL,
        retry=RetryPolicy(max_retries=5, backoff_initial=1.0, backoff_max=20.0),
    )
    searcher = STRATEGIES[args.strategy](
        typesafe=client,
        pages=pages,
        tree=tree,
        question=question["question"],
        workers=args.workers,
        api_key=os.environ["OPENAI_API_KEY"],
        answer_model=args.openai_model or os.environ.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL,
    )
    started = time.perf_counter()
    hit = None
    failure = None
    try:
        hit = searcher.search()
    except Exception as exc:  # Persist partial metrics before failing the batch job.
        failure = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    elapsed = time.perf_counter() - started
    answer = searcher.final_answer()
    usage = {
        **searcher.usage_summary(),
        "writer_calls": searcher.writer_calls,
        "writer_input_tokens": searcher.writer_input_tokens,
        "writer_output_tokens": searcher.writer_output_tokens,
        "writer_latency_seconds": round(searcher.writer_latency_seconds, 6),
    }
    if isinstance(searcher, LLMSelectedSectionSearcher):
        usage.update(searcher.selector_usage())
    parse_summary = args.grounded_dir.parent / "summary.json"
    one_time_parse_cost = None
    if parse_summary.is_file():
        parsed = json.loads(parse_summary.read_text(encoding="utf-8"))
        cents = (parsed.get("cost_breakdown") or {}).get("final_cost_cents")
        one_time_parse_cost = float(cents) / 100 if cents is not None else None
    result = {
        "schema_version": 1,
        "strategy": args.strategy,
        "question": question,
        "gold": asdict(gold),
        "retrieval_metrics": score_trace(searcher.trace, gold),
        "answer": answer,
        "answer_correct": None,
        "failure": failure,
        "hit": asdict(hit) if hit else None,
        "coverage": searcher.coverage(),
        "usage": usage,
        "cost": {
            **estimate_query_cost(usage),
            "one_time_document_parse_cost_usd": one_time_parse_cost,
        },
        "elapsed_seconds": round(elapsed, 6),
        "trace": searcher.trace,
        "request_trace": searcher.request_trace,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "strategy": args.strategy,
        "question_id": question["id"],
        "status": "failed" if failure else answer["status"],
        "retrieval_metrics": result["retrieval_metrics"],
        "usage": usage,
        "cost": result["cost"],
        "elapsed_seconds": result["elapsed_seconds"],
        "output": str(args.output),
    }, indent=2))
    return 1 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())

