"""Run the three current retrieval methods against annotated questions."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random

from typesafe_sdk import RetryPolicy, TypeSafeClient

from retrieval.tree import MODEL, load_grounded_document
from retrieval.config import load_local_env
from evaluation.pricing import estimate_query_cost
from retrieval.config import DEFAULT_OPENAI_MODEL
from retrieval import EXPERIMENT_VERSION
from retrieval.jev import JevDFSRetriever, JevExhaustiveRetriever
from retrieval.common import EvidenceBudget
from evaluation.metrics import score_retrieval
from retrieval.run import run_retrieval, runtime_sources
from retrieval.sol import SolRetriever

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = {"jev_dfs": JevDFSRetriever, "exhaustive": JevExhaustiveRetriever, "sol_only": SolRetriever}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--grounded-root", type=Path, required=True)
    parser.add_argument("--question-ids", nargs="+", required=True)
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--max-navigation-steps", type=int, default=32)
    parser.add_argument("--openai-model", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_navigation_steps < 1:
        parser.error("Steps must be positive")
    questions = {q["id"]: q for q in json.loads(args.manifest.read_text())["questions"]}
    if set(args.question_ids) - questions.keys():
        parser.error("Unknown question ID")
    cases = []
    for qid in dict.fromkeys(args.question_ids):
        grounded = args.grounded_root / Path(questions[qid]["doc_id"]).stem / "grounded"
        if not all((grounded / n).is_file() for n in ("pages.jsonl", "toc_entries.json")):
            parser.error(f"Missing grounded input: {grounded}")
        for strategy in dict.fromkeys(args.strategies):
            cases.append({"question_id": qid, "strategy": strategy, "grounded_dir": str(grounded)})
    random.Random(args.seed).shuffle(cases)
    if args.dry_run:
        print(json.dumps({"experiment": EXPERIMENT_VERSION, "cases": cases}, indent=2))
        return 0
    load_local_env(args.env_file)
    required = set()
    if "sol_only" in args.strategies:
        required.add("OPENAI_API_KEY")
    if set(args.strategies) - {"sol_only"}:
        required.add("TYPESAFE_API_KEY")
    if any(not os.environ.get(key) for key in required):
        parser.error("Required configured API key is missing")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "schedule.json").write_text(json.dumps({"experiment": EXPERIMENT_VERSION, "seed": args.seed, "cases": cases}, indent=2))
    sources = runtime_sources()
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    failed = 0
    model = args.openai_model or os.environ.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
    for i, case in enumerate(cases, 1):
        print(f"Retrieval run {i}/{len(cases)}: {case['question_id']} {case['strategy']}", flush=True)
        question, strategy = questions[case["question_id"]], case["strategy"]
        grounded = Path(case["grounded_dir"])
        pages, tree = load_grounded_document(grounded)
        sol = strategy == "sol_only"
        client = None if sol else TypeSafeClient(api_key=os.environ["TYPESAFE_API_KEY"], model=MODEL,
            retry=RetryPolicy(max_retries=5, backoff_initial=1.0, backoff_max=20.0))
        extra = {"api_key": os.environ["OPENAI_API_KEY"], "model": model,
                 "max_navigation_steps": args.max_navigation_steps} if sol else {}
        search = STRATEGIES[strategy](typesafe=client, pages=pages, tree=tree, question=question["question"],
                                      evidence_budget=EvidenceBudget(), **extra)
        result, elapsed, failure = run_retrieval(search)
        if client is not None:
            client.close()
        failed += failure is not None
        usage = {**search.usage_summary(), "writer_calls": 0, "writer_input_tokens": 0,
                 "writer_output_tokens": 0, "answer_judge_calls": 0}
        record = {
            "schema_version": 1, "experiment": EXPERIMENT_VERSION,
            "recorded_at": datetime.now(timezone.utc).isoformat(), "strategy": strategy, "question": question,
            "configuration": {
                "jev_model": None if sol else MODEL, "sol_model": model if sol else None,
                "concurrency": 1, "max_navigation_steps": args.max_navigation_steps if sol else None,
                "sufficiency_threshold": None if sol else search.sufficiency_threshold,
                "recall_threshold": None if sol else search.recall_threshold,
                "max_pages_per_read": search.max_pages_per_read, "max_read_tokens": search.max_read_tokens,
                "output_budget": asdict(search.evidence_budget),
                "page_check_policy": "raw_page_agent" if sol else "one_judgment_per_page",
                "page_scheduling": "agent_selected" if sol else "document_wide_single_pass" if strategy == "exhaustive" else "leaf_local_windows",
                "grounded_pages_sha256": hashlib.sha256((grounded / "pages.jsonl").read_bytes()).hexdigest(),
                "toc_sha256": hashlib.sha256((grounded / "toc_entries.json").read_bytes()).hexdigest(),
                "source_sha256": hashes,
            },
            "retrieval": result.to_dict(), "elapsed_seconds": elapsed, "failure": failure,
            "metrics": score_retrieval(result, question["evidence_pages"], tree, search.trace),
            "usage": usage, "cost": estimate_query_cost(usage), "trace": search.trace, "request_trace": search.request_trace,
            "measurement_boundary": "query to returned evidence, including selection and sufficiency; excludes preprocessing, gold scoring and downstream answers",
        }
        output = args.output_dir / f"{case['question_id']}.{strategy}.json"
        with output.open("x") as stream:
            json.dump(record, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        print(json.dumps({"seconds": round(elapsed, 3), "jev_calls": usage.get("jev_requests", 0),
                          "sol_calls": usage.get("agent_calls", 0), "returned_pages": list(result.evidence_pages),
                          "gold_recall": record["metrics"]["returned_gold_page_recall"], "failure": failure}), flush=True)
        if failure and "authentication" in failure["type"].lower():
            break
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
