"""Run a frozen repeated cohort through the retrieval runner.

By default one retrieval runs at a time. `--parallel N` runs N cases at once to
finish sooner; each retrieval still sends its own requests one at a time, but
concurrent cases share the network and the API, so per-question times can rise
slightly. Each repeat shuffles the same question/policy pairs. Completed runs and failures are kept;
resume skips both and never selectively retries a poor result.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from retrieval.run import runtime_sources
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    files = runtime_sources()
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def make_cases(manifest):
    questions = manifest["questions"]
    if len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Duplicate question IDs")
    cases = []
    rng = random.Random(manifest["seed"])
    for repeat in range(1, manifest["repeats"] + 1):
        block = [{"repeat": repeat, "question_id": q["id"], "doc_id": q["doc_id"], "strategy": s}
                 for q in questions for s in manifest["strategies"]]
        rng.shuffle(block)
        for case in block:
            case["directory"] = f"r{repeat}/{case['question_id']}.{case['strategy']}"
            case["result"] = case["directory"] + f"/{case['question_id']}.{case['strategy']}.json"
        cases.extend(block)
    return cases


def validate_inputs(manifest, grounded_root):
    for doc in manifest["documents"]:
        folder = grounded_root / Path(doc["doc_id"]).stem / "grounded"
        for file, field in (("pages.jsonl", "pages_sha256"), ("toc_entries.json", "toc_sha256")):
            if sha(folder / file) != doc[field]:
                raise ValueError(f"Prepared input changed: {doc['doc_id']} / {file}")


def validate_record(record, plan, case):
    if record.get("experiment") != plan["experiment"]:
        raise ValueError("Wrong experiment version in result")
    if (record["question"]["id"], record["strategy"]) != (case["question_id"], case["strategy"]):
        raise ValueError("Result does not match scheduled case")
    if record["configuration"]["source_sha256"] != plan["source_sha256"]:
        raise ValueError("Result source hashes do not match frozen cohort")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, nargs="?", default=ROOT / "evaluation/cohort.json")
    parser.add_argument("--grounded-root", type=Path, default=ROOT / "runs/evaluation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strategies", nargs="+", choices=("jev_dfs", "exhaustive", "sol_only"),
                        help="Run only these approaches (default: all in the manifest)")
    parser.add_argument("--repeats", type=int, help="Run fewer repeats than the manifest (default: all)")
    parser.add_argument("--parallel", type=int, default=1, help="Cases to run at once (default 1)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.strategies:
        manifest["strategies"] = [s for s in manifest["strategies"] if s in args.strategies]
    if args.repeats:
        if not 1 <= args.repeats <= manifest["repeats"]:
            parser.error(f"--repeats must be between 1 and {manifest['repeats']}")
        manifest["repeats"] = args.repeats
    validate_inputs(manifest, args.grounded_root)
    plan = {"experiment": manifest["experiment"], "cohort": manifest["cohort"],
            "manifest_sha256": sha(args.manifest), "source_sha256": source_hashes(),
            "execution": f"{args.parallel} case(s) at a time, each sending one request at a time; independently shuffled question/policy pairs in each repeat block",
            "cases": make_cases(manifest)}
    if args.dry_run:
        print(json.dumps({"cases": len(plan["cases"]), "first": plan["cases"][:3]}, indent=2))
        return 0
    if args.resume:
        if json.loads((args.output_dir / "run-plan.json").read_text()) != plan:
            raise ValueError("Resume refused: plan, inputs or source code changed")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "run-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        (args.output_dir / "manifest.json").write_bytes(args.manifest.read_bytes())
        for name in plan["source_sha256"]:
            snapshot = args.output_dir / "source-snapshot" / name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes((ROOT / name).read_bytes())
    if args.parallel < 1:
        parser.error("--parallel must be positive")
    state = {"count": 0, "failed": 0, "stop": None}
    lock = threading.Lock()
    pending = []
    for index, case in enumerate(plan["cases"], 1):
        result_file = args.output_dir / case["result"]
        if result_file.exists():
            record = json.loads(result_file.read_text())
            validate_record(record, plan, case)
            state["count"] += 1
            state["failed"] += bool(record.get("failure"))
        else:
            pending.append((index, case))

    def run_case(item):
        index, case = item
        if state["stop"]:
            return
        if source_hashes() != plan["source_sha256"]:
            state["stop"] = "Runtime source changed while cohort was running"
            return
        folder = args.output_dir / case["directory"]
        result_file = args.output_dir / case["result"]
        if folder.exists():
            state["stop"] = f"Incomplete prior attempt at {folder}; keep it for audit and resolve explicitly"
            return
        folder.parent.mkdir(parents=True, exist_ok=True)
        log = folder.parent / (folder.name + ".log")
        print(f"RUN {index}/{len(plan['cases'])}: repeat {case['repeat']} {Path(case['doc_id']).stem} {case['question_id']} {case['strategy']}", flush=True)
        started = time.monotonic()
        with log.open("x") as stream:
            process = subprocess.run([
                sys.executable, "-m", "evaluation.run", str(args.manifest.resolve()),
                "--grounded-root", str(args.grounded_root.resolve()), "--question-ids", case["question_id"],
                "--strategies", case["strategy"], "--output-dir", str(folder.resolve()),
                "--max-navigation-steps", str(manifest["max_navigation_steps"]),
                "--openai-model", manifest["models"]["sol"],
            ], stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT, check=False)
        if not result_file.exists():
            state["stop"] = f"no result saved; inspect {log}"
            return
        record = json.loads(result_file.read_text())
        validate_record(record, plan, case)
        failure = record.get("failure")
        with lock:
            state["count"] += 1
            state["failed"] += bool(failure)
            progress = {"completed": state["count"], "planned": len(plan["cases"]), "api_failures": state["failed"],
                        "updated_at": datetime.now(timezone.utc).isoformat(), "last": case,
                        "last_elapsed": record["elapsed_seconds"], "last_recall": record["metrics"]["returned_gold_page_recall"],
                        "last_stop": record["retrieval"]["stop_reason"], "last_failure": failure,
                        "launcher_elapsed": time.monotonic() - started}
            (args.output_dir / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
            print(json.dumps(progress), flush=True)
        if failure and any(x in (failure["type"] + " " + failure["message"]).lower() for x in ("authentication", "unauthorized", "invalid_api_key", "insufficient_quota")):
            state["stop"] = "credential/quota error; completed results preserved."
        elif process.returncode and not failure:
            state["stop"] = "child failed outside measured retrieval; inspect log."

    with ThreadPoolExecutor(args.parallel) as pool:
        list(pool.map(run_case, pending))
    if state["stop"]:
        print(f"STOP: {state['stop']}", flush=True)
        return 1
    count, failed = state["count"], state["failed"]
    print(json.dumps({"complete": True, "runs": count, "api_failures": failed}), flush=True)
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
