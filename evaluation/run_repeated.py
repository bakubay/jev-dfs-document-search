"""Run a frozen repeated cohort through the retrieval runner.

One retrieval at a time avoids cross-case resource contention. Each repeat
shuffles the same question/policy pairs. Completed runs and failures are kept;
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
import time

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
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--grounded-root", type=Path, default=ROOT / "runs/evaluation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    validate_inputs(manifest, args.grounded_root)
    plan = {"experiment": manifest["experiment"], "cohort": manifest["cohort"],
            "manifest_sha256": sha(args.manifest), "source_sha256": source_hashes(),
            "execution": "one query at a time; independently shuffled question/policy pairs in each repeat block",
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
    count = failed = 0
    for index, case in enumerate(plan["cases"], 1):
        result_file = args.output_dir / case["result"]
        if result_file.exists():
            record = json.loads(result_file.read_text())
            validate_record(record, plan, case)
            count += 1
            failed += bool(record.get("failure"))
            continue
        if source_hashes() != plan["source_sha256"]:
            raise ValueError("Runtime source changed while cohort was running")
        folder = args.output_dir / case["directory"]
        if folder.exists():
            raise ValueError(f"Incomplete prior attempt at {folder}; keep it for audit and resolve explicitly")
        folder.parent.mkdir(parents=True, exist_ok=True)
        log = folder.parent / (folder.name + ".log")
        print(f"RUN {index}/{len(plan['cases'])}: repeat {case['repeat']} {Path(case['doc_id']).stem} {case['question_id']} {case['strategy']}", flush=True)
        started = time.monotonic()
        with log.open("x") as stream:
            process = subprocess.run([
                sys.executable, "-m", "evaluation.run", str(args.manifest.resolve()),
                "--grounded-root", str(args.grounded_root.resolve()), "--question-ids", case["question_id"],
                "--strategies", case["strategy"], "--output-dir", str(folder.resolve()),
                "--workers", str(manifest["workers"]), "--max-navigation-steps", str(manifest["max_navigation_steps"]),
                "--openai-model", manifest["models"]["sol"],
            ], stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT, check=False)
        if not result_file.exists():
            print(f"STOP: no result saved; inspect {log}", flush=True)
            return 1
        record = json.loads(result_file.read_text())
        validate_record(record, plan, case)
        count += 1
        failure = record.get("failure")
        failed += bool(failure)
        progress = {"completed": count, "planned": len(plan["cases"]), "api_failures": failed,
                    "updated_at": datetime.now(timezone.utc).isoformat(), "last": case,
                    "last_elapsed": record["elapsed_seconds"], "last_recall": record["metrics"]["returned_gold_page_recall"],
                    "last_stop": record["retrieval"]["stop_reason"], "last_failure": failure,
                    "launcher_elapsed": time.monotonic() - started}
        (args.output_dir / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
        print(json.dumps(progress), flush=True)
        if failure and any(x in (failure["type"] + " " + failure["message"]).lower() for x in ("authentication", "unauthorized", "invalid_api_key", "insufficient_quota")):
            print("STOP: credential/quota error; completed results preserved.", flush=True)
            return 1
        if process.returncode and not failure:
            print("STOP: child failed outside measured retrieval; inspect log.", flush=True)
            return 1
    print(json.dumps({"complete": True, "runs": count, "api_failures": failed}), flush=True)
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
