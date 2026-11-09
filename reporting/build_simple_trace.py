"""Literal call-by-call trace from three saved retrieval runs. No API calls."""

import argparse
import hashlib
import json
from pathlib import Path

from reporting.trace_data import derive_run, recorded_sources

ROOT = Path(__file__).resolve().parents[1]
QID = "pi-oss-b728c1d683c6"
FOLDER = ROOT / "evaluation-results/retrieval-repeated-2026-11-08/r1"
METHODS = {"jev_dfs": "JEV DFS", "exhaustive": "Exhaustive JEV", "sol_only": "Sol agent"}
LOCAL = {"attempt", "scan_started", "backtrack", "evidence_selected", "evidence_assessed", "sol_pages_loaded", "sol_read_completed", "retrieval_action_rejected"}


def build_data():
    runs = {}
    for method in METHODS:
        path = FOLDER / f"{QID}.{method}" / f"{QID}.{method}.json"
        record = json.loads(path.read_text())
        assert record["failure"] is None
        sources = recorded_sources(record["configuration"]["source_sha256"], FOLDER)
        derived = derive_run(record)
        calls = sorted(derived["calls"], key=lambda c: c["start"])
        trace = record["trace"]
        windows = [e for e in trace if e["type"] == "jev_window_completed"]
        pages = {}
        for i, call in enumerate(calls):
            call["number"] = i + 1
            state = call["input"]["state"]
            for key in ("pages", "source_pages", "raw_page_window", "selected_evidence"):
                pages.update(state.get(key, {}))
            next_start = calls[i + 1]["start"] if i + 1 < len(calls) else record["elapsed_seconds"] + .001
            if method == "exhaustive" and call["purpose"] == "check_page_window":
                call["actions"] = []
            else:
                call["actions"] = [e for e in trace if e["type"] in LOCAL and call["end"] <= e["elapsed_seconds"] < next_start]
            if method == "sol_only":
                deliveries = [e for e in trace if e["type"] == "sol_read_completed" and call["start"] <= e["elapsed_seconds"] < call["end"]]
                call["actions"] = deliveries + call["actions"]
            if call["purpose"] == "check_page_window":
                ids = set(call["output"])
                window = next(e for e in windows if set(e["output"]) == ids)
                call["window"] = {k: v for k, v in window.items() if k not in ("type", "usage", "model")}
                assert set(state["pages"]) == {str(p) for p in window["source_pages"]}
            if call["purpose"] == "rank_toc_children":
                ranking = next(e for e in trace if e["type"] == "choice" and e["elapsed_seconds"] >= call["end"])
                call["ranking"] = ranking["ranking"]
            if call["purpose"] == "assess_retrieved_evidence":
                earlier = [e for e in trace if e["type"] == "evidence_selected" and e["elapsed_seconds"] <= call["start"]]
                call["selection_before"] = earlier[-1] if earlier else None
            call["is_last"] = i == len(calls) - 1
        runs[method] = {"name": METHODS[method], "calls": calls, "seconds": record["elapsed_seconds"],
                        "configuration": record["configuration"], "question": record["question"]["question"],
                        "document": record["question"]["doc_id"], "recorded_at": record["recorded_at"],
                        "result": record["retrieval"], "returned_page_ids": list(record["retrieval"]["evidence_pages"]), "pages": pages, "peak": derived["peak_concurrency"],
                        "record_file": str(path.relative_to(ROOT)), "record_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "trace": trace}
    return {"runs": runs, "sources": sources, "question_id": QID}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/traces/jev-simple-execution-trace.html")
    args = parser.parse_args()
    data = build_data()
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    template = (ROOT / "reporting/simple_trace.html").read_text()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(template.replace("__TRACE_DATA__", encoded))
    print(json.dumps({"output": str(args.output), "calls": {k: len(v["calls"]) for k, v in data["runs"].items()}, "bytes": args.output.stat().st_size, "new_api_calls": 0}))


if __name__ == "__main__":
    main()
