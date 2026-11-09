"""Offline repeated-cohort report with one compressed, inspectable payload per run."""

import argparse
import base64
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean, median
from zoneinfo import ZoneInfo

from reporting.trace_data import ROOT, derive_run, recorded_sources

NAMES = {"jev_dfs": "JEV DFS", "exhaustive": "Exhaustive JEV", "sol_only": "Sol agent"}


class TextPool:
    """Lossless storage: repeated source pages/prompts are embedded only once."""
    def __init__(self):
        self.values, self.indices = [], {}

    def encode(self, value):
        if isinstance(value, str) and len(value) >= 96:
            if value not in self.indices:
                self.indices[value] = len(self.values)
                self.values.append(value)
            return {"$text": self.indices[value]}
        if isinstance(value, dict):
            return {k: self.encode(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.encode(v) for v in value]
        return value


def compress(value):
    packed = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    return base64.b64encode(gzip.compress(packed, compresslevel=9, mtime=0)).decode()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def failure_view(record):
    """Keep failed attempts explicit, without inventing a successful response join."""
    events = sorted(record["trace"], key=lambda e: e["elapsed_seconds"])
    steps = [{"time": 0, "title": "Question + document", "phase": "input", "code": "runner",
              "input": {"question": record["question"]["question"]}, "output": {}, "path": []}]
    for e in events:
        if e["type"] not in {"retrieval_request_started", "retrieval_request_completed", "retrieval_agent_call_started", "retrieval_agent_decision", "evidence_selected", "evidence_assessed", "backtrack"}:
            continue
        purpose = e.get("purpose", "")
        phase = "route" if purpose == "rank_toc_children" or e["type"] == "backtrack" else "select" if "evidence" in purpose or e["type"].startswith("evidence_") else "read"
        steps.append({"time": e["elapsed_seconds"], "title": e["type"].replace("_", " "), "phase": phase,
                      "code": "sol" if record["strategy"] == "sol_only" else "route" if phase == "route" else "assess" if phase == "select" else "judgments",
                      "input": e.get("input", {}), "output": e.get("output", e), "path": e.get("path", [])})
    steps.append({"time": record["elapsed_seconds"], "title": "Execution error · partial evidence retained", "phase": "return", "code": "timer",
                  "input": record["failure"], "output": record["retrieval"], "path": []})
    for step in steps:
        read, verified, selected = set(), set(), []
        for e in events:
            if e["elapsed_seconds"] > step["time"]:
                break
            if e["type"] == "jev_window_completed":
                read.update(e["source_pages"])
                verified.update(e["target_pages"])
            elif e["type"] == "sol_read_completed":
                read.update(e["pages"])
            elif e["type"] == "evidence_selected":
                selected = e["pages"]
        step.update(read=sorted(read), verified=sorted(verified), selected=selected, menus={})
    return {"steps": steps, "calls": [], "timing": {}, "peak_concurrency": None, "batches": {},
            "failure_notice": "Failed attempts are shown as separate recorded events. Request intervals may be incomplete; no success-only timing decomposition is claimed."}


def summarize_rows(rows):
    """Question is the unit; repeats measure variation, not extra questions."""
    result = []
    for strategy, label in NAMES.items():
        chosen = [r for r in rows if r["strategy"] == strategy]
        if not chosen:
            continue
        groups = defaultdict(list)
        for row in chosen:
            groups[row["question_id"]].append(row)
        q_medians = [median(r["seconds"] for r in rs if not r["failure"]) for rs in groups.values() if any(not r["failure"] for r in rs)]
        result.append({"strategy": strategy, "name": label, "questions": len(groups), "runs": len(chosen),
                       "median_question_seconds": median(q_medians) if q_medians else None,
                       "mean_recall": mean(mean(r["recall"] for r in rs) for rs in groups.values()),
                       "full_recall_runs": sum(r["recall"] == 1 for r in chosen),
                       "median_question_calls": median(median(r["calls"] for r in rs) for rs in groups.values()),
                       "mean_cost": mean(mean(r["cost"] for r in rs) for rs in groups.values()),
                       "total_cost": sum(r["cost"] for r in chosen),
                       "failures": sum(bool(r["failure"]) for r in chosen),
                       "stops": dict(Counter(r["stop"] for r in chosen))})
    return result


def load_cohort(folder, allow_partial=False):
    manifest = json.loads((folder / "manifest.json").read_text())
    plan = json.loads((folder / "run-plan.json").read_text())
    if digest(folder / "manifest.json") != plan["manifest_sha256"]:
        raise ValueError("Frozen manifest was modified")
    sources = recorded_sources(plan["source_sha256"], folder)
    rows, payloads, missing = [], {}, []
    pool = TextPool()
    questions = {q["id"]: q for q in manifest["questions"]}
    documents = {d["doc_id"]: d for d in manifest["documents"]}
    for index, case in enumerate(plan["cases"]):
        path = folder / case["result"]
        if not path.is_file():
            missing.append(case)
            continue
        record = json.loads(path.read_text())
        conf = record["configuration"]
        if record["experiment"] != manifest["experiment"] or conf["source_sha256"] != plan["source_sha256"]:
            raise ValueError("Mixed experiment/runtime versions")
        if record["question"] != questions[case["question_id"]] or record["strategy"] != case["strategy"]:
            raise ValueError("Result does not match frozen question/strategy")
        doc = documents[case["doc_id"]]
        for key, expected in (("grounded_pages_sha256", doc["pages_sha256"]), ("toc_sha256", doc["toc_sha256"]),
                              ("max_pages_per_read", manifest["max_pages_per_read"]), ("max_read_tokens", manifest["max_read_tokens"]),
                              ("output_budget", manifest["output_budget"]), ("workers", manifest["workers"])):
            if conf[key] != expected:
                raise ValueError(f"Mixed input or budget: {key}")
        sol = case["strategy"] == "sol_only"
        if conf["sol_model" if sol else "jev_model"] != manifest["models"]["sol" if sol else "jev"]:
            raise ValueError("Mixed model versions")
        view = failure_view(record) if record["failure"] else derive_run(record)
        view.pop("record", None)
        # No raw payload is fetched by the HTML. Only the selected run is inflated.
        key = f"run-{index + 1}"
        payloads[key] = compress(pool.encode({"record": record, "view": view}))
        metrics, usage, coverage = record["metrics"], record["usage"], record["retrieval"]["coverage"]
        rows.append({"id": key, **case, "seconds": record["elapsed_seconds"],
                     "calls": usage.get("jev_requests", 0) + usage.get("agent_calls", 0),
                     "jev_calls": usage.get("jev_requests", 0), "sol_calls": usage.get("agent_calls", 0),
                     "cost": record["cost"]["estimated_query_cost_usd"], "recall": metrics["returned_gold_page_recall"],
                     "read_recall": metrics["read_gold_page_recall"], "section_recall": metrics["returned_gold_section_recall"],
                     "read": len(coverage["read_pages"]), "checked": len(coverage["verified_pages"]),
                     "returned": list(map(int, record["retrieval"]["evidence_pages"])),
                     "stop": record["retrieval"]["stop_reason"], "sufficient": record["retrieval"]["sufficient"],
                     "failure": record["failure"], "peak": view["peak_concurrency"],
                     "first_gold_seconds": metrics["time_to_first_gold_page_read_seconds"],
                     "frames": len(view["steps"]), "record_sha256": digest(path),
                     "recorded_at": record["recorded_at"], "timing": view["timing"], "batches": view["batches"]})
    if missing and not allow_partial:
        raise ValueError(f"Missing {len(missing)} planned runs; use --allow-partial only for a clearly labeled work-in-progress preview")
    review_path = ROOT / "docs/results" / manifest["cohort"] / "evidence-review.json"
    review = json.loads(review_path.read_text()) if review_path.is_file() else {"reviews": []}
    for item in review["reviews"]:
        page_file = ROOT / "runs/evaluation" / Path(item["doc_id"]).stem / "grounded/pages.jsonl"
        texts = {r["pdf_page_number"]: r["markdown"] for r in map(json.loads, page_file.read_text().splitlines())}
        for excerpt in item["source_excerpts"]:
            text = texts[excerpt["page"]]
            if hashlib.sha256(text.encode()).hexdigest() != excerpt["page_text_sha256"] or excerpt["excerpt"] not in text:
                raise ValueError("Evidence-review excerpt differs from the source page")
    return {"built_on": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(), "manifest": manifest,
            "rows": rows, "payloads": payloads, "text_pool": compress(pool.values), "missing": missing, "planned": len(plan["cases"]),
            "aggregate": summarize_rows(rows),
            "by_document": {d: summarize_rows([r for r in rows if r["doc_id"] == d]) for d in documents},
            "evidence_review": review,
            "sources": sources}


def write_summary(data, destination):
    rows = data["rows"]
    summary = {k: v for k, v in data.items() if k not in ("payloads", "sources", "text_pool")}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.with_suffix(".json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    lines = ["# Repeated retrieval experiment: four PDFs, twenty questions", "",
             f"Completed **{len(rows)}/{data['planned']}** planned runs. Three repetitions per question and policy.", "",
             "The four PDFs come from PageIndex-OSS-Benchmark. Eight questions are from its text-lookup subset; twelve additional existing annotations come from the pinned MMLongBench-Doc-V2 source dataset. These include table, enumeration and derived-answer questions. We score retrieval of the annotated pages, not answer calculation.", "",
             "| Policy | Median of question-median time | Mean annotated-page recall | All annotated pages returned | Median calls per question | Mean estimated cost | API failures |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in data["aggregate"]:
        t = f"{r['median_question_seconds']:.2f}s" if r["median_question_seconds"] is not None else "—"
        lines.append(f"| {r['name']} | {t} | {r['mean_recall']:.1%} | {r['full_recall_runs']}/{r['runs']} | {r['median_question_calls']:g} | ${r['mean_cost']:.5f} | {r['failures']} |")
    lines += ["", "Time is query to returned full-page evidence, including routing and stopping decisions. Preprocessing, ground-truth scoring and downstream answer generation are excluded. Error-free retrievals contribute to latency medians; all recorded attempts contribute to recall and failure counts. Step/context-limit stops remain visible and are not silently removed. Cost is estimated from the existing pinned rates; failed-request usage can be unavailable.", "",
              "## Per document", "", "| Document | Policy | Median question time | Mean page recall | Full recall runs |", "|---|---|---:|---:|---:|"]
    for doc, records in data["by_document"].items():
        for r in records:
            t = f"{r['median_question_seconds']:.2f}s" if r["median_question_seconds"] is not None else "—"
            lines.append(f"| {doc} | {r['name']} | {t} | {r['mean_recall']:.1%} | {r['full_recall_runs']}/{r['runs']} |")
    lines += ["", "## Scope and reproducibility", "",
              "- Same prepared page text, grounded ToC and 8-page / 12k-token read and output limits. Neighbor pages count toward limits.",
              "- JEV uses the frozen four-judgment combined-window implementation. The two-judgment proposal was not applied.",
              "- Each repeat shuffles all question/policy pairs; queries run sequentially. Exhaustive has eight internal window workers; DFS retains leaf-local scheduling. Sol navigates alone, with no JEV or answer writer.",
              "- Original ToC hierarchy is retained. This is a convenience cohort of three financial reports and one manual, not a representative sample of PDFs.",
              "- All twenty questions are answerable. This cohort does not establish absent-evidence reliability, image understanding or correctness on arbitrary PDFs.",
              "- Exact annotated-page recall can undercount valid evidence on alternative/repeated pages. Full recall is not answer accuracy. Repetitions are not additional independent questions.",
              "- Client-observed time includes network/provider/SDK time. No region-controlled or server-only timing is available.",
              "- This was not an isolated benchmark host: local report development/validation occurred during collection. Queries themselves ran one at a time.",
              "- The report embeds every completed run with exact requests, parsed responses, source code and a downloadable JSON record. Missing attempts, if any, are explicitly listed.", "",
              "Sources: [PageIndex-OSS-Benchmark](https://github.com/VectifyAI/PageIndex-OSS-Benchmark), [MMLongBench-Doc-V2](https://github.com/VectifyAI/MMLongBench-Doc-V2). Both revisions and input hashes are pinned in manifest.json.", ""]
    lines += ["## Review annotation misses before drawing quality conclusions", "",
              "The original scores above are unchanged. Source inspection found repeated or alternative evidence for several questions. This review is by the assistant, not independent human adjudication; it does not provide an adjusted accuracy score.", ""]
    for review in data["evidence_review"]["reviews"]:
        q = next(q for q in data["manifest"]["questions"] if q["id"] == review["question_id"])
        lines += [f"- **{q['doc_id']} / {review['kind']}:** {review['note']}"]
    lines += ["", "Open the standalone HTML's source-review details to inspect excerpts and the original run payloads.", ""]
    destination.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    data = load_cohort(args.runs, args.allow_partial)
    template = Path(__file__).with_name("cohort_report.html").read_text()
    serialized = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(template.replace("__COHORT_DATA__", serialized))
    write_summary(data, args.summary)
    print(json.dumps({"completed": len(data["rows"]), "planned": data["planned"], "html_bytes": args.output.stat().st_size,
                      "frames": sum(r["frames"] for r in data["rows"])}))


if __name__ == "__main__":
    main()
