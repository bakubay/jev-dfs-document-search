"""Lossless trace projection and hash-verified source snippets. No model calls."""
import ast
from collections import defaultdict
import hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

SPECS = {'route': ('retrieval/tree.py', 'OrderedDFSSearcher._rank_targets'),
 'dfs': ('retrieval/jev.py', 'JevDFSRetriever._scan_leaf'),
 'exhaustive': ('retrieval/jev.py', 'JevExhaustiveRetriever.search'),
 'windows': ('retrieval/jev.py', 'JevDFSRetriever._check_combined_windows'),
 'judgments': ('retrieval/jev.py', 'JevDFSRetriever._check_combined_window'),
 'pack': ('retrieval/jev.py', 'JevDFSRetriever._pack_evidence'),
 'assess': ('retrieval/jev.py', 'JevDFSRetriever._assess_evidence'),
 'audit': ('retrieval/jev.py', 'JevDFSRetriever.search'),
 'sol': ('retrieval/sol.py', 'SolRetriever.search'),
 'sol_call': ('retrieval/sol.py', 'act_with_sol'),
 'read': ('retrieval/sol.py', 'SolRetriever._read'),
 'select': ('retrieval/sol.py', 'SolRetriever._select'),
 'tree': ('retrieval/tree.py', 'build_toc_tree'),
 'timer': ('retrieval/run.py', 'run_retrieval'),
 'jev_timer': ('retrieval/tree.py', 'OrderedDFSSearcher._system_one'),
 'runner': ('evaluation/run.py', 'main'),
 'metrics': ('evaluation/metrics.py', 'score_retrieval'),
 'limits': ('retrieval/windows.py', 'pack_page_windows'),
 'backtrack': ('retrieval/tree.py', 'OrderedDFSSearcher._search_target'),
 'navigate': ('retrieval/tree.py', 'OrderedDFSSearcher._search_node')}

# File/symbol map for recorded pre-cleanup runs; no legacy runtime is imported.
RECORDED_SPECS = {'route': ('datalab_dfs_search.py', 'OrderedDFSSearcher._rank_targets'),
 'dfs': ('retrieval/combined.py', 'CombinedDFSRetriever._scan_leaf'),
 'exhaustive': ('retrieval/combined.py', 'CombinedExhaustiveRetriever.search'),
 'windows': ('retrieval/combined.py', 'CombinedDFSRetriever._check_combined_windows'),
 'judgments': ('retrieval/combined.py', 'CombinedDFSRetriever._check_combined_window'),
 'pack': ('retrieval/jev.py', 'JevDFSRetriever._pack_evidence'),
 'assess': ('retrieval/jev.py', 'JevDFSRetriever._assess_evidence'),
 'audit': ('retrieval/jev.py', 'JevDFSRetriever.search'),
 'sol': ('retrieval/sol.py', 'SolRetriever.search'),
 'sol_call': ('retrieval/sol.py', 'act_with_sol'),
 'read': ('evaluation/sol_only_agent.py', 'SolOnlySearcher._read'),
 'select': ('retrieval/sol.py', 'SolRetriever._select'),
 'tree': ('datalab_dfs_search.py', 'build_toc_tree'),
 'timer': ('retrieval/run_case.py', 'run_retrieval'),
 'jev_timer': ('datalab_dfs_search.py', 'OrderedDFSSearcher._system_one'),
 'runner': ('experiments/run_combined.py', 'main'),
 'metrics': ('retrieval/metrics.py', 'score_retrieval'),
 'limits': ('page_windows.py', 'pack_page_windows'),
 'backtrack': ('datalab_dfs_search.py', 'OrderedDFSSearcher._search_target'),
 'navigate': ('datalab_dfs_search.py', 'OrderedDFSSearcher._search_node')}

def snippet(file, symbol, root=ROOT):
    text = (root / file).read_text()
    node = ast.parse(text)
    for name in symbol.split("."):
        node = next(n for n in node.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
    return {"file": file, "symbol": symbol, "line": node.lineno,
            "text": "\n".join(text.splitlines()[node.lineno - 1:node.end_lineno]),
            "sha256": hashlib.sha256(text.encode()).hexdigest()}

def recorded_sources(hashes, folder):
    """Never attach today's code to yesterday's trace without matching hashes."""
    candidates = [ROOT, folder / "source-snapshot", folder.parent / "source-snapshot"]
    for root in candidates:
        if all((root / name).is_file() and hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
               for name, digest in hashes.items()):
            specs = RECORDED_SPECS if "retrieval/combined.py" in hashes else SPECS
            return {key: snippet(*spec, root=root) for key, spec in specs.items()}
    raise ValueError("Recorded source differs or is unavailable. Keep the original source-snapshot beside the run folder.")


def union_duration(intervals):
    total, end = 0., -float("inf")
    for a, b in sorted(intervals):
        total += max(0., b - max(a, end))
        end = max(end, b)
    return total

def lane_layout(calls):
    ends = []
    for call in sorted(calls, key=lambda c: c["start"]):
        lane = next((i for i, end in enumerate(ends) if end <= call["start"]), len(ends))
        if lane == len(ends):
            ends.append(call["end"])
        else:
            ends[lane] = call["end"]
        call["lane"] = lane
    return len(ends)

def derive_run(record):
    trace = sorted(record["trace"], key=lambda e: e["elapsed_seconds"])
    pending, completed = defaultdict(list), defaultdict(list)
    for e in trace:
        if e["type"] == "retrieval_request_started":
            pending[(e["purpose"], tuple(sorted(e["questions"])))].append(e)
        elif e["type"] == "retrieval_request_completed":
            completed[(e["purpose"], tuple(sorted(e["output"])))].append(e)
    calls = []
    if record["strategy"] == "sol_only":
        starts = [e for e in trace if e["type"] == "retrieval_agent_call_started"]
        finishes = [e for e in trace if e["type"] == "retrieval_agent_decision"]
        prompt = next(e for e in trace if e["type"] == "retrieval_agent_started")
        for start, finish, req in zip(starts, finishes, record["request_trace"], strict=True):
            assert start["step"] == finish["step"]
            action = (finish["output"] or {}).get("action", "unavailable")
            calls.append({"purpose": "sol_agent", "start": start["elapsed_seconds"],
                          "end": finish["elapsed_seconds"], "duration": req["latency_seconds"],
                          "input": {"system_prompt": prompt["prompt"], "state": start["input"],
                                    "output_schema": prompt["output_schema"]},
                          "output": finish["output"], "title": f"Sol → {action}", "code": "sol",
                          "path": start["input"]["current_path"],
                          "phase": "select" if action in ("finish", "stop") else "read" if start["input"]["raw_page_window"] else "route",
                          "usage": req})
    else:
        for req in record["request_trace"]:
            key = req["purpose"], tuple(sorted(req["question_ids"]))
            start, finish = pending[key].pop(0), completed[key].pop(0)
            purpose = req["purpose"]
            phase = "route" if purpose == "rank_toc_children" else "select" if purpose == "assess_retrieved_evidence" else "read"
            title = {"rank_toc_children": "JEV → rank branches", "assess_retrieved_evidence": "JEV → enough evidence?",
                     "screen_page_window": "JEV → screen pages", "verify_page_window": "JEV → verify pages",
                     "check_page_window": "JEV → check pages once"}[purpose]
            state = start["input"]
            calls.append({"purpose": purpose, "start": req["elapsed_seconds"] - req["latency_seconds"],
                          "end": req["elapsed_seconds"], "event_time": finish["elapsed_seconds"],
                          "duration": req["latency_seconds"],
                          "input": {"state": state, "questions": start["questions"]},
                          "output": finish["output"], "title": title, "phase": phase,
                          "code": "route" if phase == "route" else "assess" if phase == "select" else "judgments",
                          "path": state.get("current_toc_path", []), "usage": req})
        assert not any(pending.values()) and not any(completed.values())
    calls.sort(key=lambda c: c["end"])
    for i, c in enumerate(calls):
        c["id"] = i
    peak = lane_layout(calls)
    groups = defaultdict(list)
    for c in calls:
        groups[c["purpose"]].append([c["start"], c["end"]])
    timing = {k: {"calls": len(v), "active_seconds": union_duration(v),
                  "summed_seconds": sum(b-a for a, b in v)} for k, v in groups.items()}
    network = union_duration([iv for v in groups.values() for iv in v])
    timing["outside_calls"] = {"calls": 0, "active_seconds": max(0., record["elapsed_seconds"] - network)}
    # Build replay frames from observed state changes. Concurrent requests retain
    # their actual intervals; the event rail is explicitly completion ordered.
    steps = [{"time": 0., "title": "Question + document", "phase": "input", "code": "runner",
              "input": {"question": record["question"]["question"], "document": record["question"]["doc_id"]},
              "output": {"read_limits": {"pages": record["configuration"]["max_pages_per_read"], "tokens": record["configuration"]["max_read_tokens"]},
                         "output_budget": record["configuration"]["output_budget"]}, "path": []}]
    for c in calls:
        steps.append({"time": c.get("event_time", c["end"]), "call_id": c["id"], "title": c["title"],
                      "phase": c["phase"], "code": c["code"], "path": c["path"],
                      "input": c["input"], "output": c["output"]})
    mapping = {
        "scan_started": ("Open section", "read", "dfs"),
        "backtrack": ("Backtrack", "route", "audit"),
        "evidence_selected": ("Select evidence pages", "select", "pack" if record["strategy"] != "sol_only" else "select"),
        "evidence_assessed": ("Evidence decision", "select", "assess"),
        "sol_pages_loaded": ("Load source pages", "read", "read"),
    }
    for e in trace:
        if e["type"] in mapping:
            title, phase, code = mapping[e["type"]]
            steps.append({"time": e["elapsed_seconds"], "title": title, "phase": phase, "code": code,
                          "input": {"path": e.get("path", [])}, "output": e, "path": e.get("path", [])})
    steps.sort(key=lambda s: s["time"])
    steps.append({"time": record["elapsed_seconds"], "title": "Return source evidence · stop timer", "phase": "return", "code": "timer",
                  "input": {"question": record["question"]["question"]}, "output": record["retrieval"], "path": []})
    steps.append({"time": record["elapsed_seconds"], "title": "Evaluate returned evidence · outside timer", "phase": "return", "code": "metrics",
                  "input": {"gold_pages": record["question"]["evidence_pages"], "returned_pages": list(record["retrieval"]["evidence_pages"])},
                  "output": record["metrics"], "path": []})
    for step in steps:
        read, verified, selected, menus, path = set(), set(), [], {}, []
        for e in trace:
            if e["elapsed_seconds"] > step["time"]:
                break
            if e["type"] == "jev_window_completed":
                read.update(e["source_pages"])
                if e["phase"] in ("verify", "combined"):
                    verified.update(e["target_pages"])
            elif e["type"] == "sol_read_completed":
                read.update(e["pages"])
            elif e["type"] == "evidence_selected":
                selected = e["pages"]
            elif e["type"] == "choice":
                menus[" → ".join(e["path"])] = e["ranking"]
            if e["type"] in ("scan_started", "sol_pages_loaded", "backtrack"):
                path = e.get("path", [])
        step["read"] = sorted(read)
        step["verified"] = sorted(verified)
        step["selected"] = selected
        step["path"] = step["path"] or path
        step["menus"] = menus
    windows = [e for e in trace if e["type"] == "jev_window_completed"]
    batches = {}
    for phase in ("screen", "verify", "combined"):
        ws = [w for w in windows if w["phase"] == phase]
        batches[phase] = {"calls": len(ws), "targets": sum(len(w["target_pages"]) for w in ws),
                          "source_page_appearances": sum(len(w["source_pages"]) for w in ws),
                          "single_target_windows": sum(len(w["target_pages"]) == 1 for w in ws)}
    return {"calls": calls, "steps": steps, "timing": timing, "peak_concurrency": peak,
            "batches": batches, "record": record}
