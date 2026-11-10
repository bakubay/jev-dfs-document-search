"""Exact payload joins, source provenance, and concurrent timing."""
import hashlib
import json
import pytest
from reporting.trace_data import ROOT, derive_run, snippet, union_duration, recorded_sources

@pytest.fixture(scope="module")
def records():
    paths = sorted((ROOT / "evaluation-results/retrieval-v3-one-judgment-2026-11-09/r1").glob("*/*.json"))
    records = [json.loads(p.read_text()) for p in paths if p.name != "schedule.json"]
    if not records:
        pytest.skip("Live records are local, not vendored")
    return records


def test_overlap_is_counted_once():
    assert union_duration([(0, 4), (1, 2), (3, 8), (9, 10)]) == 9


def test_every_recorded_call_is_preserved_with_matching_questions(records):
    for record in records:
        result = derive_run(record)
        assert len(result["calls"]) == len(record["request_trace"])
        for call in result["calls"]:
            if record["strategy"] != "sol_only":
                assert set(call["input"]["questions"]) == set(call["output"])
                source = [e for e in record["trace"] if e["type"] == "retrieval_request_started"
                          and e["purpose"] == call["purpose"] and e["input"] == call["input"]["state"]]
                assert source
        assert abs(sum(t["active_seconds"] for t in result["timing"].values()) - record["elapsed_seconds"]) < .001
        last = result["steps"][-1]
        assert set(map(int, last["selected"])) == set(map(int, record["retrieval"]["evidence_pages"]))
        assert last["read"] == record["retrieval"]["coverage"]["read_pages"]


def test_concurrent_completion_does_not_swap_page_responses(records):
    for record in records:
        if record["strategy"] == "sol_only":
            continue
        for call in derive_run(record)["calls"]:
            if "page_window" not in call["purpose"]:
                continue
            matches = [e for e in record["trace"] if e["type"] == "jev_window_completed"
                       and set(e["output"]) == set(call["output"])]
            assert len(matches) == 1
            assert {k: v["noul"] for k, v in call["output"].items()} == matches[0]["output"]


def test_observed_concurrency_is_not_configured_worker_count(records):
    for record in records:
        result = derive_run(record)
        if record["strategy"] == "exhaustive":
            assert result["peak_concurrency"] == 8
        elif record["strategy"] == "sol_only":
            assert result["peak_concurrency"] == 1


def test_source_snippet_is_exact_not_rewritten():
    s = snippet("retrieval/jev.py", "JevDFSRetriever._scan_leaf")
    text = (ROOT / s["file"]).read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == s["sha256"]
    assert "\n".join(text.splitlines()[s["line"]-1:s["line"]-1+len(s["text"].splitlines())]) == s["text"]


def test_source_mismatch_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Recorded source differs"):
        recorded_sources({"retrieval/jev.py": "wrong"}, tmp_path)
