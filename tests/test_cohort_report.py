"""Report storage is lossless and repeated measurements retain their denominator."""

import base64
from copy import deepcopy
import gzip
import json
from pathlib import Path

import pytest

from reporting.build_cohort_report import TextPool, compress, failure_view, load_cohort, summarize_rows

ROOT = Path(__file__).parents[1]


def restore(value, pool):
    if isinstance(value, dict):
        if list(value) == ["$text"]:
            return pool[value["$text"]]
        return {k: restore(v, pool) for k, v in value.items()}
    if isinstance(value, list):
        return [restore(v, pool) for v in value]
    return value


def test_shared_source_text_round_trip_is_exact():
    text = "<script>literal document data</script>\nCoffee → pages € " * 40
    raw = {"pages": {"1": text, "2": text}, "nested": [{"source": text, "probability": .83}], "none": None}
    pool = TextPool()
    packed = compress(pool.encode(raw))
    assert len(pool.values) == 1
    decoded = json.loads(gzip.decompress(base64.b64decode(packed)))
    assert restore(decoded, pool.values) == raw


def test_repeats_do_not_become_independent_questions_and_failures_remain():
    rows = [{"question_id": q, "strategy": "jev_dfs", "seconds": t, "calls": 4, "cost": .01,
             "recall": recall, "failure": failed, "stop": "error" if failed else "sufficient_evidence",
             "returned": returned, "precision": prec, "read": 10}
            for q, t, recall, failed, returned, prec in [
                ("a", 1, 1, None, [1, 2], .5), ("a", 2, 1, None, [1, 2], .5), ("a", 3, 1, None, [1, 2], .5),
                ("b", 10, 0, {"type": "APIError"}, [], None)]]
    summary = summarize_rows(rows)[0]
    assert summary["questions"] == 1 and summary["runs"] == 4  # b only failed: no evidence about quality
    assert summary["mean_recall"] == 1
    assert summary["median_question_seconds"] == 2  # Error duration is not a success latency.
    assert summary["full_recall_runs"] == 3 and summary["failures"] == 1
    assert summary["mean_returned_pages"] == 2
    assert summary["mean_precision"] == .5
    assert summary["mean_read_pages"] == 10


def test_combining_folders_takes_each_approach_from_one_folder(tmp_path):
    folder = ROOT / "evaluation-results/retrieval-v3-one-judgment-2026-11-09"
    if not (folder / "run-plan.json").is_file():
        pytest.skip("Repeated live records are local")
    data = load_cohort([(folder, {"sol_only"}), (folder, {"jev_dfs"})], allow_partial=True)
    assert {r["strategy"] for r in data["rows"]} == {"sol_only", "jev_dfs"}
    assert len(set(data["payloads"])) == len(data["rows"])
    with pytest.raises(ValueError, match="appears in both"):
        load_cohort([(folder, {"sol_only"}), (folder, {"sol_only"})], allow_partial=True)


@pytest.fixture(scope="module")
def local_data():
    folder = ROOT / "evaluation-results/retrieval-v3-one-judgment-2026-11-09"
    if not (folder / "run-plan.json").is_file():
        pytest.skip("Repeated live records are local")
    return load_cohort(folder, allow_partial=True)


def test_embedded_records_equal_original_and_all_frames_have_sources(local_data):
    pool = json.loads(gzip.decompress(base64.b64decode(local_data["text_pool"])))
    root = ROOT / "evaluation-results/retrieval-v3-one-judgment-2026-11-09"
    for row in local_data["rows"]:
        decoded = json.loads(gzip.decompress(base64.b64decode(local_data["payloads"][row["id"]])))
        payload = restore(decoded, pool)
        assert payload["record"] == json.loads((root / row["result"]).read_text())
        view = payload["view"]
        assert all(s["code"] in local_data["sources"] for s in view["steps"])
        assert view["steps"][-1]["verified"] == payload["record"]["retrieval"]["coverage"]["verified_pages"]
        assert set(map(int, view["steps"][-1]["selected"])) == set(row["returned"])
    assert len(local_data["rows"]) + len(local_data["missing"]) == 180


def test_failed_attempt_is_explicit_and_not_a_success_timing(local_data):
    row = local_data["rows"][0]
    record = json.loads((ROOT / "evaluation-results/retrieval-v3-one-judgment-2026-11-09" / row["result"]).read_text())
    record = deepcopy(record)
    record["failure"] = {"type": "APIError", "message": "No response"}
    view = failure_view(record)
    assert "error" in view["steps"][-1]["title"]
    assert view["peak_concurrency"] is None and view["failure_notice"]


def test_reviewed_alternative_pages_count_as_supporting_the_answer():
    from reporting.build_cohort_report import mark_supported
    rows = [{"question_id": "q", "returned": [39, 40], "recall": .5, "failure": None},
            {"question_id": "q", "returned": [39], "recall": .5, "failure": None},
            {"question_id": "q", "returned": [25, 39], "recall": 1, "failure": None},
            {"question_id": "q", "returned": [], "recall": 0, "failure": "RateLimitError"}]
    mark_supported(rows, [{"question_id": "q", "alternative_page_sets": [[39, 40]]}])
    assert [r["supported"] for r in rows] == [True, False, True, False]
