"""The frozen question set, balanced repeats and no selective resume/rerun."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest

from evaluation.run_repeated import make_cases, validate_record

ROOT = Path(__file__).parents[1]


def test_balanced_deterministic_repeats():
    manifest = {"seed": 42, "repeats": 3, "strategies": ["jev_dfs", "exhaustive", "sol_only"],
                "questions": [{"id": str(i), "doc_id": "doc.pdf"} for i in range(20)]}
    cases = make_cases(manifest)
    assert len(cases) == 180 and cases == make_cases(manifest)
    assert len({c["result"] for c in cases}) == 180
    assert set(Counter((c["question_id"], c["strategy"]) for c in cases).values()) == {3}
    assert Counter(c["repeat"] for c in cases) == {1: 60, 2: 60, 3: 60}


def test_resume_accepts_saved_failure_but_rejects_source_change():
    plan = {"experiment": "test-experiment", "source_sha256": {"a.py": "hash"}}
    case = {"question_id": "q", "strategy": "jev_dfs"}
    record = {"experiment": plan["experiment"], "question": {"id": "q"}, "strategy": "jev_dfs",
              "configuration": {"source_sha256": {"a.py": "hash"}}, "failure": {"type": "Error"}}
    validate_record(record, plan, case)
    bad = deepcopy(record)
    bad["configuration"]["source_sha256"]["a.py"] = "changed"
    with pytest.raises(ValueError, match="source hashes"):
        validate_record(bad, plan, case)


def test_cohort_has_twenty_labeled_questions_for_the_current_experiment():
    from retrieval import EXPERIMENT_VERSION
    manifest = json.loads((ROOT / "evaluation/cohort.json").read_text())
    assert manifest["experiment"] == EXPERIMENT_VERSION
    assert len(manifest["documents"]) == 4 and len(manifest["questions"]) == 20
    assert set(Counter(q["doc_id"] for q in manifest["questions"]).values()) == {5}
    assert sum(q["in_pageindex_oss_subset"] for q in manifest["questions"]) == 8
    for q in manifest["questions"]:
        assert q["evidence_pages"] and set(map(int, q["reference_page_text_sha256"])) == set(q["evidence_pages"])
    for doc in manifest["documents"]:
        assert doc["root_options"] > 1

