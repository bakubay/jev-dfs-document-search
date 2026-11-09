"""Public commands wire the three methods to the same prepared inputs."""
import json
import sys
from pathlib import Path

import pytest

from evaluation import run as benchmark
from retrieval import __main__ as single
from retrieval.sol import SolRetriever
from test_retrieval import Jev, action


class FakeClient(Jev):
    def __init__(self, **kwargs):
        super().__init__(useful={2}, required={2})

    def close(self):
        pass


def fake_sol(**kwargs):
    moves = iter([action("read", "node_toc-0000"), action("finish", selected=[2])])
    return SolRetriever(actor=lambda **_: next(moves), **kwargs)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    grounded = tmp_path / "doc" / "grounded"
    grounded.mkdir(parents=True)
    (grounded / "pages.jsonl").write_text("\n".join(json.dumps({
        "page_id": p - 1, "pdf_page_number": p, "markdown": f"Full source page {p}."
    }) for p in (1, 2, 3)))
    (grounded / "toc_entries.json").write_text(json.dumps([
        {"title": "Contents", "printed_page": 1, "pdf_page_number": 1}
    ]))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"questions": [{"id": "q", "doc_id": "doc.pdf",
        "question": "Find the fact", "evidence_pages": [2]}]}))
    for module in (benchmark, single):
        monkeypatch.setattr(module, "TypeSafeClient", FakeClient)
        monkeypatch.setattr(module, "SolRetriever", fake_sol)
        monkeypatch.setattr(module, "load_local_env", lambda _: None)
    monkeypatch.setitem(benchmark.STRATEGIES, "sol_only", fake_sol)
    monkeypatch.setenv("TYPESAFE_API_KEY", "unused-test-value")
    monkeypatch.setenv("OPENAI_API_KEY", "unused-test-value")
    return grounded, manifest


def test_evaluation_runner_records_all_three_methods(inputs, tmp_path, monkeypatch):
    grounded, manifest = inputs
    output = tmp_path / "comparison"
    monkeypatch.setattr(sys, "argv", ["evaluation.run", str(manifest), "--grounded-root", str(tmp_path),
        "--question-ids", "q", "--output-dir", str(output)])
    assert benchmark.main() == 0
    for method in ("jev_dfs", "exhaustive", "sol_only"):
        record = json.loads((output / f"q.{method}.json").read_text())
        assert record["failure"] is None
        assert record["metrics"]["returned_gold_page_recall"] == 1
        assert record["usage"]["writer_calls"] == 0
        assert "answer" not in record["retrieval"]
        assert "retrieval/jev.py" in record["configuration"]["source_sha256"]
        assert "evaluated_dfs.py" not in record["configuration"]["source_sha256"]


@pytest.mark.parametrize("method", ["jev_dfs", "exhaustive", "sol_only"])
def test_single_question_cli_returns_evidence(inputs, tmp_path, monkeypatch, method):
    grounded, _ = inputs
    output = tmp_path / f"{method}.json"
    monkeypatch.setattr(sys, "argv", ["retrieval", str(grounded), "Find the fact", "--method", method,
        "--output", str(output)])
    assert single.main() == 0
    record = json.loads(output.read_text())
    assert record["failure"] is None and "2" in record["retrieval"]["evidence_pages"]
