"""Retrieval boundaries, recoverability, evidence output and measurement contracts."""

from types import SimpleNamespace as Obj

import pytest

from retrieval.tree import OrderedDFSSearcher, PageRecord, TocNode
from retrieval.common import EvidenceBudget
from retrieval.jev import JevDFSRetriever, JevExhaustiveRetriever
from evaluation.metrics import score_retrieval
from retrieval.run import run_retrieval
from retrieval.sol import AgentResult, RetrievalAction, SolRetriever


class Jev:
    def __init__(self, useful=(), required=(), invalid=False, fail=None):
        self.useful, self.required = set(useful), set(required)
        self.invalid, self.fail = invalid, fail
        self.states = []

    def system_one(self, *, state, questions):
        self.states.append(state)
        if self.fail in questions:
            raise RuntimeError("transport failed")
        if "next_section" in questions:
            keys = list(questions["next_section"].criteria)
            answers = {"next_section": Obj(probabilities={k: float(i == 0) for i, k in enumerate(keys)}, confidence=1)}
        elif "sufficient_evidence" in questions:
            value = float("nan") if self.invalid else (.99 if self.required <= set(state["source_pages"]) else .1)
            answers = {"sufficient_evidence": Obj(noul=value)}
        else:
            answers = {key: Obj(noul=.9 if int(key[1:].split("_")[0]) in self.useful else .01) for key in questions}
        return Obj(answers=answers, usage=Obj(input_tokens=10, output_tokens=2))


def jev_search(cls=JevDFSRetriever, client=None, **kwargs):
    # Gaps, nested nodes, and zero-probability siblings must remain reachable.
    pages = {p: PageRecord(p-1, p, f"Full source page {p}.") for p in range(1, 13)}
    tree = [TocNode("first", "First", 2, 2, 5),
            TocNode("later", "Later", 6, 6, 11, [TocNode("child", "Nested", 8, 8, 11)])]
    return cls(typesafe=client or Jev(), pages=pages, tree=tree,
               question="Find both facts", workers=2, **kwargs)


@pytest.mark.parametrize("cls", [JevDFSRetriever, JevExhaustiveRetriever])
def test_no_evidence_checks_every_page_without_a_writer(cls, monkeypatch):
    import openai
    monkeypatch.setattr(openai.OpenAI, "__init__", lambda *a, **k: pytest.fail("No OpenAI in JEV retrieval"))
    search = jev_search(cls)
    result = search.search()
    assert result.stop_reason == "all_text_checked"
    assert result.coverage["verified_pages"] == list(range(1, 13))
    assert not result.sufficient and result.evidence_pages == {}
    assert result.coverage["all_physical_pages_verified"]
    assert all(e["type"] not in {"draft_started", "candidate_answer", "answer_evaluation"} for e in search.trace)
    assert not any("source_pages" in s for s in search.typesafe.states)  # No pointless empty-bundle assessment.


def test_partial_evidence_accumulates_across_branches_before_stop():
    client = Jev(useful={2, 9}, required={2, 9})
    search = jev_search(client=client)
    result = search.search()
    judgments = [e for e in search.trace if e["type"] == "evidence_assessed"]
    assert any(not e["sufficient"] for e in judgments)
    assert judgments[-1]["sufficient"]
    assert {2, 9} <= set(result.evidence_pages)
    assert result.stop_reason == "sufficient_evidence"
    assert result.evidence_pages[2] == search.pages[2].markdown
    assert result.evidence_tokens <= 12_000 and len(result.evidence_pages) <= 8
    assert any(e["type"] == "backtrack" for e in search.trace)


def test_dfs_stops_early_but_exhaustive_finishes_all_windows():
    dfs = jev_search(client=Jev(useful={2}, required={2})).search()
    exhaustive = jev_search(JevExhaustiveRetriever, Jev(useful={2}, required={2})).search()
    assert dfs.sufficient and exhaustive.sufficient
    assert len(dfs.coverage["verified_pages"]) < 12
    assert len(exhaustive.coverage["verified_pages"]) == 12


def test_output_budget_does_not_prune_pending_pages():
    search = jev_search(client=Jev(useful=range(1, 13), required={99}), evidence_budget=EvidenceBudget(3, 1000))
    result = search.search()
    assert len(result.evidence_pages) <= 3
    assert result.coverage["all_physical_pages_verified"]
    assert any(e.get("omitted_candidate_pages_due_to_output_budget") for e in search.trace)


def test_invalid_sufficiency_is_an_error_not_success():
    search = jev_search(client=Jev(useful={2}, invalid=True))
    result, _, failure = run_retrieval(search)
    assert failure["type"] == "ValueError"
    assert result.stop_reason == "error" and not result.sufficient


def test_failed_windows_stay_unchecked():
    search = jev_search(client=Jev(fail="p2_evidence"))
    result, _, failure = run_retrieval(search)
    assert failure and 2 not in result.coverage["verified_pages"]
    assert not result.coverage["all_physical_pages_verified"]


def test_empty_and_missing_pages_are_reported_without_full_page_claim():
    search = jev_search()
    search.pages[1] = PageRecord(0, 1, "")
    del search.pages[12]
    del search.pages[7]
    result = search.search()
    assert result.coverage["empty_pages"] == [1]
    assert result.coverage["missing_pages"] == [7]
    assert result.coverage["all_nonempty_pages_verified"]
    assert not result.coverage["all_physical_pages_verified"]


def action(kind, target=None, selected=(), start=None):
    return AgentResult(RetrievalAction(action=kind, target_id=target, start_page=start,
                       reason="Read or select evidence", selected_pages=list(selected), sufficient=kind == "finish"), 10, 2)


def sol_search(actor, **kwargs):
    pages = {p: PageRecord(p-1, p, f"Full source page {p}.") for p in range(1, 5)}
    return SolRetriever(pages=pages, tree=[TocNode("all", "All", 1, 1, 4)],
                        question="Find evidence", api_key="unused", model="fake", actor=actor, **kwargs)


def test_sol_returns_full_selected_pages_with_no_answer_or_jev(monkeypatch):
    monkeypatch.setattr(OrderedDFSSearcher, "_system_one", lambda *a, **k: pytest.fail("No JEV in Sol retrieval"))
    moves = iter([action("read", "node_all"), action("finish", selected=[3])])
    search = sol_search(lambda **k: next(moves))
    result = search.search()
    assert result.evidence_pages == {3: "Full source page 3."}
    assert result.sufficient and result.stop_reason == "sufficient_evidence"
    assert "answer" not in result.to_dict()
    assert "answer" not in RetrievalAction.model_fields
    assert search.usage_summary()["jev_requests"] == 0


@pytest.mark.parametrize("moves, budgets", [
    ([action("finish", selected=[1]), action("stop")], {}),
    ([action("read", "node_all"), action("finish", selected=[1, 2]), action("stop")], {"evidence_budget": EvidenceBudget(1, 1000)}),
    ([action("read", "node_all"), action("finish", selected=[1, 1]), action("stop")], {}),
])
def test_invalid_evidence_cannot_finish(moves, budgets):
    moves = iter(moves)
    search = sol_search(lambda **k: next(moves), **budgets)
    result = search.search()
    assert not result.sufficient
    assert any(e["type"] == "retrieval_action_rejected" for e in search.trace)


def test_failed_sol_delivery_is_not_counted_as_read():
    def actor(**kwargs):
        if kwargs["state"]["raw_page_window"]:
            raise RuntimeError("failed delivery")
        return action("read", "node_all")
    result, _, failure = run_retrieval(sol_search(actor))
    assert failure and result.coverage["read_pages"] == []


def test_sol_step_limit_preserves_partial_evidence():
    moves = iter([action("read", "node_all"), action("read", "node_all", selected=[1])])
    search = sol_search(lambda **k: next(moves), max_navigation_steps=2, max_pages_per_read=2)
    result = search.search()
    assert result.stop_reason == "step_budget_reached" and not result.sufficient
    assert result.evidence_pages == {1: "Full source page 1."}
    assert result.coverage["read_pages"] == [1, 2]  # Last loaded window was not delivered.


def test_returned_recall_is_not_read_recall():
    moves = iter([action("read", "node_all"), action("finish", selected=[1])])
    search = sol_search(lambda **k: next(moves))
    result = search.search()
    metrics = score_retrieval(result, [3], search.tree, search.trace)
    assert metrics["read_gold_page_recall"] == 1
    assert metrics["returned_gold_page_recall"] == 0
    assert metrics["missing_gold_pages"] == [3]
    assert score_retrieval(result, [], search.tree, search.trace)["returned_gold_page_recall"] is None
