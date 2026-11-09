"""One-pass judgments retain DFS recovery, evidence gates and window budgets."""

from collections import Counter
from threading import Event

import pytest

from retrieval.tree import PageRecord, TocNode
from retrieval.jev import JevDFSRetriever, JevExhaustiveRetriever
from retrieval.run import run_retrieval
from test_retrieval import Jev, jev_search


def completed(search):
    return [e for e in search.trace if e["type"] == "jev_window_completed"]


def test_four_predicates_share_one_complete_window():
    search = jev_search()
    search._check_combined_windows([2, 3, 4], ["test"])
    event = next(e for e in search.trace if e["type"] == "jev_window_started")
    assert set(event["questions"]) == {f"p{p}_{kind}" for p in (2, 3, 4)
                                      for kind in ("direct", "partial", "evidence", "sufficient")}
    assert event["question_count"] == 12 and search.jev_requests == 1
    assert event["source_pages"] == [1, 2, 3, 4, 5]
    assert set(search.partial_pages) == {2, 3, 4}


@pytest.mark.parametrize("cls", [JevDFSRetriever, JevExhaustiveRetriever])
def test_every_page_including_gaps_and_zero_ranked_siblings_is_checked_once(cls, monkeypatch):
    import openai
    monkeypatch.setattr(openai.OpenAI, "__init__", lambda *a, **k: pytest.fail("No answer writer"))
    search = jev_search(cls)
    result = search.search()
    assert result.stop_reason == "all_text_checked" and not result.sufficient
    assert result.coverage["verified_pages"] == list(range(1, 13))
    targets = [p for e in completed(search) for p in e["target_pages"]]
    assert Counter(targets) == Counter(range(1, 13))
    for event in completed(search):
        assert event["phase"] == "combined"
        assert len(event["source_pages"]) <= 8 and event["page_text_tokens"] <= 12_000
        assert event["question_count"] == 4 * len(event["target_pages"])
        assert all(set(search._page_context(p)) <= set(event["source_pages"]) for p in event["target_pages"])
    assert not any(r["purpose"] in {"screen_page_window", "verify_page_window"} for r in search.request_trace)
    assert search.jev_questions == sum(len(r["question_ids"]) for r in search.request_trace)


def test_cross_branch_evidence_accumulates_and_dfs_stops_only_when_bundle_is_sufficient():
    search = jev_search(JevDFSRetriever, Jev(useful={2, 9}, required={2, 9}))
    result = search.search()
    gates = [e for e in search.trace if e["type"] == "evidence_assessed"]
    assert any(not e["sufficient"] for e in gates) and gates[-1]["sufficient"]
    assert {2, 9} <= set(result.evidence_pages)
    assert result.stop_reason == "sufficient_evidence"
    assert any(e["type"] == "backtrack" for e in search.trace)


def test_dfs_stops_early_exhaustive_finishes_and_leaf_pages_use_physical_order():
    dfs = jev_search(JevDFSRetriever, Jev(useful={2}, required={2}))
    exhaustive = jev_search(JevExhaustiveRetriever, Jev(useful={2}, required={2}))
    assert dfs.search().sufficient and exhaustive.search().sufficient
    assert len(dfs.partial_pages) < 12 and len(exhaustive.partial_pages) == 12
    for event in completed(dfs):
        assert event["target_pages"] == sorted(event["target_pages"])


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -0.1, 1.1])
def test_missing_or_invalid_fourth_answer_commits_nothing(bad):
    class Malformed(Jev):
        def system_one(self, **kwargs):
            response = super().system_one(**kwargs)
            if bad is None:
                del response.answers["p3_sufficient"]
            else:
                response.answers["p3_sufficient"].noul = bad
            return response
    search = jev_search(JevDFSRetriever, Malformed())
    with pytest.raises((KeyError, ValueError)):
        search._check_combined_windows([2, 3], ["test"])
    assert not search.screens and not search.partial_pages and not search.read_pages
    assert not completed(search)


def test_transport_failure_retains_other_completed_windows():
    search = jev_search(JevExhaustiveRetriever, Jev(fail="p8_evidence"))
    result, _, failure = run_retrieval(search)
    assert failure and result.stop_reason == "error"
    assert 8 not in search.partial_pages and 1 in search.partial_pages
    assert set(search.partial_pages) == {p for e in completed(search) for p in e["target_pages"]}
    assert not result.coverage["all_nonempty_pages_verified"]


def test_exhaustive_workers_refill_across_sections_while_first_window_waits():
    class Concurrent(Jev):
        def __init__(self):
            super().__init__()
            self.third_window = Event()

        def system_one(self, *, state, questions):
            # 7 targets in first window; 6 in the second (neighbors count too).
            if "p1_direct" in questions:
                assert self.third_window.wait(3), "Pool stalled behind the first window"
            if "p14_direct" in questions:
                self.third_window.set()
            return super().system_one(state=state, questions=questions)

    client = Concurrent()
    search = jev_search(JevExhaustiveRetriever, client)
    search.pages = {p: PageRecord(p-1, p, f"Full page {p}") for p in range(1, 25)}
    search.tree = [TocNode(str(p), f"Section {p}", p, p, p) for p in search.pages]
    assert search.search().coverage["all_physical_pages_verified"]
    assert client.third_window.is_set()
    assert len(completed(search)) == 4


def test_oversize_context_fails_before_call_without_truncation():
    search = jev_search(JevDFSRetriever)
    search.pages[2] = PageRecord(1, 2, "token " * 13_000)
    with pytest.raises(ValueError, match="not truncated or checked"):
        search._check_combined_windows([2], ["test"])
    assert not search.typesafe.states and not search.partial_pages
