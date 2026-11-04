from __future__ import annotations

from types import SimpleNamespace

from datalab_dfs_search import PageRecord, build_toc_tree
from evaluation.strategies import (
    ExhaustiveSectionSearcher,
    LLMSelectedSectionSearcher,
    TocSelectionResult,
)
from openai_grounded_answer import AnswerResult


class AllUsefulJev:
    def __init__(self):
        self.checked = []

    def system_one(self, *, state, questions):
        usage = SimpleNamespace(input_tokens=5, output_tokens=1)
        if "answer_kind" in questions:
            return SimpleNamespace(usage=usage, answers={
                "answer_kind": SimpleNamespace(probabilities={"direct": .99}),
                "complete": SimpleNamespace(noul=.99),
                "supported": SimpleNamespace(noul=.99),
            })
        if "direct" in questions:
            return SimpleNamespace(usage=usage, answers={
                "direct": SimpleNamespace(noul=.9),
                "partial": SimpleNamespace(noul=.9),
            })
        self.checked.append(state["candidate_physical_pdf_page"])
        return SimpleNamespace(usage=usage, answers={
            "evidence": SimpleNamespace(noul=.95),
            "sufficient": SimpleNamespace(noul=.95),
        })


def complete_draft(**kwargs):
    page = min(kwargs["evidence_pages"])
    return AnswerResult(
        answer="answer",
        fully_answered=True,
        limitations="",
        citations=[{"physical_pdf_page": page, "quote": kwargs["evidence_pages"][page]}],
        model="fake",
        response_id=None,
        input_tokens=10,
        output_tokens=2,
    )


def fixture(searcher, **extra):
    pages = {page: PageRecord(page - 1, page, f"page {page}") for page in range(1, 5)}
    entries = [
        {"title": "1 First", "printed_page": 1, "pdf_page_number": 1},
        {"title": "2 Second", "printed_page": 3, "pdf_page_number": 3},
    ]
    client = AllUsefulJev()
    instance = searcher(
        typesafe=client,
        pages=pages,
        tree=build_toc_tree(entries, 4),
        question="question",
        workers=1,
        api_key="unused",
        answer_model="fake",
        drafter=complete_draft,
        **extra,
    )
    return instance, client


def test_exhaustive_policy_checks_every_page_and_drafts_once():
    search, client = fixture(ExhaustiveSectionSearcher)
    assert search.search() is not None
    assert sorted(client.checked) == [1, 2, 3, 4]
    assert search.writer_calls == 1
    assert search.coverage()["all_pages_checked"] is True


def test_llm_policy_checks_only_selected_leaf():
    def selector(**kwargs):
        assert len(kwargs["candidates"]) == 2
        return TocSelectionResult(selected_ids=["leaf_0001"])

    search, client = fixture(LLMSelectedSectionSearcher, selector=selector)
    assert search.search() is not None
    assert sorted(client.checked) == [3, 4]
    assert search.coverage()["all_pages_checked"] is False
    assert search.selector_calls == 1
