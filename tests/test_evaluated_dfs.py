"""Behavioral tests for the early-stop regression and exhaustive coverage."""

import unittest
from types import SimpleNamespace

from datalab_dfs_search import PageRecord, build_toc_tree
from evaluated_dfs import EvaluatedDFSSearcher
from openai_grounded_answer import AnswerResult


class Judgments:
    def __init__(self, accepted=True, verdict="direct"):
        self.accepted = accepted
        self.verdict = verdict
        self.checked = []
        self.menus = 0
        self.gates = []

    def system_one(self, *, state, questions):
        usage = SimpleNamespace(input_tokens=10)
        if "next_section" in questions:
            self.menus += 1
            keys = list(questions["next_section"].criteria)
            # Intentionally omit the low-ranked branch: code must still retain it.
            return SimpleNamespace(usage=usage, answers={"next_section": SimpleNamespace(
                probabilities={keys[0]: 1.0}, confidence=1,
            )})
        if "answer_kind" in questions:
            self.gates.append(state)
            good = self.accepted and "actual fee" in " ".join(state["source_pages"].values())
            return SimpleNamespace(usage=usage, answers={
                "answer_kind": SimpleNamespace(probabilities={
                    self.verdict if good else "partial": .99,
                }),
                "complete": SimpleNamespace(noul=.99 if good else .1),
                "supported": SimpleNamespace(noul=.99),
            })
        if "direct" in questions:
            return SimpleNamespace(usage=usage, answers={
                "direct": SimpleNamespace(noul=.01),
                "partial": SimpleNamespace(noul=.01),
            })
        self.checked.append(state["candidate_physical_pdf_page"])
        return SimpleNamespace(usage=usage, answers={
            "evidence": SimpleNamespace(noul=.9),
            "sufficient": SimpleNamespace(noul=.99),
        })


class EvaluatedDFSTests(unittest.TestCase):
    def make_search(self, *, accepted=True, empty=False, truthful_writer=True, verdict="direct"):
        pages = {p: PageRecord(p-1, p, "Free publication" if p < 5 else "actual fee: none")
                 for p in range(1,6)}
        if empty:
            pages[3] = PageRecord(2, 3, "")
        entries = [
            {"title":"1 Publication access", "printed_page":1,"pdf_page_number":1},
            {"title":"2 Licensing", "printed_page":4,"pdf_page_number":4},
        ]
        client = Judgments(accepted, verdict)
        bundles = []

        def draft(**kwargs):
            evidence = kwargs["evidence_pages"]
            bundles.append(dict(evidence))
            complete = 5 in evidence
            p = 5 if complete else 1
            return AnswerResult(
                answer="No actual fee." if complete else "Only publication access is free.",
                fully_answered=complete if truthful_writer else True,
                limitations="" if complete else "Missing explicit licensing information.",
                citations=[{"physical_pdf_page":p,"quote":evidence[p]}],
                model="fake-sol",response_id=None,input_tokens=20,output_tokens=10,
            )

        search = EvaluatedDFSSearcher(
            typesafe=client, pages=pages, tree=build_toc_tree(entries,5),
            question="What licensing fee?",workers=1,drafter=draft,
            max_verify_per_leaf=1,
        )
        return search, client, bundles

    def test_rejects_premature_answer_resumes_same_dfs_and_accumulates(self):
        search, client, bundles = self.make_search()
        hit = search.search()
        self.assertIsNotNone(hit)
        self.assertEqual(client.menus, 1)  # No root restart.
        self.assertEqual(len(bundles),2)
        self.assertIn(1,bundles[1])
        self.assertIn(5,bundles[1])
        self.assertEqual(sorted(client.checked),[1,2,3,4,5])
        self.assertEqual(len(set(client.checked)),len(client.checked))
        self.assertEqual([a["evaluation"]["accepted"] for a in search.answer_attempts],[False,True])
        self.assertEqual(search.final_answer()["status"],"answered")
        self.assertTrue(search.coverage()["all_pages_checked"])

    def test_jev_can_reject_a_writer_that_claims_completeness(self):
        search, _, _ = self.make_search(truthful_writer=False)
        search.search()
        self.assertFalse(search.answer_attempts[0]["evaluation"]["accepted"])

    def test_negative_question_exhausts_despite_low_screen_and_top_k_limit(self):
        search, client, _ = self.make_search(accepted=False)
        self.assertIsNone(search.search())
        self.assertEqual(sorted(client.checked),[1,2,3,4,5])
        self.assertEqual(search.final_answer()["status"],"exhausted")
        self.assertEqual(search.final_answer()["answer"],
                         "We searched every page and found no sufficient evidence.")

    def test_empty_page_prevents_false_full_coverage_claim(self):
        search, _, _ = self.make_search(accepted=False,empty=True)
        search.search()
        self.assertEqual(search.coverage()["empty_pages"],[3])
        self.assertEqual(search.final_answer()["status"],"incomplete_coverage")
        self.assertEqual(search.coverage()["checked_count"],4)

    def test_explicit_source_negative_can_stop(self):
        search, _, _ = self.make_search(verdict="explicit_negative")
        self.assertIsNotNone(search.search())
        self.assertEqual(search.answer_attempts[-1]["evaluation"]["verdict"],"explicit_negative")

    def test_failed_citation_does_not_end_dfs(self):
        search, _, _ = self.make_search()
        good_drafter = search.drafter

        def draft(**kwargs):
            result = good_drafter(**kwargs)
            if 5 not in kwargs["evidence_pages"]:
                from dataclasses import replace
                return replace(result, citations=[{"physical_pdf_page":1,"quote":"invented fee"}])
            return result

        search.drafter = draft
        self.assertIsNotNone(search.search())
        self.assertTrue(any(e.get("verdict") == "invalid_citation" for e in search.trace))

    def test_incomplete_writer_blocks_stop_even_when_jev_says_complete(self):
        search, _, _ = self.make_search()
        original = search.drafter

        def draft(**kwargs):
            from dataclasses import replace
            return replace(original(**kwargs), fully_answered=False, limitations="Missing condition")

        search.drafter = draft
        self.assertIsNone(search.search())
        self.assertEqual(search.answer_attempts[-1]["evaluation"]["complete"], .99)
        self.assertFalse(search.answer_attempts[-1]["evaluation"]["accepted"])


if __name__ == "__main__":
    unittest.main()
