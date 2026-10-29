from __future__ import annotations

import unittest
from types import SimpleNamespace

from datalab_dfs_search import OrderedDFSSearcher, PageRecord, build_toc_tree


class FakeTypeSafe:
    """Rank the wrong branch first, then fail it and verify the right branch."""

    def system_one(self, *, state, questions):
        usage = SimpleNamespace(input_tokens=10)
        if "next_section" in questions:
            criteria = questions["next_section"].criteria
            probabilities = {}
            for key, description in criteria.items():
                probabilities[key] = 0.9 if "Wrong branch" in description else 0.1
            answer = SimpleNamespace(probabilities=probabilities, confidence=0.8)
            return SimpleNamespace(answers={"next_section": answer}, usage=usage)

        page = state.get("physical_pdf_page") or state["candidate_physical_pdf_page"]
        good = page == 3
        if "direct" in questions:
            answers = {
                "direct": SimpleNamespace(noul=0.9 if good else 0.1),
                "partial": SimpleNamespace(noul=0.8 if good else 0.2),
            }
        else:
            answers = {
                "evidence": SimpleNamespace(noul=0.95 if good else 0.1),
                "sufficient": SimpleNamespace(noul=0.9 if good else 0.1),
            }
        return SimpleNamespace(answers=answers, usage=usage)


class TocTreeTests(unittest.TestCase):
    def setUp(self):
        self.entries = [
            {"title": "Executive Summary", "printed_page": 1, "pdf_page_number": 1},
            {"title": "Part 1: Wrong branch", "printed_page": 2, "pdf_page_number": 2},
            {"title": "1 Wrong branch", "printed_page": 2, "pdf_page_number": 2},
            {"title": "Part 2: Correct branch", "printed_page": 3, "pdf_page_number": 3},
            {"title": "2 Correct branch", "printed_page": 4, "pdf_page_number": 4},
        ]

    def test_numbered_sections_nest_under_parts_and_ranges_do_not_overlap(self):
        tree = build_toc_tree(self.entries, page_count=4)
        self.assertEqual([node.title for node in tree], [
            "Executive Summary",
            "Part 1: Wrong branch",
            "Part 2: Correct branch",
        ])
        self.assertEqual(tree[1].children[0].title, "1 Wrong branch")
        self.assertEqual(tree[2].children[0].title, "2 Correct branch")
        self.assertEqual((tree[0].start_page, tree[0].end_page), (1, 1))
        self.assertEqual((tree[1].start_page, tree[1].end_page), (2, 2))

    def test_ordered_dfs_backtracks_without_pruning_siblings(self):
        tree = build_toc_tree(self.entries, page_count=4)
        pages = {
            page: PageRecord(page - 1, page, f"Page {page} text")
            for page in range(1, 5)
        }
        searcher = OrderedDFSSearcher(
            typesafe=FakeTypeSafe(),
            pages=pages,
            tree=tree,
            question="Where is the answer?",
            workers=1,
        )
        hit = searcher.search()
        self.assertIsNotNone(hit)
        self.assertEqual(hit.candidate_page, 3)
        attempted = [event["target"] for event in searcher.trace if event["type"] == "attempt"]
        self.assertIn("Part 1: Wrong branch", attempted)
        self.assertIn("Part 2: Correct branch", attempted)
        backtracked = [event["path"] for event in searcher.trace if event["type"] == "backtrack"]
        self.assertIn(["Part 1: Wrong branch"], backtracked)

    def test_event_sink_receives_the_same_ordered_hops_as_the_saved_trace(self):
        tree = build_toc_tree(self.entries, page_count=4)
        pages = {
            page: PageRecord(page - 1, page, f"Page {page} text")
            for page in range(1, 5)
        }
        streamed = []
        searcher = OrderedDFSSearcher(
            typesafe=FakeTypeSafe(),
            pages=pages,
            tree=tree,
            question="Where is the answer?",
            workers=1,
            event_sink=streamed.append,
        )
        searcher.search()
        self.assertEqual(streamed, searcher.trace)
        self.assertEqual(streamed[0]["type"], "choice")
        self.assertTrue(any(event["type"] == "backtrack" for event in streamed))


if __name__ == "__main__":
    unittest.main()
