from __future__ import annotations

import unittest
from types import SimpleNamespace

from datalab_dfs_search import OrderedDFSSearcher, PageRecord, build_toc_tree
from datalab_ground_toc import extract_toc


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

    def test_index_heading_and_unpaged_part_rows_form_a_10k_hierarchy(self):
        pages = [{
            "children": [
                {
                    "block_type": "SectionHeader",
                    "markdown": "# AMAZON.COM FORM 10-K INDEX",
                },
                {
                    "block_type": "TableOfContents",
                    "html": (
                        "<table><tr><th></th><th></th><th>Page</th></tr>"
                        "<tr><td colspan='3'>PART I</td></tr>"
                        "<tr><td>Item 1.</td><td>Business</td><td>3</td></tr>"
                        "<tr><td>Item 1A.</td><td>Risk Factors</td><td>6</td></tr>"
                        "<tr><td colspan='3'>PART II</td></tr>"
                        "<tr><td>Item 5.</td><td>Market Information</td><td>17</td></tr>"
                        "</table>"
                    ),
                },
            ]
        }]
        entries = extract_toc(pages)
        self.assertEqual([entry["title"] for entry in entries], [
            "PART I", "Item 1. Business", "Item 1A. Risk Factors",
            "PART II", "Item 5. Market Information",
        ])
        tree = build_toc_tree(
            [{**entry, "pdf_page_number": entry["printed_page"]} for entry in entries],
            page_count=20,
        )
        self.assertEqual([node.title for node in tree], ["PART I", "PART II"])
        self.assertEqual([node.title for node in tree[0].children], [
            "Item 1. Business", "Item 1A. Risk Factors",
        ])

    def test_later_toc_is_nested_under_the_deepest_enclosing_section(self):
        entries = [
            {"title": "PART II", "printed_page": 17, "pdf_page_number": 17, "toc_page_id": 1},
            {"title": "Item 8. Financial Statements", "printed_page": 35, "pdf_page_number": 35, "toc_page_id": 1},
            {"title": "Item 9. Controls", "printed_page": 73, "pdf_page_number": 73, "toc_page_id": 1},
            {"title": "Notes to Financial Statements", "printed_page": 42, "pdf_page_number": 42, "toc_page_id": 35},
            {"title": "Revenue", "printed_page": 48, "pdf_page_number": 48, "toc_page_id": 35},
        ]
        tree = build_toc_tree(entries, page_count=85)
        item8 = tree[0].children[0]
        self.assertEqual(item8.title, "Item 8. Financial Statements")
        self.assertEqual([node.title for node in item8.children], [
            "Notes to Financial Statements", "Revenue",
        ])
        self.assertEqual((item8.start_page, item8.end_page), (35, 72))
        self.assertEqual((item8.children[-1].start_page, item8.children[-1].end_page), (48, 72))

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
