from retrieval.tree import TocNode
from evaluation.metrics import map_gold_pages
from evaluation.dataset import parse_evidence_pages, score_toc_text
from evaluation.pricing import estimate_query_cost


def test_parse_upstream_evidence_pages():
    assert parse_evidence_pages("[3, 4, 4]") == [3, 4]
    assert parse_evidence_pages([9, 2]) == [2, 9]

def test_toc_screen_requires_a_heading_and_entries():
    text = """TABLE OF CONTENTS
Introduction ........ 1
1 Methods ........... 5
1.1 Evaluation ...... 9
"""
    assert score_toc_text(text) == (1, 3)
    assert score_toc_text("Introduction ........ 1") == (0, 1)


def test_gold_pages_map_to_deepest_toc_path():
    child = TocNode("child", "1.1 Details", 2, 3, 5, [])
    root = TocNode("root", "Part 1", 1, 1, 8, [child])
    assert map_gold_pages([root], [4]).toc_paths == [["Part 1", "1.1 Details"]]


def test_all_retrieval_model_tokens_count_toward_cost():
    usage = {"jev_input_tokens": 1_000_000, "agent_input_tokens": 1_000_000,
             "agent_output_tokens": 1_000_000}
    assert estimate_query_cost(usage)["estimated_query_cost_usd"] == 12.042
