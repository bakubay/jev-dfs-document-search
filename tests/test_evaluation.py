from __future__ import annotations

from datalab_dfs_search import TocNode
from evaluation.metrics import map_gold_pages, score_trace
from evaluation.pageindex_oss import parse_evidence_pages, score_toc_text
from evaluation.pricing import estimate_query_cost
from evaluation.summarize import summarize


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


def test_gold_pages_map_to_deepest_toc_path_and_score_visited_range():
    child = TocNode("child", "1.1 Details", 2, 3, 5, [])
    root = TocNode("root", "Part 1", 1, 1, 8, [child])
    gold = map_gold_pages([root], [4])
    assert gold.toc_paths == [["Part 1", "1.1 Details"]]

    metrics = score_trace(
        [
            {
                "type": "leaf_scan",
                "path": ["Part 1", "1.1 Details"],
                "start_page": 3,
                "end_page": 5,
                "elapsed_seconds": 1.25,
            }
        ],
        gold,
    )
    assert metrics["gold_page_recall"] == 1.0
    assert metrics["gold_section_recall"] == 1.0
    assert metrics["time_to_first_gold_range_seconds"] == 1.25


def test_cost_and_summary_keep_requests_distinct_from_questions():
    usage = {
        "jev_input_tokens": 1_000_000,
        "jev_requests": 2,
        "jev_questions": 5,
        "writer_calls": 1,
        "writer_input_tokens": 1_000_000,
        "writer_output_tokens": 1_000_000,
    }
    cost = estimate_query_cost(usage)
    assert cost["estimated_query_cost_usd"] == 12.042
    report = summarize([{
        "strategy": "jev_dfs",
        "failure": None,
        "answer_correct": True,
        "retrieval_metrics": {"gold_section_recall": 1.0, "gold_page_recall": 1.0},
        "coverage": {"screened_pages": [1, 2, 3, 4], "checked_count": 3},
        "elapsed_seconds": 2.0,
        "usage": usage,
        "cost": cost,
    }])
    row = report["strategies"][0]
    assert row["mean_jev_requests"] == 2
    assert row["mean_jev_questions"] == 5
    assert row["answer_accuracy"] == 1.0
    assert row["mean_pages_screened"] == 4
    assert row["mean_pages_verified"] == 3

