"""Gold-page and gold-section metrics shared by all search strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datalab_dfs_search import TocNode


@dataclass(frozen=True)
class GoldMapping:
    evidence_pages: list[int]
    toc_paths: list[list[str]]


def _deepest_paths(nodes: list[TocNode], page: int, prefix: list[str]) -> list[list[str]]:
    matches = []
    for node in nodes:
        end = node.end_page if node.end_page is not None else node.start_page
        if not node.start_page <= page <= end:
            continue
        path = prefix + [node.title]
        children = _deepest_paths(node.children, page, path)
        matches.extend(children or [path])
    return matches


def map_gold_pages(tree: list[TocNode], evidence_pages: list[int]) -> GoldMapping:
    paths: list[list[str]] = []
    seen = set()
    for page in sorted(set(evidence_pages)):
        for path in _deepest_paths(tree, page, []):
            key = tuple(path)
            if key not in seen:
                seen.add(key)
                paths.append(path)
    return GoldMapping(evidence_pages=sorted(set(evidence_pages)), toc_paths=paths)


def _path_reached(gold: list[str], visited: list[str]) -> bool:
    """A leaf below a gold parent section counts as reaching that section."""
    return visited[: len(gold)] == gold or gold[: len(visited)] == visited


def score_trace(trace: list[dict[str, Any]], gold: GoldMapping) -> dict[str, Any]:
    scans = [event for event in trace if event.get("type") == "leaf_scan"]
    visited_paths = [list(event.get("path") or []) for event in scans]
    covered_gold_pages = {
        page
        for page in gold.evidence_pages
        if any(int(scan["start_page"]) <= page <= int(scan["end_page"]) for scan in scans)
    }
    reached_paths = [
        path for path in gold.toc_paths
        if any(_path_reached(path, visited) for visited in visited_paths)
    ]
    first_gold = next(
        (
            scan.get("elapsed_seconds")
            for scan in scans
            if any(
                int(scan["start_page"]) <= page <= int(scan["end_page"])
                for page in gold.evidence_pages
            )
        ),
        None,
    )
    return {
        "gold_page_recall": len(covered_gold_pages) / len(gold.evidence_pages),
        "gold_section_recall": (
            len(reached_paths) / len(gold.toc_paths) if gold.toc_paths else None
        ),
        "covered_gold_pages": sorted(covered_gold_pages),
        "gold_toc_paths": gold.toc_paths,
        "reached_gold_toc_paths": reached_paths,
        "leaf_ranges_inspected": len(scans),
        "time_to_first_gold_range_seconds": first_gold,
    }

