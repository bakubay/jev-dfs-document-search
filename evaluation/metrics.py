"""Annotated-page and section recall, computed after retrieval finishes."""
from __future__ import annotations
from dataclasses import dataclass
from retrieval.tree import TocNode

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

def score_retrieval(result, gold_pages, tree, trace):
    gold = set(gold_pages)
    returned = set(result.evidence_pages)
    read = set(result.coverage["read_pages"])
    verified = set(result.coverage["verified_pages"])
    gold_paths = {tuple(p) for p in map_gold_pages(tree, list(gold)).toc_paths}
    returned_paths = {tuple(p) for p in map_gold_pages(tree, list(returned)).toc_paths}

    def recall(pages):
        return len(gold & pages) / len(gold) if gold else None

    first_read = None
    for event in sorted(trace, key=lambda e: e.get("elapsed_seconds", 0)):
        if event.get("type") == "jev_window_completed":
            pages = event["source_pages"]
        elif event.get("type") == "sol_read_completed":
            pages = event["pages"]
        else:
            continue
        if gold.intersection(pages):
            first_read = event.get("elapsed_seconds")
            break
    return {
        "metric_version": "retrieval-v1",
        "returned_gold_page_recall": recall(returned),
        "returned_all_gold_pages": gold <= returned if gold else None,
        "returned_any_gold_page": bool(gold & returned) if gold else None,
        "returned_gold_section_recall": len(gold_paths & returned_paths) / len(gold_paths) if gold_paths else None,
        "read_gold_page_recall": recall(read),
        "verified_gold_page_recall": recall(verified),
        "returned_page_count": len(returned),
        "returned_text_tokens": result.evidence_tokens,
        "read_page_count": len(read),
        "verified_page_count": len(verified),
        "time_to_first_gold_page_read_seconds": first_read,
        "returned_gold_pages": sorted(gold & returned),
        "missing_gold_pages": sorted(gold - returned),
        "annotation_note": "Strict annotated-page recall. Other returned pages may contain valid alternative evidence; not automatically false positives.",
    }
