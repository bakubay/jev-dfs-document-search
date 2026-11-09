"""Grounded document types, ToC construction, and ordered DFS traversal.

This module owns branch order and backtracking. Page judgments and evidence
stopping live in retrieval.jev; Sol reuses only the deterministic ToC helpers.
"""
from __future__ import annotations
import json
import re
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, TypeVar
import tiktoken
from typesafe_sdk import Choice
from retrieval.common import RetrievalResult

MODEL = "jev-1.13.0"
DEFAULT_WORKERS = 8
DEFAULT_RECALL_THRESHOLD = 0.35
DEFAULT_MAX_VERIFY_PER_LEAF = 8
MAX_CHOICE_OPTIONS = 255
T = TypeVar("T")


@dataclass
class TocNode:
    id: str
    title: str
    printed_page: int
    start_page: int
    end_page: int | None = None
    children: list[TocNode] = field(default_factory=list)

    def description(self) -> str:
        end = self.end_page if self.end_page is not None else self.start_page
        description = f"{self.title}. Physical PDF pages {self.start_page} through {end}."
        if self.children:
            child_titles = "; ".join(child.title for child in self.children)
            description += f" Contains these exact ToC subsections: {child_titles}."
        return description

@dataclass(frozen=True)
class SearchTarget:
    key: str
    title: str
    start_page: int
    end_page: int
    node: TocNode | None
    description: str

@dataclass(frozen=True)
class PageRecord:
    page_id: int
    pdf_page_number: int
    markdown: str
    source_block_id: str | None = None
    bbox: list[float] | None = None

@dataclass(frozen=True)
class PageScreen:
    page: int
    direct: float
    partial: float

    @property
    def recall_score(self) -> float:
        return max(self.direct, self.partial)

@dataclass(frozen=True)
class PageVerification:
    page: int
    direct: float
    partial: float
    evidence: float
    sufficient: float

def run_parallel(items: Iterable[T], workers: int, fn: Callable[[T], Any]) -> list[Any]:
    values = list(items)
    if not values:
        return []
    results = []
    with ThreadPoolExecutor(max_workers=min(workers, len(values))) as executor:
        futures = {executor.submit(fn, item): item for item in values}
        for future in as_completed(futures):
            results.append(future.result())
    return results

def load_pages(path: Path) -> dict[int, PageRecord]:
    pages: dict[int, PageRecord] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        page = PageRecord(
            page_id=int(raw["page_id"]),
            pdf_page_number=int(raw["pdf_page_number"]),
            markdown=str(raw.get("markdown") or "").strip(),
            source_block_id=raw.get("source_block_id"),
            bbox=raw.get("bbox"),
        )
        if page.pdf_page_number in pages:
            raise ValueError(f"Duplicate PDF page {page.pdf_page_number} at line {line_number}")
        pages[page.pdf_page_number] = page
    if not pages:
        raise ValueError(f"No pages found in {path}")
    return pages

def _number_path(title: str) -> tuple[str, ...] | None:
    item = re.match(r"^\s*item\s+(\d+[a-z]?)[.\s]", title, re.IGNORECASE)
    if item:
        return (item.group(1).lower(),)
    match = re.match(r"^\s*(\d+(?:\.\d+)*)[.\s]", title)
    return tuple(match.group(1).split(".")) if match else None

def build_toc_tree(entries: list[dict[str, Any]], page_count: int) -> list[TocNode]:
    """Recover hierarchy, including later sub-indexes inside a parent section."""

    def build_group(group: list[tuple[int, dict[str, Any]]]) -> list[TocNode]:
        roots: list[TocNode] = []
        active_part: TocNode | None = None
        numbered: dict[tuple[str, ...], TocNode] = {}
        for index, entry in group:
            title = str(entry["title"]).strip()
            start = int(entry["pdf_page_number"])
            if not 1 <= start <= page_count:
                raise ValueError(f"ToC entry {title!r} points outside the PDF: {start}")
            node = TocNode(
                id=f"toc-{index:04d}",
                title=title,
                printed_page=int(entry["printed_page"]),
                start_page=start,
            )

            is_part = bool(
                re.match(r"^part\s+(?:\d+|[ivxlcdm]+)\b", title, re.IGNORECASE)
            )
            is_appendix = bool(re.match(r"^appendix\b", title, re.IGNORECASE))
            number_path = _number_path(title)

            if is_part:
                roots.append(node)
                active_part = node
                numbered.clear()
            elif is_appendix:
                roots.append(node)
                active_part = None
                numbered.clear()
            elif number_path:
                if len(number_path) == 1:
                    parent = active_part
                else:
                    parent = numbered.get(number_path[:-1]) or active_part
                (parent.children if parent else roots).append(node)
                numbered[number_path] = node
                for path in list(numbered):
                    if len(path) > len(number_path) and path[: len(number_path)] != number_path:
                        numbered.pop(path)
            else:
                roots.append(node)
        return roots

    def assign_ranges(siblings: list[TocNode], parent_end: int) -> None:
        for position, node in enumerate(siblings):
            next_start = (
                siblings[position + 1].start_page
                if position + 1 < len(siblings)
                else parent_end + 1
            )
            node.end_page = max(node.start_page, next_start - 1)
            assign_ranges(node.children, node.end_page)

    # Consecutive physical ToC pages are one index. A later, separated ToC is
    # usually a sub-index (for example, financial statement notes inside Item 8).
    groups: list[list[tuple[int, dict[str, Any]]]] = []
    last_toc_page: int | None = None
    for index, entry in enumerate(entries):
        toc_page = int(entry.get("toc_page_id", -1))
        if not groups or (
            toc_page >= 0 and last_toc_page is not None and toc_page > last_toc_page + 1
        ):
            groups.append([])
        groups[-1].append((index, entry))
        if toc_page >= 0:
            last_toc_page = toc_page

    if not groups:
        return []
    roots = build_group(groups[0])
    assign_ranges(roots, page_count)

    def deepest_container(nodes: list[TocNode], page: int) -> TocNode | None:
        for node in nodes:
            end = node.end_page if node.end_page is not None else node.start_page
            if node.start_page <= page <= end:
                return deepest_container(node.children, page) or node
        return None

    for group in groups[1:]:
        toc_page_id = int(group[0][1].get("toc_page_id", -1))
        anchor_page = toc_page_id + 1 if toc_page_id >= 0 else min(
            int(entry["pdf_page_number"]) for _, entry in group
        )
        parent = deepest_container(roots, anchor_page)
        subroots = build_group(group)
        if parent is None:
            roots.extend(subroots)
            assign_ranges(roots, page_count)
        else:
            parent.children.extend(subroots)
            parent.children.sort(key=lambda node: node.start_page)
            assign_ranges(parent.children, parent.end_page or parent.start_page)
    return roots

def load_grounded_document(directory: Path) -> tuple[dict[int, PageRecord], list[TocNode]]:
    pages = load_pages(directory / "pages.jsonl")
    entries = json.loads((directory / "toc_entries.json").read_text(encoding="utf-8"))
    return pages, build_toc_tree(entries, max(pages))

class OrderedDFSSearcher:
    """Navigation base; a retriever supplies `_scan_leaf` for page processing."""
    def __init__(
        self,
        *,
        typesafe: Any,
        pages: dict[int, PageRecord],
        tree: list[TocNode],
        question: str,
        workers: int = DEFAULT_WORKERS,
        recall_threshold: float = DEFAULT_RECALL_THRESHOLD,
        max_verify_per_leaf: int = DEFAULT_MAX_VERIFY_PER_LEAF,
        max_depth: int = 64,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.typesafe = typesafe
        self.pages = pages
        self.tree = tree
        self.question = question
        self.workers = workers
        self.recall_threshold = recall_threshold
        self.max_verify_per_leaf = max_verify_per_leaf
        self.max_depth = max_depth
        self.encoder = tiktoken.get_encoding("o200k_base")
        self.trace: list[dict[str, Any]] = []
        self.request_trace: list[dict[str, Any]] = []
        self.event_sink = event_sink
        self.partial_pages: dict[int, PageVerification] = {}
        self.api_input_tokens = 0
        self.api_output_tokens = 0
        self.jev_requests = 0
        self.jev_questions = 0
        self.jev_latency_seconds = 0.0
        self.started_at = time.perf_counter()
        self.usage_lock = Lock()

    def _emit(self, event: dict[str, Any]) -> None:
        """Record a search hop and optionally stream it to a UI or logger."""
        event = {
            **event,
            "elapsed_seconds": round(time.perf_counter() - self.started_at, 6),
        }
        self.trace.append(event)
        if self.event_sink is not None:
            self.event_sink(event)

    def _record_usage(
        self,
        response: Any,
        *,
        question_count: int = 0,
        latency_seconds: float = 0.0,
    ) -> None:
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        with self.usage_lock:
            self.api_input_tokens += input_tokens
            self.api_output_tokens += output_tokens
            self.jev_requests += 1
            self.jev_questions += question_count
            self.jev_latency_seconds += latency_seconds

    def _system_one(
        self,
        *,
        purpose: str,
        state: dict[str, Any],
        questions: dict[str, Any],
    ) -> Any:
        """Call JEV and record requests separately from batched questions."""
        started = time.perf_counter()
        response = self.typesafe.system_one(state=state, questions=questions)
        latency = time.perf_counter() - started
        self._record_usage(
            response,
            question_count=len(questions),
            latency_seconds=latency,
        )
        self.request_trace.append(
            {
                "type": "jev_request",
                "purpose": purpose,
                "question_ids": list(questions),
                "latency_seconds": round(latency, 6),
                "elapsed_seconds": round(time.perf_counter() - self.started_at, 6),
                "input_tokens": int(
                    getattr(getattr(response, "usage", None), "input_tokens", 0) or 0
                ),
                "output_tokens": int(
                    getattr(getattr(response, "usage", None), "output_tokens", 0) or 0
                ),
            }
        )
        return response

    def usage_summary(self) -> dict[str, Any]:
        return {
            "jev_requests": self.jev_requests,
            "jev_questions": self.jev_questions,
            "jev_input_tokens": self.api_input_tokens,
            "jev_output_tokens": self.api_output_tokens,
            "jev_latency_seconds": round(self.jev_latency_seconds, 6),
        }

    def _targets_for(
        self,
        *,
        parent: TocNode | None,
        children: list[TocNode],
        range_start: int,
        range_end: int,
        parent_label: str,
    ) -> list[SearchTarget]:
        """Preserve pages that belong to a parent but precede its first child."""
        targets: list[SearchTarget] = []
        cursor = range_start
        for child in sorted(children, key=lambda item: item.start_page):
            child_end = child.end_page if child.end_page is not None else child.start_page
            if cursor < child.start_page:
                label = (
                    f"Opening pages of {parent_label}"
                    if cursor == range_start
                    else f"Unsectioned pages in {parent_label}"
                )
                targets.append(
                    SearchTarget(
                        key=f"gap_{cursor}_{child.start_page - 1}",
                        title=label,
                        start_page=cursor,
                        end_page=child.start_page - 1,
                        node=None,
                        description=(
                            f"{label}. Physical PDF pages {cursor} through "
                            f"{child.start_page - 1}. These pages occur directly inside "
                            f"{parent_label}, before the next named ToC subsection."
                        ),
                    )
                )
            targets.append(
                SearchTarget(
                    key=f"node_{child.id}",
                    title=child.title,
                    start_page=child.start_page,
                    end_page=child_end,
                    node=child,
                    description=child.description(),
                )
            )
            cursor = max(cursor, child_end + 1)

        if cursor <= range_end:
            label = f"Unsectioned ending pages of {parent_label}"
            targets.append(
                SearchTarget(
                    key=f"gap_{cursor}_{range_end}",
                    title=label,
                    start_page=cursor,
                    end_page=range_end,
                    node=None,
                    description=(
                        f"{label}. Physical PDF pages {cursor} through {range_end}. "
                        f"These pages occur directly inside {parent_label}, after its "
                        "named ToC subsections."
                    ),
                )
            )
        return targets

    def _rank_targets(
        self, targets: list[SearchTarget], path: list[str]
    ) -> list[tuple[SearchTarget, float]]:
        if len(targets) > MAX_CHOICE_OPTIONS:
            raise ValueError(
                f"{' > '.join(path or ['Document root'])} has {len(targets)} targets; "
                f"Choice supports at most {MAX_CHOICE_OPTIONS}."
            )
        if len(targets) == 1:
            ranked = [(targets[0], 1.0)]
            confidence = 1.0
        else:
            by_key = {f"target_{i}": target for i, target in enumerate(targets)}
            response = self._system_one(
                purpose="rank_toc_children",
                state={
                    "question": self.question,
                    "current_toc_path": path or ["Document root"],
                    "missing_information": getattr(self, "missing_information", ""),
                },
                questions={
                    "next_section": Choice(
                        instructions=(
                            "Rank which immediate ToC child or uncovered parent-page range "
                            "should be searched first for "
                            "concrete evidence answering the question. This judgment only "
                            "orders an exhaustive depth-first search; it discards nothing."
                        ),
                        criteria={key: target.description for key, target in by_key.items()},
                    )
                },
            )
            answer = response.answers["next_section"]
            probabilities = {
                key: float(value) for key, value in answer.probabilities.items()
            }
            confidence = float(getattr(answer, "confidence", 0.0) or 0.0)
            ranked = sorted(
                ((target, probabilities.get(key, 0.0)) for key, target in by_key.items()),
                key=lambda pair: pair[1],
                reverse=True,
            )

        self._emit(
            {
                "type": "choice",
                "path": path or ["Document root"],
                "confidence": confidence,
                "ranking": [
                    {
                        "title": node.title,
                        "start_page": node.start_page,
                        "end_page": node.end_page,
                        "probability": probability,
                    }
                    for node, probability in ranked
                ],
            }
        )
        print(f"\nCHOICE  {' > '.join(path or ['Document root'])}  confidence={confidence:.3f}")
        for index, (node, probability) in enumerate(ranked, 1):
            print(
                f"  {index:>2}. {probability:.3f}  {node.title} "
                f"[PDF p{node.start_page}–p{node.end_page}]"
            )
        return ranked

    def _search_target(
        self, target: SearchTarget, parent_path: list[str], depth: int
    ) -> RetrievalResult | None:
        path = parent_path + [target.title]
        if target.node is None or not target.node.children:
            result = self._scan_leaf(target.start_page, target.end_page, path)
        else:
            result = self._search_node(target.node, path, depth + 1)
        if result is None:
            self._emit({"type": "backtrack", "path": path})
            print(f"BACK    {' > '.join(path)}")
        return result

    def _search_node(self, node: TocNode, path: list[str], depth: int) -> RetrievalResult | None:
        if depth > self.max_depth:
            raise ValueError(f"ToC exceeded max depth {self.max_depth} at {' > '.join(path)}")
        if not node.children:
            end = node.end_page if node.end_page is not None else node.start_page
            return self._scan_leaf(node.start_page, end, path)
        node_end = node.end_page if node.end_page is not None else node.start_page
        targets = self._targets_for(
            parent=node,
            children=node.children,
            range_start=node.start_page,
            range_end=node_end,
            parent_label=node.title,
        )
        for target, probability in self._rank_targets(targets, path):
            self._emit(
                {
                    "type": "attempt",
                    "parent_path": path,
                    "target": target.title,
                    "local_choice_probability": probability,
                }
            )
            result = self._search_target(target, path, depth)
            if result is not None:
                return result
        return None

    def search(self) -> RetrievalResult | None:
        root_targets = self._targets_for(
            parent=None,
            children=self.tree,
            range_start=min(self.pages),
            range_end=max(self.pages),
            parent_label="Document root",
        )
        for target, probability in self._rank_targets(root_targets, []):
            self._emit(
                {
                    "type": "attempt",
                    "parent_path": ["Document root"],
                    "target": target.title,
                    "local_choice_probability": probability,
                }
            )
            result = self._search_target(target, [], 0)
            if result is not None:
                return result
        return None

    def _scan_leaf(self, start_page, end_page, path):
        raise NotImplementedError("Use JevDFSRetriever or JevExhaustiveRetriever")
