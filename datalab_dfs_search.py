#!/usr/bin/env python3
"""Search Datalab-grounded PDF pages with JEV-ordered depth-first search.

No PageIndex tree, embeddings, or generated section summaries are used. The
document's own table of contents supplies the hierarchy, Datalab supplies the
physical pages, and JEV supplies only bounded semantic judgments:

1. Choice orders the immediate children at the current ToC node.
2. Every sibling stays available; Choice never prunes a branch.
3. Every page in a leaf section is screened independently with two Nouls.
4. Candidate pages are verified with adjacent-page context.
5. Only a sufficient-evidence Noul may stop the search; otherwise DFS backs up.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, TypeVar

import tiktoken
from typesafe_sdk import Choice, Noul, RetryPolicy, TypeSafeClient

from datalab_parse import load_local_env
from openai_grounded_answer import DEFAULT_OPENAI_MODEL

MODEL = "jev-1.13.0"
DEFAULT_WORKERS = 8
DEFAULT_RECALL_THRESHOLD = 0.35
DEFAULT_VERIFY_THRESHOLD = 0.50
DEFAULT_MAX_VERIFY_PER_LEAF = 8
MAX_CHOICE_OPTIONS = 255
MAX_PAGE_TOKENS = 24_000
VERIFY_CENTER_TOKENS = 16_000
VERIFY_NEIGHBOR_TOKENS = 5_000

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


@dataclass(frozen=True)
class SearchHit:
    path: list[str]
    candidate_page: int
    evidence_pages: list[int]
    evidence: float
    sufficient: float
    snippets: dict[int, str]


def truncate_tokens(text: str, limit: int, encoder: Any) -> str:
    tokens = encoder.encode(text)
    return text if len(tokens) <= limit else encoder.decode(tokens[:limit])


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


def _number_path(title: str) -> tuple[int, ...] | None:
    match = re.match(r"^\s*(\d+(?:\.\d+)*)[.\s]", title)
    return tuple(int(part) for part in match.group(1).split(".")) if match else None


def build_toc_tree(entries: list[dict[str, Any]], page_count: int) -> list[TocNode]:
    """Recover a hierarchy from numbered ToC titles and explicit Part containers."""
    roots: list[TocNode] = []
    active_part: TocNode | None = None
    numbered: dict[tuple[int, ...], TocNode] = {}

    for index, entry in enumerate(entries):
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
            # A new peer invalidates deeper paths from the previous branch.
            for path in list(numbered):
                if len(path) > len(number_path) and path[: len(number_path)] != number_path:
                    numbered.pop(path)
        else:
            roots.append(node)

    def assign_ranges(siblings: list[TocNode], parent_end: int) -> None:
        for position, node in enumerate(siblings):
            next_start = (
                siblings[position + 1].start_page
                if position + 1 < len(siblings)
                else parent_end + 1
            )
            node.end_page = max(node.start_page, next_start - 1)
            assign_ranges(node.children, node.end_page)

    assign_ranges(roots, page_count)
    return roots


def load_grounded_document(directory: Path) -> tuple[dict[int, PageRecord], list[TocNode]]:
    pages = load_pages(directory / "pages.jsonl")
    entries = json.loads((directory / "toc_entries.json").read_text(encoding="utf-8"))
    return pages, build_toc_tree(entries, max(pages))


class OrderedDFSSearcher:
    def __init__(
        self,
        *,
        typesafe: Any,
        pages: dict[int, PageRecord],
        tree: list[TocNode],
        question: str,
        workers: int = DEFAULT_WORKERS,
        recall_threshold: float = DEFAULT_RECALL_THRESHOLD,
        verify_threshold: float = DEFAULT_VERIFY_THRESHOLD,
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
        self.verify_threshold = verify_threshold
        self.max_verify_per_leaf = max_verify_per_leaf
        self.max_depth = max_depth
        self.encoder = tiktoken.get_encoding("o200k_base")
        self.trace: list[dict[str, Any]] = []
        self.event_sink = event_sink
        self.partial_pages: dict[int, PageVerification] = {}
        self.api_input_tokens = 0
        self.usage_lock = Lock()

    def _emit(self, event: dict[str, Any]) -> None:
        """Record a search hop and optionally stream it to a UI or logger."""
        self.trace.append(event)
        if self.event_sink is not None:
            self.event_sink(event)

    def _record_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        tokens = int(getattr(usage, "input_tokens", 0) or 0)
        with self.usage_lock:
            self.api_input_tokens += tokens

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
            response = self.typesafe.system_one(
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
            self._record_usage(response)
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

    def _screen_page(self, page: int) -> PageScreen:
        record = self.pages[page]
        response = self.typesafe.system_one(
            state={
                "question": self.question,
                "physical_pdf_page": page,
                "page_text": truncate_tokens(record.markdown, MAX_PAGE_TOKENS, self.encoder),
            },
            questions={
                "direct": Noul(
                    instructions=(
                        "Does this page directly state at least one concrete fact required "
                        "to answer the question? A mere topical mention is false."
                    )
                ),
                "partial": Noul(
                    instructions=(
                        "Does this page contain partial evidence genuinely useful for "
                        "answering the question, even if an adjacent page is also needed?"
                    )
                ),
            },
        )
        self._record_usage(response)
        return PageScreen(
            page=page,
            direct=float(response.answers["direct"].noul),
            partial=float(response.answers["partial"].noul),
        )

    def _context_for(self, page: int) -> str:
        blocks = []
        for number in (page - 1, page, page + 1):
            record = self.pages.get(number)
            if not record or not record.markdown:
                continue
            limit = VERIFY_CENTER_TOKENS if number == page else VERIFY_NEIGHBOR_TOKENS
            text = truncate_tokens(record.markdown, limit, self.encoder)
            blocks.append(f"--- Physical PDF page {number} ---\n{text}")
        return "\n\n".join(blocks)

    def _verify_page(self, screen: PageScreen) -> PageVerification:
        response = self.typesafe.system_one(
            state={
                "question": self.question,
                "candidate_physical_pdf_page": screen.page,
                "candidate_with_adjacent_context": self._context_for(screen.page),
            },
            questions={
                "evidence": Noul(
                    instructions=(
                        "Does the candidate page, interpreted with its adjacent pages, "
                        "contain concrete evidence relevant to the question?"
                    )
                ),
                "sufficient": Noul(
                    instructions=(
                        "Does this candidate and adjacent-page context contain enough "
                        "concrete information to answer the entire question?"
                    )
                ),
            },
        )
        self._record_usage(response)
        return PageVerification(
            page=screen.page,
            direct=screen.direct,
            partial=screen.partial,
            evidence=float(response.answers["evidence"].noul),
            sufficient=float(response.answers["sufficient"].noul),
        )

    def _scan_leaf(
        self, start_page: int, end_page: int, path: list[str]
    ) -> SearchHit | None:
        available = [
            page
            for page in range(start_page, end_page + 1)
            if page in self.pages and self.pages[page].markdown
        ]
        print(
            f"\nSCAN    {' > '.join(path)}  PDF p{start_page}–p{end_page} "
            f"({len(available)} text pages)"
        )
        if not available:
            self._emit(
                {
                    "type": "leaf_scan",
                    "path": path,
                    "start_page": start_page,
                    "end_page": end_page,
                    "status": "no_text",
                    "screens": [],
                    "verifications": [],
                }
            )
            return None

        screens = sorted(
            run_parallel(available, self.workers, self._screen_page),
            key=lambda result: result.recall_score,
            reverse=True,
        )
        candidates = [
            screen for screen in screens if screen.recall_score >= self.recall_threshold
        ][: self.max_verify_per_leaf]
        fallback_used = False
        if not candidates:
            candidates = screens[: min(2, self.max_verify_per_leaf)]
            fallback_used = True

        verifications = sorted(
            run_parallel(candidates, self.workers, self._verify_page),
            key=lambda result: result.sufficient,
            reverse=True,
        )
        for result in verifications:
            existing = self.partial_pages.get(result.page)
            if existing is None or result.evidence > existing.evidence:
                self.partial_pages[result.page] = result
            print(
                f"  PDF p{result.page}: direct={result.direct:.3f} "
                f"partial={result.partial:.3f} evidence={result.evidence:.3f} "
                f"sufficient={result.sufficient:.3f}"
            )

        sufficient = [
            result
            for result in verifications
            if result.sufficient >= self.verify_threshold
        ]
        self._emit(
            {
                "type": "leaf_scan",
                "path": path,
                "start_page": start_page,
                "end_page": end_page,
                "status": "sufficient" if sufficient else "failed",
                "fallback_candidates_used": fallback_used,
                "screens": [
                    {**asdict(screen), "recall_score": screen.recall_score}
                    for screen in screens
                ],
                "verifications": [asdict(result) for result in verifications],
            }
        )
        if not sufficient:
            print("  No candidate passed sufficiency; backtrack to the next sibling.")
            return None

        best = sufficient[0]
        evidence_pages = [
            page for page in (best.page - 1, best.page, best.page + 1) if page in self.pages
        ]
        return SearchHit(
            path=path,
            candidate_page=best.page,
            evidence_pages=evidence_pages,
            evidence=best.evidence,
            sufficient=best.sufficient,
            snippets={page: self.pages[page].markdown[:1_500] for page in evidence_pages},
        )

    def _search_target(
        self, target: SearchTarget, parent_path: list[str], depth: int
    ) -> SearchHit | None:
        path = parent_path + [target.title]
        if target.node is None or not target.node.children:
            result = self._scan_leaf(target.start_page, target.end_page, path)
        else:
            result = self._search_node(target.node, path, depth + 1)
        if result is None:
            self._emit({"type": "backtrack", "path": path})
            print(f"BACK    {' > '.join(path)}")
        return result

    def _search_node(self, node: TocNode, path: list[str], depth: int) -> SearchHit | None:
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

    def search(self) -> SearchHit | None:
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

    def best_partial_results(self, limit: int = 10) -> list[dict[str, Any]]:
        ranked = sorted(
            self.partial_pages.values(),
            key=lambda result: (result.sufficient, result.evidence),
            reverse=True,
        )
        return [asdict(result) for result in ranked[:limit]]


def serialize_tree(nodes: list[TocNode]) -> list[dict[str, Any]]:
    return [
        {
            "id": node.id,
            "title": node.title,
            "printed_page": node.printed_page,
            "start_page": node.start_page,
            "end_page": node.end_page,
            "children": serialize_tree(node.children),
        }
        for node in nodes
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("grounded_dir", type=Path)
    parser.add_argument("question")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--recall-threshold", type=float, default=DEFAULT_RECALL_THRESHOLD)
    parser.add_argument("--verify-threshold", type=float, default=DEFAULT_VERIFY_THRESHOLD)
    parser.add_argument("--max-verify-per-leaf", type=int, default=DEFAULT_MAX_VERIFY_PER_LEAF)
    parser.add_argument("--max-depth", type=int, default=64)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--answer",
        action="store_true",
        help="Draft with Sol, evaluate with JEV, and continue DFS until accepted or exhausted.",
    )
    parser.add_argument(
        "--openai-model",
        default=None,
        help="OpenAI answer model; defaults to OPENAI_MODEL or gpt-6.1-sol.",
    )
    parser.add_argument("--env-file", type=Path, default=Path(__file__).with_name(".env"))
    args = parser.parse_args()
    if not args.grounded_dir.is_dir():
        parser.error(f"Grounded Datalab directory not found: {args.grounded_dir}")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if not 0 <= args.recall_threshold <= 1 or not 0 <= args.verify_threshold <= 1:
        parser.error("thresholds must be between 0 and 1")
    if args.max_verify_per_leaf < 1 or args.max_depth < 1:
        parser.error("max verification and depth values must be positive")
    return args


def main() -> int:
    args = parse_args()
    load_local_env(args.env_file)
    if not os.environ.get("TYPESAFE_API_KEY"):
        raise SystemExit(f"TYPESAFE_API_KEY is missing from {args.env_file}")
    if args.answer and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(f"OPENAI_API_KEY is missing from {args.env_file}")

    pages, tree = load_grounded_document(args.grounded_dir)
    client = TypeSafeClient(
        api_key=os.environ["TYPESAFE_API_KEY"],
        model=MODEL,
        retry=RetryPolicy(max_retries=5, backoff_initial=1.0, backoff_max=20.0),
    )
    search_class = OrderedDFSSearcher
    answer_options = {}
    if args.answer:
        from evaluated_dfs import EvaluatedDFSSearcher

        search_class = EvaluatedDFSSearcher
        answer_options = {
            "api_key": os.environ["OPENAI_API_KEY"],
            "answer_model": args.openai_model or os.environ.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL,
        }
    searcher = search_class(
        typesafe=client,
        pages=pages,
        tree=tree,
        question=args.question,
        workers=args.workers,
        recall_threshold=args.recall_threshold,
        verify_threshold=args.verify_threshold,
        max_verify_per_leaf=args.max_verify_per_leaf,
        max_depth=args.max_depth,
        **answer_options,
    )
    started = time.perf_counter()
    hit = searcher.search()
    elapsed = time.perf_counter() - started
    grounded_answer = searcher.final_answer() if args.answer else None
    result = {
        "question": args.question,
        "model": MODEL,
        "strategy": "Datalab-grounded ToC + JEV-ordered exhaustive DFS + Noul verification",
        "found": hit is not None,
        "hit": asdict(hit) if hit else None,
        "grounded_answer": grounded_answer,
        "best_partial_results": searcher.best_partial_results(),
        "input_tokens": searcher.api_input_tokens,
        "elapsed_seconds": round(elapsed, 3),
        "settings": {
            "workers": args.workers,
            "recall_threshold": args.recall_threshold,
            "verify_threshold": args.verify_threshold,
            "max_verify_per_leaf": args.max_verify_per_leaf,
        },
        "toc_tree": serialize_tree(tree),
        "trace": searcher.trace,
        "coverage": searcher.coverage() if args.answer else None,
        "answer_attempts": searcher.answer_attempts if args.answer else [],
    }
    print("\n" + "=" * 72)
    if hit:
        print("FOUND   " + " > ".join(hit.path))
        print(f"CENTER  physical PDF page {hit.candidate_page}")
        print("PAGES   " + ", ".join(str(page) for page in hit.evidence_pages))
        print(f"SCORES  evidence={hit.evidence:.3f} sufficient={hit.sufficient:.3f}")
        if grounded_answer:
            print("\nANSWER")
            print(grounded_answer["answer"])
            for citation in grounded_answer["citations"]:
                print(
                    f"  [PDF p{citation['physical_pdf_page']}] "
                    f"{citation['quote']}"
                )
            if grounded_answer["limitations"]:
                print(f"LIMITS  {grounded_answer['limitations']}")
    else:
        print("NOT FOUND: every reachable ToC branch was exhausted.")
        if grounded_answer:
            print(grounded_answer["answer"])
            print("Coverage:", grounded_answer["coverage"]["checked_count"], "/",
                  grounded_answer["coverage"]["total_pages"])
    print(f"Elapsed: {elapsed:.2f}s · JEV input tokens: {searcher.api_input_tokens}")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Trace written to {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
