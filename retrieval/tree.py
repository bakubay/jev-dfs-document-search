"""Grounded document types, ToC construction, and the targets beneath a ToC node.

Search order and page checks live in retrieval.jev; Sol reuses `child_targets`.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
import tiktoken

MODEL = "jev-1.13.0"
ENCODER = tiktoken.get_encoding("o200k_base")


@dataclass
class TocNode:
    id: str
    title: str
    printed_page: int
    start_page: int
    end_page: int | None = None
    children: list[TocNode] = field(default_factory=list)

    @property
    def end(self) -> int:
        return self.end_page if self.end_page is not None else self.start_page

    def description(self) -> str:
        description = f"{self.title}. Physical PDF pages {self.start_page} through {self.end}."
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
class PageCheck:
    page: int
    evidence: float

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

def child_targets(
    node: TocNode | None, tree: list[TocNode], pages: dict[int, PageRecord]
) -> list[SearchTarget]:
    """Children of `node` (None = document root), plus the parent pages they skip.

    Gap targets cover every page of the parent's range that no child covers, so
    walking these targets recursively reaches every physical page.
    """
    if node is None:
        children, range_start, range_end, label = tree, min(pages), max(pages), "Document root"
    else:
        children, range_start, range_end, label = node.children, node.start_page, node.end, node.title
    targets: list[SearchTarget] = []
    cursor = range_start
    for child in sorted(children, key=lambda item: item.start_page):
        if cursor < child.start_page:
            title = f"Opening pages of {label}" if cursor == range_start else f"Unsectioned pages in {label}"
            targets.append(SearchTarget(
                key=f"gap_{cursor}_{child.start_page - 1}", title=title,
                start_page=cursor, end_page=child.start_page - 1, node=None,
                description=(
                    f"{title}. Physical PDF pages {cursor} through "
                    f"{child.start_page - 1}. These pages occur directly inside "
                    f"{label}, before the next named ToC subsection."
                ),
            ))
        targets.append(SearchTarget(
            key=f"node_{child.id}", title=child.title, start_page=child.start_page,
            end_page=child.end, node=child, description=child.description(),
        ))
        cursor = max(cursor, child.end + 1)
    if cursor <= range_end:
        title = f"Unsectioned ending pages of {label}"
        targets.append(SearchTarget(
            key=f"gap_{cursor}_{range_end}", title=title,
            start_page=cursor, end_page=range_end, node=None,
            description=(
                f"{title}. Physical PDF pages {cursor} through {range_end}. "
                f"These pages occur directly inside {label}, after its "
                "named ToC subsections."
            ),
        ))
    return targets
