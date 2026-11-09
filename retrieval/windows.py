"""Whole-page read budgets shared by JEV and the Sol-only agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

MAX_READ_PAGES = 8
MAX_READ_TOKENS = 12_000


@dataclass(frozen=True)
class PageWindow:
    targets: tuple[int, ...]
    pages: dict[int, str]
    text_tokens: int


def pack_page_windows(
    groups: Iterable[tuple[int, Iterable[int]]], text: dict[int, str], encoder: Any,
    *, max_pages: int = MAX_READ_PAGES, max_tokens: int = MAX_READ_TOKENS,
) -> Iterator[PageWindow]:
    """Keep targets in order; each target's complete context must fit together.

    A Sol read target has one page; a JEV target includes its neighbors.
    Count the union of actual source pages, never just the candidate count.
    Nothing is truncated or omitted to make a group fit.
    """
    if min(max_pages, max_tokens) < 1:
        raise ValueError("Read budgets must be positive")
    targets, current, used = [], {}, 0
    for target, context in groups:
        group = {p: text[p] for p in dict.fromkeys(context)}
        candidate = {**current, **group}
        tokens = len(encoder.encode(json.dumps(candidate, ensure_ascii=False)))
        if current and (len(candidate) > max_pages or tokens > max_tokens):
            yield PageWindow(tuple(targets), current, used)
            targets, current = [], {}
            candidate = group
            tokens = len(encoder.encode(json.dumps(candidate, ensure_ascii=False)))
        if len(candidate) > max_pages or tokens > max_tokens:
            raise ValueError(
                f"Full context for PDF page {target} exceeds the read budget "
                f"({len(candidate)} pages / {tokens} tokens); it was not truncated or checked"
            )
        targets.append(target)
        current, used = candidate, tokens
    if current:
        yield PageWindow(tuple(targets), current, used)
