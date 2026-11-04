"""Comparable traversal policies over the same page checks and answer gate."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from openai import OpenAI
from pydantic import BaseModel, Field

from datalab_dfs_search import SearchHit, SearchTarget, TocNode
from evaluated_dfs import EvaluatedDFSSearcher


@dataclass(frozen=True)
class TocSelectionResult:
    selected_ids: list[str]
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0


class SectionSelection(BaseModel):
    selected_ids: list[str] = Field(
        description=(
            "IDs of every section that may contain concrete evidence needed to answer "
            "the question, ordered from most to least promising."
        )
    )


def select_sections_with_openai(
    *,
    api_key: str,
    model: str,
    question: str,
    candidates: list[dict[str, Any]],
) -> TocSelectionResult:
    started = time.perf_counter()
    response = OpenAI(api_key=api_key).responses.parse(
        model=model,
        store=False,
        input=[
            {
                "role": "system",
                "content": (
                    "Select table-of-contents sections to inspect for the user's "
                    "question. Use only the supplied titles, paths, and physical page "
                    "ranges. Select every plausibly relevant section, but do not answer "
                    "the question. Return only candidate IDs that exist in the input."
                ),
            },
            {
                "role": "user",
                "content": {"question": question, "toc_leaf_candidates": candidates},
            },
        ],
        text_format=SectionSelection,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise RuntimeError("OpenAI returned no parsed ToC selection")
    allowed = {candidate["id"] for candidate in candidates}
    selected = []
    for identifier in parsed.selected_ids:
        if identifier in allowed and identifier not in selected:
            selected.append(identifier)
    usage = getattr(response, "usage", None)
    return TocSelectionResult(
        selected_ids=selected,
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        latency_seconds=time.perf_counter() - started,
    )


class LeafTargetMixin:
    """Enumerate the exact same named sections and uncovered page gaps as DFS."""

    pages: dict[int, Any]
    tree: list[TocNode]

    def leaf_targets(self) -> list[tuple[list[str], SearchTarget]]:
        leaves: list[tuple[list[str], SearchTarget]] = []

        def descend(target: SearchTarget, parent_path: list[str]) -> None:
            path = parent_path + [target.title]
            if target.node is None or not target.node.children:
                leaves.append((path, target))
                return
            end = target.node.end_page or target.node.start_page
            children = self._targets_for(
                parent=target.node,
                children=target.node.children,
                range_start=target.node.start_page,
                range_end=end,
                parent_label=target.node.title,
            )
            for child in children:
                descend(child, path)

        roots = self._targets_for(
            parent=None,
            children=self.tree,
            range_start=min(self.pages),
            range_end=max(self.pages),
            parent_label="Document root",
        )
        for root in roots:
            descend(root, [])
        return sorted(leaves, key=lambda item: (item[1].start_page, item[1].end_page, item[0]))


class ExhaustiveSectionSearcher(LeafTargetMixin, EvaluatedDFSSearcher):
    """Inspect every leaf range in document order, then draft once."""

    strategy_name = "exhaustive_sections"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._defer_answer = True

    def _try_answer(self, path: list[str]) -> bool:
        if self._defer_answer:
            return False
        return super()._try_answer(path)

    def search(self) -> SearchHit | None:
        leaves = self.leaf_targets()
        self._emit({"type": "exhaustive_plan", "leaf_range_count": len(leaves)})
        for path, target in leaves:
            self._scan_leaf(target.start_page, target.end_page, path)
        self.exhausted = True
        self._progress(["Document root"])
        self._defer_answer = False
        if super()._try_answer(["Document root"]):
            ranked = sorted(
                self.partial_pages.values(),
                key=lambda result: (result.sufficient, result.evidence),
                reverse=True,
            )
            if ranked:
                best = ranked[0]
                return SearchHit(
                    path=["Document root"],
                    candidate_page=best.page,
                    evidence_pages=sorted(self.evidence_pages),
                    evidence=best.evidence,
                    sufficient=best.sufficient,
                    snippets={page: text[:1500] for page, text in self.evidence_pages.items()},
                )
        self._emit({"type": "exhausted", "path": ["Document root"], "coverage": self.coverage()})
        return None


class LLMSelectedSectionSearcher(LeafTargetMixin, EvaluatedDFSSearcher):
    """Have one LLM call choose ToC leaves, then use the shared evidence pipeline."""

    strategy_name = "llm_toc_selection"

    def __init__(
        self,
        *,
        selector: Callable[..., TocSelectionResult] = select_sections_with_openai,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.selector = selector
        self.selector_calls = 0
        self.selector_input_tokens = 0
        self.selector_output_tokens = 0
        self.selector_latency_seconds = 0.0

    def selector_usage(self) -> dict[str, Any]:
        return {
            "selector_calls": self.selector_calls,
            "selector_input_tokens": self.selector_input_tokens,
            "selector_output_tokens": self.selector_output_tokens,
            "selector_latency_seconds": round(self.selector_latency_seconds, 6),
        }

    def search(self) -> SearchHit | None:
        leaves = self.leaf_targets()
        by_id = {f"leaf_{index:04d}": item for index, item in enumerate(leaves)}
        candidates = [
            {
                "id": identifier,
                "toc_path": path,
                "physical_pages": [target.start_page, target.end_page],
            }
            for identifier, (path, target) in by_id.items()
        ]
        selection = self.selector(
            api_key=self.api_key,
            model=self.answer_model,
            question=self.question,
            candidates=candidates,
        )
        self.selector_calls += 1
        self.selector_input_tokens += selection.input_tokens
        self.selector_output_tokens += selection.output_tokens
        self.selector_latency_seconds += selection.latency_seconds
        self._emit(
            {
                "type": "llm_toc_selection",
                "candidate_count": len(candidates),
                "selected_ids": selection.selected_ids,
                "selected_paths": [by_id[item][0] for item in selection.selected_ids],
            }
        )
        for identifier in selection.selected_ids:
            path, target = by_id[identifier]
            hit = self._scan_leaf(target.start_page, target.end_page, path)
            if hit is not None:
                return hit
        return None

