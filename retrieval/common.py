"""Shared evidence contract and budgets, independent of answer generation."""

from dataclasses import asdict, dataclass
import json


@dataclass(frozen=True)
class EvidenceBudget:
    max_pages: int = 8
    max_tokens: int = 12_000

    def __post_init__(self):
        if min(self.max_pages, self.max_tokens) < 1:
            raise ValueError("Evidence budgets must be positive")

    def tokens(self, pages, encoder):
        return len(encoder.encode(json.dumps(pages, ensure_ascii=False))) if pages else 0

    def fits(self, pages, encoder):
        return len(pages) <= self.max_pages and self.tokens(pages, encoder) <= self.max_tokens


@dataclass(frozen=True)
class RetrievalResult:
    # Dict insertion order is the retriever's evidence order. Text is copied from
    # the parser, never generated. Sufficiency is a model judgment, not a label.
    evidence_pages: dict[int, str]
    stop_reason: str
    sufficient: bool
    coverage: dict
    evidence_tokens: int
    output_budget: dict

    def to_dict(self):
        return asdict(self)


def coverage(pages, *, read=(), verified=(), exhausted=False):
    """read: pages whose text a model saw. verified: pages JEV judged as a target."""
    expected = set(range(1, max(pages) + 1))
    empty = {p for p, row in pages.items() if not row.markdown.strip()}
    nonempty = set(pages) - empty
    read = set(read)
    return {
        "total_pages": len(expected), "nonempty_pages": len(nonempty),
        "read_pages": sorted(read), "verified_pages": sorted(set(verified)),
        "empty_pages": sorted(empty), "missing_pages": sorted(expected - set(pages)),
        "remaining_text_pages": sorted(nonempty - read),
        "exhausted": exhausted,
    }
