"""Run one retrieval and record the measurement boundary."""
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def runtime_sources():
    return [p for folder in ("retrieval", "evaluation", "preparation")
            for p in sorted((ROOT / folder).glob("*.py"))]


def run_retrieval(searcher):
    """The clock contains model navigation, reads, selection and stop decisions."""
    failure = None
    started = time.perf_counter()
    searcher.started_at = started
    try:
        result = searcher.search()
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        result = searcher.result("error")
    elapsed = time.perf_counter() - started
    return result, elapsed, failure
