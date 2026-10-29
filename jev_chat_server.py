#!/usr/bin/env python3
"""Local streaming chat UI for the grounded JEV document search demo."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from typesafe_sdk import RetryPolicy, TypeSafeClient

from datalab_dfs_search import (
    DEFAULT_MAX_VERIFY_PER_LEAF,
    DEFAULT_RECALL_THRESHOLD,
    DEFAULT_VERIFY_THRESHOLD,
    DEFAULT_WORKERS,
    MODEL,
    load_grounded_document,
    serialize_tree,
)
from datalab_parse import load_local_env
from evaluated_dfs import EvaluatedDFSSearcher
from openai_grounded_answer import DEFAULT_OPENAI_MODEL


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    project = here.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grounded-dir",
        type=Path,
        default=project / "work" / "datalab-runs" / "nist-full" / "grounded",
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=project / "outputs" / "jev-search-chat.html",
    )
    parser.add_argument("--env-file", type=Path, default=here / ".env")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    return parser.parse_args()


def make_handler(
    *,
    html_path: Path,
    pages: dict[int, Any],
    tree: list[Any],
    workers: int,
) -> type[BaseHTTPRequestHandler]:
    class ChatHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[chat] {self.address_string()} {format % args}")

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            route = urlparse(self.path).path
            recorded = {
                "/api/recorded/early": "jev-functions-evaluated-trace.json",
                "/api/recorded/late": "jev-late-success-evaluated-trace.json",
                "/api/recorded/absent": "jev-licensing-evaluated-trace.json",
                "/api/recorded-licensing": "jev-licensing-evaluated-trace.json",
            }
            if route in recorded:
                path = html_path.parent / recorded[route]
                if not path.is_file():
                    self._json(404, {"error": "Recorded example not found"})
                else:
                    self._json(200, json.loads(path.read_text(encoding="utf-8")))
                return
            if route not in {"/", "/jev-search-chat.html"}:
                self._json(404, {"error": "Not found"})
                return
            body = html_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/ask":
                self._json(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = json.loads(self.rfile.read(length) or b"{}")
                question = str(raw.get("question", "")).strip()
            except (ValueError, json.JSONDecodeError):
                self._json(400, {"error": "Expected JSON with a question"})
                return
            if not question or len(question) > 2_000:
                self._json(400, {"error": "Question must be 1–2,000 characters"})
                return

            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()

            connected = True

            def send(kind: str, **payload: Any) -> None:
                nonlocal connected
                if not connected:
                    return
                line = json.dumps({"kind": kind, **payload}, ensure_ascii=False) + "\n"
                try:
                    self.wfile.write(line.encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    connected = False

            try:
                send(
                    "started",
                    question=question,
                    model=MODEL,
                    strategy="ToC-grounded ordered DFS",
                    toc_tree=serialize_tree(tree),
                )
                typesafe = TypeSafeClient(
                    api_key=os.environ["TYPESAFE_API_KEY"],
                    model=MODEL,
                    retry=RetryPolicy(
                        max_retries=5,
                        backoff_initial=1.0,
                        backoff_max=20.0,
                    ),
                )
                searcher = EvaluatedDFSSearcher(
                    typesafe=typesafe,
                    pages=pages,
                    tree=tree,
                    question=question,
                    workers=workers,
                    recall_threshold=DEFAULT_RECALL_THRESHOLD,
                    verify_threshold=DEFAULT_VERIFY_THRESHOLD,
                    max_verify_per_leaf=DEFAULT_MAX_VERIFY_PER_LEAF,
                    event_sink=lambda event: send("hop", event=event),
                    api_key=os.environ["OPENAI_API_KEY"],
                    answer_model=os.environ.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL,
                )
                started = time.perf_counter()
                hit = searcher.search()
                search_seconds = round(time.perf_counter() - started, 3)
                send(
                    "search_complete",
                    found=hit is not None,
                    hit=asdict(hit) if hit else None,
                    elapsed_seconds=search_seconds,
                    input_tokens=searcher.api_input_tokens,
                    coverage=searcher.coverage(),
                    evaluation_accepted=searcher.accepted_answer is not None,
                )
                send("complete", **searcher.final_answer())
            except Exception as exc:  # Keep credentials and full payloads out of the UI.
                send("error", message=f"{type(exc).__name__}: search interrupted; no exhaustive conclusion was reached.")

    return ChatHandler


def main() -> int:
    args = parse_args()
    load_local_env(args.env_file)
    missing = [
        name
        for name in ("TYPESAFE_API_KEY", "OPENAI_API_KEY")
        if not os.environ.get(name)
    ]
    if missing:
        raise SystemExit(f"Missing {', '.join(missing)} in {args.env_file}")
    if not args.html.is_file():
        raise SystemExit(f"HTML UI not found: {args.html}")
    pages, tree = load_grounded_document(args.grounded_dir)
    handler = make_handler(
        html_path=args.html,
        pages=pages,
        tree=tree,
        workers=args.workers,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"JEV chat: http://{args.host}:{args.port}")
    print(f"Document: {args.grounded_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
