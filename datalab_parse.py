#!/usr/bin/env python3
"""Parse a PDF with Datalab and preserve page-grounded outputs locally."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from datalab_sdk import DatalabClient
from datalab_sdk.models import ConvertOptions


def load_local_env(path: Path) -> None:
    """Load simple KEY=VALUE pairs without adding another dependency."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not os.environ.get(key):
            os.environ[key] = value.strip().strip('"').strip("'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pages", type=int)
    parser.add_argument(
        "--mode",
        choices=("lite", "fast", "balanced", "accurate"),
        default="balanced",
    )
    parser.add_argument("--env-file", type=Path, default=Path(__file__).with_name(".env"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_local_env(args.env_file)
    if not os.environ.get("DATALAB_API_KEY"):
        raise SystemExit(f"DATALAB_API_KEY is missing from {args.env_file}")
    if not args.pdf.is_file():
        raise SystemExit(f"PDF not found: {args.pdf}")

    args.output.mkdir(parents=True, exist_ok=True)
    options = ConvertOptions(
        max_pages=args.max_pages,
        paginate=True,
        output_format="json,markdown",
        mode=args.mode,
        include_markdown_in_chunks=True,
        disable_image_extraction=True,
        disable_image_captions=True,
    )

    started = time.monotonic()
    result = DatalabClient(api_key=os.environ["DATALAB_API_KEY"]).convert(
        args.pdf, options=options
    )
    elapsed = time.monotonic() - started

    summary = {
        "source_pdf": str(args.pdf.resolve()),
        "requested_max_pages": args.max_pages,
        "requested_mode": args.mode,
        "wall_time_seconds": round(elapsed, 3),
        "success": result.success,
        "status": result.status,
        "error": result.error,
        "page_count": result.page_count,
        "runtime": result.runtime,
        "parse_quality_score": result.parse_quality_score,
        "cost_breakdown": result.cost_breakdown,
        "request_id": result.request_id,
        "versions": result.versions,
        "metadata": result.metadata,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if result.markdown is not None:
        (args.output / "document.md").write_text(result.markdown, encoding="utf-8")
    if result.json is not None:
        (args.output / "document.json").write_text(
            json.dumps(result.json, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # Keep the complete SDK response, excluding image payloads, for debugging.
    raw = asdict(result)
    raw["images"] = sorted((raw.get("images") or {}).keys())
    (args.output / "response.json").write_text(
        json.dumps(raw, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not result.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
