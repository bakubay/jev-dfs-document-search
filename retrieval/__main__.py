"""Search a prepared document with one of the three retrieval methods."""
import argparse
import json
import os
from pathlib import Path

from typesafe_sdk import RetryPolicy, TypeSafeClient

from retrieval.config import DEFAULT_OPENAI_MODEL, load_local_env
from retrieval.jev import JevDFSRetriever, JevExhaustiveRetriever
from retrieval.run import run_retrieval
from retrieval.sol import SolRetriever
from retrieval.tree import MODEL, load_grounded_document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grounded_dir", type=Path)
    parser.add_argument("question")
    parser.add_argument("--method", choices=("jev_dfs", "exhaustive", "sol_only"), default="jev_dfs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(__file__).resolve().parents[1] / ".env")
    parser.add_argument("--max-navigation-steps", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new path")
    if args.max_navigation_steps < 1:
        parser.error("Navigation budget must be positive")
    pages, tree = load_grounded_document(args.grounded_dir)
    load_local_env(args.env_file)
    sol = args.method == "sol_only"
    key_name = "OPENAI_API_KEY" if sol else "TYPESAFE_API_KEY"
    if not os.environ.get(key_name):
        parser.error(f"{key_name} is required")
    client = None if sol else TypeSafeClient(api_key=os.environ[key_name], model=MODEL,
        retry=RetryPolicy(max_retries=5, backoff_initial=1.0, backoff_max=20.0))
    try:
        if sol:
            search = SolRetriever(pages=pages, tree=tree, question=args.question,
                api_key=os.environ[key_name], model=os.environ.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL,
                max_navigation_steps=args.max_navigation_steps)
        else:
            cls = JevDFSRetriever if args.method == "jev_dfs" else JevExhaustiveRetriever
            search = cls(typesafe=client, pages=pages, tree=tree, question=args.question)
        result, elapsed, failure = run_retrieval(search)
        record = {"method": args.method, "question": args.question,
                  "retrieval": result.to_dict(), "elapsed_seconds": elapsed, "failure": failure,
                  "usage": search.usage_summary(), "trace": search.trace, "request_trace": search.request_trace}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(record, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        print(json.dumps({"output": str(args.output), "pages": list(result.evidence_pages),
                          "stop": result.stop_reason, "seconds": elapsed, "failure": failure}))
        return int(failure is not None)
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
