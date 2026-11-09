"""Freeze 20 existing annotated questions before any repeated retrieval runs."""

import argparse
import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

from retrieval.tree import load_grounded_document
from retrieval.jev import JevDFSRetriever

ROOT = Path(__file__).resolve().parents[1]
SELECTION = {
    "AMAZON_2017_10K.pdf": [1037, 1040, 1041, 1042, 1043],
    "BESTBUY_2023_10K.pdf": [1052, 1057, 1058, 1060, 1061],
    "ACTIVISIONBLIZZARD_2019_10K.pdf": [988, 989, 990, 991, 992],
    "User_Manual_1500S_Classic_EN.pdf": [470, 472, 476, 477, 478],
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(dataset, output):
    samples = json.loads((dataset / "data/samples.json").read_text())
    types = {(r["doc_id"], r["question"]): r for r in json.loads((dataset / "data/question_types.json").read_text())}
    previous = json.loads((ROOT / "runs/evaluation/pageindex-oss-manifest.json").read_text())
    old = {(r["doc_id"], r["question"]): r for r in previous["questions"]}
    doc_meta = {d["doc_id"]: d for d in previous["documents"]}
    questions, documents = [], []
    for doc, indices in SELECTION.items():
        grounded = ROOT / "runs/evaluation" / Path(doc).stem / "grounded"
        pages, tree = load_grounded_document(grounded)
        search = JevDFSRetriever(typesafe=None, pages=pages, tree=tree, question="Preflight only", workers=8)
        windows = search._combined_windows(sorted(p for p, r in pages.items() if r.markdown.strip()))
        selected = []
        for i in indices:
            row = samples[i]
            assert row["doc_id"] == doc, "Upstream row ordering changed; review selection instead of silently replacing questions"
            gold = ast.literal_eval(row["evidence_pages"])
            assert gold and all(p in pages and pages[p].markdown.strip() for p in gold)
            existing = old.get((doc, row["question"]))
            qid = existing["id"] if existing else "mlb2-" + hashlib.sha256((doc + "\n" + row["question"]).encode()).hexdigest()[:12]
            labels = types[(doc, row["question"])]
            selected.append(qid)
            questions.append({
                "id": qid, **{k: row[k] for k in ("doc_id", "doc_type", "question", "answer", "answer_format")},
                "evidence_pages": gold, "evidence_sources": ast.literal_eval(row["evidence_sources"]),
                "answerable": True, "task_type": labels["task_type"],
                "upstream_requires_visual": labels["requires_visual"],
                "in_pageindex_oss_subset": existing is not None,
                "source": "MMLongBench-Doc-V2", "source_row_index": i,
                "reference_page_text_sha256": {str(p): hashlib.sha256(pages[p].markdown.encode()).hexdigest() for p in gold},
            })
        documents.append({"doc_id": doc, "page_count": len(pages),
                          "nonempty_pages": sum(bool(r.markdown.strip()) for r in pages.values()),
                          "root_options": len(search._targets_for(parent=None, children=tree, range_start=1, range_end=max(pages), parent_label="Document root")),
                          "question_ids": selected, "pdf_sha256": sha(Path(doc_meta[doc]["source_pdf"])),
                          "pages_sha256": sha(grounded / "pages.jsonl"), "toc_sha256": sha(grounded / "toc_entries.json"),
                          "preflight_combined_windows": len(windows), "tree": [asdict(n) for n in tree]})
    manifest = {
        "schema_version": 1, "cohort": "retrieval-repeated-2026-11-08", "experiment": "retrieval-v2-combined",
        "selected_on": "2026-11-08", "selection_policy": {
            "documents": "Four already parsed PageIndex-OSS PDFs with visible ToCs: 3 financial reports and 1 manual, 75–198 pages; convenience cohort, not representative sampling.",
            "questions": "Retain all eight questions from the prior OSS subset, then select 12 existing V2 annotations to reach five per document, adding table and multi-page evidence. No new questions or labels generated. Selection frozen before this cohort's runs.",
            "scope": "Answerable text/table evidence only. Parsed supporting pages inspected before execution. Formula/count questions test retrieval of the cited inputs, not answer calculation. No image-only or absent-evidence cases in this cohort.",
            "annotation_limit": "Reference pages are the upstream annotation, not all possible valid evidence locations; exact page recall can penalize alternative/repeated evidence.",
        },
        "source": {"url": "https://github.com/VectifyAI/MMLongBench-Doc-V2",
                   "commit": subprocess.check_output(["git", "-C", str(dataset), "rev-parse", "HEAD"], text=True).strip(),
                   "samples_sha256": sha(dataset / "data/samples.json"),
                   "pageindex_pdf_source": previous["source"]},
        "strategies": ["jev_dfs", "exhaustive", "sol_only"], "repeats": 3, "seed": 20261108,
        "workers": 8, "models": {"jev": "jev-1.13.0", "sol": "gpt-6.1-sol"},
        "max_navigation_steps": 32, "max_pages_per_read": 8, "max_read_tokens": 12000,
        "output_budget": {"max_pages": 8, "max_tokens": 12000}, "page_judgments_per_target": 4,
        "documents": documents, "questions": questions,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("/private/tmp/mmlongbench-doc-v2"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = prepare(args.dataset, args.output)
    print(json.dumps({"questions": len(data["questions"]), "documents": len(data["documents"]),
                      "upstream_oss_questions": sum(q["in_pageindex_oss_subset"] for q in data["questions"])}))
