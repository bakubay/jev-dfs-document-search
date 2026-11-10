# Evaluate the three approaches

All approaches return source pages. Gold annotations are used only after
retrieval and its timer finish. Approaches, read limits and output limits are
described in [the main README](../README.md).

## The question set

[`cohort.json`](cohort.json) holds twenty answerable questions with annotated
evidence pages over four PageIndex-OSS-Benchmark PDFs. Eight come from the OSS
text-lookup subset and twelve are existing MMLongBench-Doc-V2 annotations
(tables, multi-page evidence, derived values). It also pins the input hashes,
budgets, models, repeats and seed. [`evidence-review.json`](evidence-review.json)
notes questions where valid evidence also appears on unannotated pages.

## Prepare documents

`python -m evaluation.dataset --help` describes importing a local checkout of
[PageIndex-OSS-Benchmark](https://github.com/VectifyAI/PageIndex-OSS-Benchmark).
Its ToC screening needs Poppler's `pdfinfo` and `pdftotext`. Parsing and page
grounding then use `preparation.parse` and `preparation.ground_toc`; parsing
consumes Datalab credits. Prepared documents live under
`runs/evaluation/<PDF name without extension>/grounded/`.

## Run the comparison

```bash
python -m evaluation.run_repeated --output-dir evaluation-results/<new name>
```

Runs every question × approach `repeats` times (180 runs), one query at a time
in seeded shuffled order, and writes each record plus a source snapshot. JEV
needs `TYPESAFE_API_KEY`; Sol needs `OPENAI_API_KEY`. Use `--dry-run` to see the
plan without API calls. `--resume` continues an interrupted run only if inputs,
plan and source code are unchanged; recorded failures are kept, never retried.

`--strategies jev_dfs exhaustive` runs only those approaches (120 runs), e.g. to
rerun JEV after a JEV-only change without spending OpenAI credits. `--repeats 2` runs only the first
two repeat blocks.
`--parallel 4` runs four cases at once (about 4× sooner); each retrieval still
sends one request at a time, but shared network and API load can raise
per-question times slightly.

For a few questions or approaches:

```bash
python -m evaluation.run evaluation/cohort.json --grounded-root runs/evaluation \
  --question-ids pi-oss-ad5dbad521ca pi-oss-b728c1d683c6 \
  --strategies jev_dfs exhaustive --output-dir evaluation-results/<new name>
```

## Build the report

```bash
python -m reporting.build_cohort_report \
  --runs evaluation-results/<run name> \
  --output docs/results/report.html --summary docs/results/summary.md
```

To combine folders, give each one, optionally naming which approaches to take
from it: `--runs evaluation-results/<new> evaluation-results/<old>:sol_only`.
Each folder is checked against its own frozen plan, and an approach may come
from only one folder.

Add `--allow-partial` for an incomplete run; missing runs are listed. The report
checks input and source hashes and shows the code from the run's own snapshot.
Raw records stay local and ignored by Git; the HTML embeds everything it shows.

Recall counts returned annotated pages, separately from pages merely read.
Precision counts how many returned pages are annotated pages. Costs use the
pinned rates in `evaluation/pricing.py`, not invoices.
