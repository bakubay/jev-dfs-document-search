# Evaluate the three retrieval methods

All methods return source pages. Gold annotations are used only after retrieval
and its timer finish. The policies, read budgets and evidence budgets are defined
in [the main README](../README.md).

## Prepare documents

`python -m evaluation.dataset --help` describes importing a local checkout of
[PageIndex-OSS-Benchmark](https://github.com/VectifyAI/PageIndex-OSS-Benchmark).
Its inexpensive ToC screening requires the `pdfinfo` and `pdftotext` commands
from Poppler. PDF parsing and page grounding then use `preparation.parse` and
`preparation.ground_toc`; parser calls consume Datalab credits.

The retained [study manifest](../docs/results/retrieval-repeated-2026-11-08/manifest.json)
contains twenty known questions and evidence-page labels over four PageIndex
PDFs. Eight questions come from the OSS subset; twelve are existing annotations
from its MMLongBench-Doc-V2 source. `evaluation.prepare_cohort` records that
selection from the pinned upstream dataset; it does not generate questions.

## Run all three methods

Prepared documents live under `runs/evaluation/<PDF filename without extension>/grounded/`.

```bash
python -m evaluation.run \
  docs/results/retrieval-repeated-2026-11-08/manifest.json \
  --grounded-root runs/evaluation \
  --question-ids pi-oss-ad5dbad521ca pi-oss-b728c1d683c6 \
  --output-dir evaluation-results/new-comparison
```

Use `--dry-run` to inspect the schedule without API calls. `--strategies
jev_dfs exhaustive` selects just the JEV methods. Runs use a seeded shuffled
order. Queries run one at a time; workers within a query can run concurrently.

## Repeat the frozen cohort

```bash
python -m evaluation.run_repeated \
  docs/results/retrieval-repeated-2026-11-08/manifest.json \
  --grounded-root runs/evaluation \
  --output-dir evaluation-results/new-repeated-study
```

Each question/method pair runs three times. Use `--resume` only when the input,
source hashes and run plan match. Completed failures remain recorded; they are
not selectively retried. The runner saves a source snapshot with the results.
The repository cleanup changed file hashes, so old studies cannot be resumed
with the consolidated source. Use a new output directory.

## Build offline reports

```bash
python -m reporting.build_cohort_report \
  --runs evaluation-results/new-repeated-study \
  --output docs/traces/new-study.html \
  --summary evaluation-results/new-repeated-study/summary.md
```

Reports verify input and source hashes. Historical reports use the original
`source-snapshot/` beside the run records, never today's source under an old
label. Raw records and snapshots are local and ignored by Git. The committed
HTML includes its recorded data and snippets and works without those files.

`python -m reporting.build_simple_trace` rebuilds the retained coffee timeline
from the local November 8 repeated-study records. It makes no model calls.

Recall measures returned annotated pages, separately from pages merely read.
Costs use the pinned assumptions in `evaluation/pricing.py`, not invoices.
Source preparation and scoring are outside the retrieval timer. See the study
README for cohort selection, repeated-measurement aggregation and limitations.
