# Evaluation dataset preparation

The first benchmark source is
[`VectifyAI/PageIndex-OSS-Benchmark`](https://github.com/VectifyAI/PageIndex-OSS-Benchmark):
62 text-lookup questions over 34 PDFs, with reference answers and physical
evidence pages. The source files are downloaded separately and are not vendored
here.

Prepare a local manifest:

```bash
git clone --depth 1 \
  https://github.com/VectifyAI/PageIndex-OSS-Benchmark.git \
  runs/upstream/PageIndex-OSS-Benchmark

python -m evaluation.pageindex_oss \
  runs/upstream/PageIndex-OSS-Benchmark \
  --output runs/evaluation/pageindex-oss-manifest.json
```

The screen uses `pdftotext` on the first and last 20 pages to find documents
that appear to expose a real table of contents. The initial cohort also requires
at least 70 pages; change that with `--min-pages`. The screen is deliberately
cheap and conservative. Selected PDFs must still pass the actual Datalab
extraction and physical-page grounding checks before an evaluation run.

The normalized question records retain the upstream answer and evidence pages.
After grounding, `evaluation.metrics.map_gold_pages` maps each evidence page to
the deepest containing ToC path. `score_trace` then reports gold-page recall,
gold-section recall, inspected leaf ranges, and time to the first gold range.

The benchmark subset contains answerable text lookups. A separate, manually
audited unanswerable set is still required to evaluate exhaustive negative
claims.

Run one grounded case with any of the three policies:

```bash
python -m evaluation.run_case \
  runs/evaluation/AMAZON_2017_10K/grounded \
  runs/evaluation/pageindex-oss-manifest.json \
  QUESTION_ID \
  --strategy jev_dfs \
  --output evaluation-results/QUESTION_ID.jev_dfs.json
```

Valid strategies are:

- `jev_dfs`: JEV orders branches; failed answer verification resumes DFS;
- `exhaustive`: inspect every leaf range in document order, then draft once;
- `llm_toc`: one Sol call selects ToC leaf ranges, followed by the same JEV
  page checks and answer gate used by the other methods.

The runner records gold-page/section recall, coverage, JEV requests, batched
JEV questions, writer/selector calls and tokens, and separate model latencies.
`answer_correct` remains null until the common semantic judge is run.

Per-query cost estimates pin the public prices observed on 2026-11-04:
[JEV at $0.042/M input tokens](https://help.typesafe.ai/articles/5187202169-how-much-does-jev-cost)
and standard short-context
[`gpt-6.1-sol` at $2/M input and $10/M output](https://platform.openai.com/pricing).
The Datalab parse charge is recorded separately as a one-time document cost.

After the common judge populates `answer_correct`, aggregate case files with:

```bash
python -m evaluation.summarize evaluation-results/*.json \
  --json-output evaluation-results/summary.json \
  --markdown-output evaluation-results/summary.md
```

