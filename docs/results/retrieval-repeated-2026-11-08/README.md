# Repeated retrieval study

**Four PDFs × five questions × three approaches × three repetitions = 180 runs.**

Start with [HOW_TO_READ.md](HOW_TO_READ.md) and the standalone
[report and replays](../../traces/jev-retrieval-study.html).
The [results](summary.md) describe timing, calls, estimated cost and exact
annotated-page recall. Every run is accessible in the HTML, with its own raw
JSON download, prompts, decisions, source text and matching code snippets.

## Data

The PDFs are Amazon 2017 (85 pages), Best Buy 2023 (75), Activision Blizzard
2019 (198), and the WMF coffee manual (108). All four are in
[PageIndex-OSS-Benchmark](https://github.com/VectifyAI/PageIndex-OSS-Benchmark)
and were already parsed and grounded before this study.

The OSS subset contains only eight questions across these four PDFs. To reach
five questions per document, we retain those eight and add twelve **existing**
annotations from the pinned [MMLongBench-Doc-V2](https://github.com/VectifyAI/MMLongBench-Doc-V2)
source dataset. No questions, answers or evidence-page labels were generated.
This is an expanded cohort, not the original 62-question OSS benchmark.

[manifest.json](manifest.json) records exact questions, upstream row indices,
dataset revisions, page labels, source PDF hashes, parsed-text/ToC hashes,
selection policy, models and limits. Selection was frozen before collection.

All questions are answerable and their annotated pages are present in the parse.
The additional cases include tables, counting, and inputs to calculations.
Only retrieval of source evidence is evaluated. The coffee milk-temperature
question has an upstream model-produced visual flag, but this PDF's parsed
table explicitly contains the color words and temperatures; no image inference
is required by this input. No absent-evidence or image-only cases are included.

This convenience cohort consists of three financial reports and one manual;
it is not representative of all PDFs. The eight previously tested questions
are explicitly identified, not presented as unseen test cases.

## Fixed implementation

- JEV DFS: original grounded ToC, immediate-child ranking, ordered DFS,
  recoverable branches and an uncovered-page audit.
- Exhaustive JEV: document-wide combined window queue, up to eight concurrent
  workers, followed by bounded evidence selection and one joint-sufficiency check.
- Sol agent: its own descend/read/backtrack/finish actions over the same source;
  no JEV, answer writer or answer grading.
- Both JEV policies use the **four-judgment** combined call. The discussed
  two-judgment simplification remains a proposal.
- Complete windows and returned evidence are separately limited to eight pages
  and 12,000 text tokens. Neighbor pages count toward limits. No summaries,
  truncation or hierarchy repairs were introduced.

## Run or resume

The repository was consolidated after these runs. Their recorded source
snippets remain historical; the current implementation is mapped in the root
README. Use a new result directory with the cleaned source. Historical raw
records retain their original `source-snapshot/` for offline report rebuilding.

Use the repository's existing environment and `.env`. Raw run files are ignored
by Git and stored locally under `evaluation-results/`.

```bash
python -m evaluation.run_repeated \
  docs/results/retrieval-repeated-2026-11-08/manifest.json \
  --grounded-root runs/evaluation \
  --output-dir evaluation-results/new-repeated-study
```

Use a new directory for a fresh experiment. Add `--dry-run` to inspect the
balanced schedule without model calls. Add `--resume` only to an existing run
directory with unchanged inputs and source files. Saved failures and poor
retrievals are retained; resume never selectively retries them. Authentication
or quota errors stop the cohort explicitly.

Each repetition independently shuffles the sixty question/policy pairs using
a fixed seed. One query runs at a time, with fresh search state and clients.
JEV reuses a client within each run; Sol creates an OpenAI client per turn.
There is no application response cache; provider-side caching was not controlled.
This was not an isolated benchmark host: local report work ran during collection.

```bash
python -m reporting.build_cohort_report \
  --runs evaluation-results/retrieval-repeated-2026-11-08 \
  --output docs/traces/jev-retrieval-study.html \
  --summary docs/results/retrieval-repeated-2026-11-08/summary.md
```

The builder refuses missing runs by default and verifies source/input hashes.
`--allow-partial` is only for a preview labeled as incomplete. The report keeps
all payloads offline, stores repeated source text once, and losslessly restores
the selected run. No remote service is needed to view it in a modern browser.

## Interpret the measurements

Time ends at returned source evidence and includes all routing, reading,
selection and stopping decisions. Parsing, ToC extraction/grounding, gold
scoring and downstream answer generation are excluded. API timings include
network, provider and SDK work; no server-only or regional comparison is claimed.

The aggregate latency is the median of each question's median across its three
repetitions. Recall averages repetitions within each question, then questions.
Repetitions do not become extra independent test questions. Failures and
context/step-limit stops remain visible; error-free runs contribute latency
medians, while every saved attempt contributes to recall and failure counts.
Token pricing is estimated using the pre-existing pinned rates, not invoices.

Exact annotated-page recall is reproducible but does not recognize every valid
alternative page. [evidence-review.json](evidence-review.json) documents observed
duplicates, redundant annotations and possible genuine gaps using source
excerpts. This is assistant source inspection, not independent human adjudication;
it does not change the original scores or claim an adjusted accuracy result.

PageIndex and its JEV search work inspired the experiment. These are our
retrieval policies and measurements, not PageIndex's implementation or results.
