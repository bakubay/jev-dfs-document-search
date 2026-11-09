![A scout traverses a document graph toward evidence.](docs/assets/jev-dfs-header.png)

# ToC evidence retrieval with JEV

A small experiment inspired by [PageIndex](https://github.com/VectifyAI/PageIndex)
and its [JEV document search](https://github.com/VectifyAI/jev-doc-search).
Can we reuse a PDF's real table of contents and parsed pages, then use fast
decision models to find evidence without generating a separate summary tree?

This repository contains **three retrieval methods**. Each receives a question
and the same prepared document, then returns original source pages. Answer
generation is downstream and is not implemented here.

| Method | Navigation and reading | When it stops |
|---|---|---|
| JEV DFS | JEV ranks immediate ToC children; Python searches them depth first and checks bounded page windows. Every sibling stays available. | Selected evidence passes a joint sufficiency check, or all remaining nonempty pages have been checked. |
| Exhaustive JEV | The same page checks run across the whole document with up to eight concurrent workers. No ToC ranking. | All windows finish; code selects and assesses evidence once. |
| Sol agent | Sol chooses descend, read, backtrack, finish or stop actions. Python supplies raw pages. No JEV calls. | Sol finishes, abstains, or reaches its step/context budget. |

Both JEV methods ask four independent judgments per target page in **one call
per window**. Evidence selection is followed by a separate joint check because
its input depends on the completed page judgments. Read windows and returned
evidence are each capped at **8 complete pages / 12,000 text tokens**; neighbors
count toward the limits. Page text is never truncated to fit.

## Start here

- [One-question timeline](docs/traces/jev-simple-execution-trace.html): actual requests, replies, code actions and timing for the coffee example.
- [Study and all replays](docs/traces/jev-retrieval-study.html): four PDFs, twenty annotated questions, three methods, three repeats.
- [Results and limitations](docs/results/retrieval-repeated-2026-11-08/summary.md).

The HTML files work offline. They are recorded snapshots, not live searches.
Their source snippets match the code at recording time, before this repository
was consolidated. The file map below describes the current implementation.

## Code map

```text
retrieval/
  jev.py          JEV DFS + exhaustive JEV; page questions, evidence selection, stopping
  sol.py          Sol prompt, action loop, page reads and evidence selection
  tree.py         document types, ToC construction, ranking, descent and backtracking
  windows.py      complete-page batching shared by both providers
  common.py       evidence budgets, result type and coverage accounting
  config.py       model default and local .env loading
  run.py          retrieval timing and source-version recording helpers
  __main__.py     single-question command
preparation/
  parse.py        Datalab PDF parsing
  ground_toc.py   extract the real ToC and map it to physical PDF pages
evaluation/      dataset preparation, three-method runner, repeats, recall and cost
reporting/       builders for the two retained HTML views
tests/           deterministic tests; no live model calls
docs/            latest study, frozen question manifest and two HTML snapshots
```

Start reading **`JevDFSRetriever._scan_leaf()` in [retrieval/jev.py](retrieval/jev.py)**:
check windows → pack useful evidence → assess sufficiency → continue if needed.
`OrderedDFSSearcher._search_node()` in [retrieval/tree.py](retrieval/tree.py)
handles the surrounding branch traversal. For Sol, start at `SolRetriever.search()`
in [retrieval/sol.py](retrieval/sol.py).

## Setup and one search

Python 3.12 is used for the prototype.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in the ignored `.env`: JEV needs `TYPESAFE_API_KEY`; Sol needs
`OPENAI_API_KEY` and optionally `OPENAI_MODEL`. Parsing needs `DATALAB_API_KEY`.
The two JEV methods do not need an OpenAI key.

```bash
python -m preparation.parse document.pdf --output runs/document --mode balanced
python -m preparation.ground_toc runs/document/document.json --output runs/document/grounded
python -m retrieval runs/document/grounded "What evidence supports this claim?" \
  --method jev_dfs --output runs/document/dfs.json
```

Use `--method exhaustive` or `--method sol_only` for the other methods, with a
new output filename. The result includes full evidence pages, coverage, model
requests/replies, usage and elapsed time. Prepared inputs are `pages.jsonl` and
`toc_entries.json`; raw PDFs, parser outputs, keys and live run logs stay ignored.

## Compare the methods

See [evaluation/README.md](evaluation/README.md) for dataset preparation, paired
runs, repeated experiments and offline report rebuilding.

```bash
python -m pytest -q
```

Tests cover branch recovery, uncovered pages, joint evidence across sections,
parallel windows, page/token limits, failed requests, invalid agent actions,
returned-page recall and faithful trace reconstruction. No API keys are needed.

## What this experiment can and cannot establish

Timing runs from the question to returned evidence, including navigation and
stopping decisions. It excludes parsing, ground-truth scoring and answer writing.
Client-observed API time includes network, provider and SDK overhead.

DFS may spend extra calls navigating weak ToC labels. Parallel exhaustive checks
can finish sooner despite reading more pages. Model sufficiency is a judgment,
not a guarantee; thresholds are experimental. Exhausting all parsed text does
not prove an answer is absent from the original PDF. Parsing, grounding, missing
text, images and model mistakes remain limitations. The grounding heuristic
uses heading matches and printed/physical-page offsets; it is not universal.

This is a quick trial, not a claim of the best retrieval architecture. The study
uses a convenience cohort and exact annotated-page recall, which can miss valid
alternative evidence. See the retained study's evidence review before drawing
quality conclusions.

## Attribution

PageIndex is the principal inspiration. This work began from VectifyAI's
open-source JEV search example and tests direct traversal of parser output and
real ToC entries. It does not use generated section summaries or the PageIndex
package. The original [Apache 2.0 license](LICENSE) is preserved.
