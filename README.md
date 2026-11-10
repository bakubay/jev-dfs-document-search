![A scout traverses a document graph toward evidence.](docs/assets/jev-dfs-header.png)

# Walking the table of contents with JEV

**Claim being tested:** a fast judgment model like JEV can treat a PDF's real
table of contents as a tree, walk it depth first, and find the evidence pages
for a question while reading only a small part of the document. The idea comes
from tree indexes such as PageIndex.

Two baselines isolate what each part contributes:

| Approach | What it is | What the comparison shows |
|---|---|---|
| **JEV DFS** | JEV ranks the ToC children; Python walks them depth first, checking page windows. Every sibling stays reachable. Stops when the selected pages pass a joint "is this enough?" check. | The method under test. |
| **Exhaustive JEV** | The same page checks over every page, keeping the best-scoring pages. No tree, no stop decision. | What the tree saves: does skipping most pages lose evidence? |
| **Sol agent** | An OpenAI model walks the same tree itself (descend / read / backtrack / finish). No JEV. | What JEV adds: is it a better driver for the walk than a general LLM? |

Same inputs and limits for all three: the same parsed pages and ToC, at most
**8 complete pages / 12,000 text tokens** per read and in the returned
evidence, page text never truncated. **Every approach sends one request at a
time**, so time per question follows the work done, not scheduling. No answers
are generated here.

JEV asks **one judgment per page**: does this page, read with the page before
and after as context, state evidence for part of the question? Only the judged
page can be returned; a neighbor is returned only on its own score. DFS stops
once JEV judges that nothing the question needs is still missing from the
selected pages (score ≥ **0.85**).

## Results

**[Read the report](docs/results/REPORT.md)**: approaches, one question step by
step, results and where each approach misses. Also: a
[30-second video](docs/assets/jev-three-ways.mp4) and the
[interactive report with every request and reply](docs/results/report.html).

20 annotated questions over 4 PDFs (three annual reports, one manual):

| Approach | Answer supported | Pages read | Requests | Cost / question | Time / question |
|---|---:|---:|---:|---:|---:|
| **JEV DFS** | 98% | 62 | 15.5 | **$0.0045** | **4.8 s** |
| Exhaustive JEV | **100%** | 116 | 16 | $0.0056 | 5.3 s |
| Sol agent¹ | **100%** | **10** | **3** | $0.056 | 12.5 s |

All three return pages that support the gold answer almost every time. The tree
walk gets there about 2.6× faster and 12× cheaper than an LLM agent on the same
tree, and reads about half the pages that checking every page does. "Answer
supported" also accepts other pages that give the same gold answer; the report
explains why annotated-page recall undercounts.

¹ Sol: 30 completed runs from a separate session; 2 runs that failed on network
or credit errors are left out. Details and limits are in the report.
How the settings were chosen: [`docs/experiments/`](docs/experiments/README.md).

## Code map

```text
retrieval/
  jev.py          JEV DFS + exhaustive JEV; ranking, descent, backtracking, page checks, stopping
  sol.py          Sol prompt, action loop, page reads and evidence selection
  tree.py         document types, ToC construction, child targets incl. uncovered gaps
  windows.py      complete-page batching shared by both providers
  common.py       evidence budgets, result type and coverage accounting
  config.py       model default and local .env loading
  run.py          retrieval timing and source-version recording helpers
  __main__.py     single-question command
preparation/
  parse.py        Datalab PDF parsing
  ground_toc.py   extract the real ToC and map it to physical PDF pages
evaluation/      the 20-question set, runner, repeats, recall/precision and cost
reporting/       offline HTML report builder
tests/           deterministic tests; no live model calls
docs/results/    current summary and interactive report
```

Start reading **`JevDFSRetriever._scan_leaf()` in [retrieval/jev.py](retrieval/jev.py)**:
check windows → pack useful evidence → assess sufficiency → continue if needed.
`JevDFSRetriever._search_children()` in the same file handles the surrounding
rank → descend → backtrack loop. For Sol, start at `SolRetriever.search()`
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

See [evaluation/README.md](evaluation/README.md) for running the 20-question
comparison and rebuilding the report.

```bash
python -m pytest -q
```

Tests cover branch recovery, uncovered pages, joint evidence across sections,
one-request-at-a-time windows, neighbor handling, page/token limits, failed requests, invalid agent actions,
returned-page recall and faithful trace reconstruction. No API keys are needed.

## What this experiment can and cannot establish

Timing runs from the question to returned evidence, including navigation and
stopping decisions. It excludes parsing, ground-truth scoring and answer writing.
Client-observed API time includes network, provider and SDK overhead.

DFS may spend extra calls navigating weak ToC labels. Exhaustive checks could be
parallelized in deployment; they are not here, so the comparison measures work
rather than scheduling. Model sufficiency is a judgment,
not a guarantee; thresholds are experimental. Exhausting all parsed text does
not prove an answer is absent from the original PDF. Parsing, grounding, missing
text, images and model mistakes remain limitations. The grounding heuristic
uses heading matches and printed/physical-page offsets; it is not universal.

This is a quick trial, not a claim of the best retrieval architecture. The study
uses a convenience sample and exact annotated-page recall, which can miss valid
alternative evidence. See the evidence review before drawing quality conclusions.

## Attribution

PageIndex is the principal inspiration. This work began from VectifyAI's
open-source JEV search example and tests direct traversal of parser output and
real ToC entries. It does not use generated section summaries or the PageIndex
package. The original [Apache 2.0 license](LICENSE) is preserved.
