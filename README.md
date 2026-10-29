![A small scout exhaustively traverses a document graph, including a branch ranked 0.00, toward an evidence folder.](docs/assets/jev-dfs-header.png)

<p align="center"><em>Ranking sets the route. It does not erase the map.</em></p>

# JEV exhaustive ToC search experiment

This repository tests one narrow idea:

> Can a parser give us enough structure to search a PDF directly through its
> real table of contents and physical pages, while cheap, fast JEV judgments
> make an exhaustive depth-first fallback practical?

It is an experimental prototype, not a final retrieval architecture and not a
claim that this is the best way to search documents.

## Why this exists

[PageIndex](https://github.com/VectifyAI/PageIndex) is the major inspiration.
Its hierarchical view of a document makes long-document search easier to reason
about than one flat retrieval operation.

This experiment asks a different follow-up question: for a text-based PDF with
a useful table of contents, can we avoid generating a separate summary tree?
Instead, we try to:

1. parse the PDF with a document parser such as Datalab;
2. recover the document's own ToC;
3. ground printed ToC page numbers to physical PDF pages;
4. let JEV order the available branches at each level;
5. traverse those branches with depth-first search;
6. keep every sibling available, even when JEV gives it a low probability; and
7. in the worst case, check every parsed text page before reporting that no
   sufficient evidence was found.

PageIndex inspired the hierarchical traversal. This repository deliberately
tests direct traversal over parser output, real ToC entries, and real page text.
It does not use the PageIndex package or generated section summaries.

## Pipeline

```text
PDF
  ↓
Datalab paginated parse
  ↓
real ToC entries + physical page text
  ↓
printed-page → PDF-page grounding
  ↓
ToC tree with explicit uncovered page ranges
  ↓
JEV Choice orders immediate children
  ↓
ordered DFS visits branches (Choice never prunes)
  ↓
JEV screens and verifies pages with adjacent-page context
  ↓
useful evidence accumulates across branches
  ↓
Sol drafts a source-grounded answer
  ↓
exact citation validation + JEV completeness/support gate
  ├── accepted → return the answer
  └── incomplete → resume the saved DFS frontier
                         ↓
                frontier exhausted
                         ↓
        audit any parsed pages missed by the ToC
                         ↓
        report measured page coverage and limitations
```

Python owns traversal, coverage, citation checks, thresholds, and stopping.
JEV supplies typed semantic judgments (`Choice` and `Noul`). Sol writes the
candidate answer; it does not decide whether the search is complete.

## What “exhaustive” means here

In evaluated mode, JEV ranking changes only the order of traversal. A branch is
not deleted because its local probability is small or zero. If no answer passes
the stopping gate, DFS continues through the saved siblings and finally audits
parsed pages that were not reached through the ToC.

An `all_pages_checked` result means that the configured model checks completed
against every expected, non-truncated parsed text page. It does **not** prove
that the original PDF contains no answer. Parsing errors, missing text, tables,
images, incorrect ToC grounding, model mistakes, and context limits remain
possible. Empty, missing, or truncated pages prevent the implementation from
making the clean all-pages claim.

## Stopping rule

A candidate answer stops search only when all of these conditions hold:

- Sol reports the original question fully answered;
- at least one exact source citation validates against retained page text;
- JEV classifies the result as `direct` or `explicit_negative`;
- JEV completeness is at least `0.85`; and
- JEV source support is at least `0.85`.

The `0.85` values are conservative demo policies, not calibrated guarantees.
If the gate rejects a draft, its useful evidence and missing-information note
remain available while DFS resumes from the existing frontier.

## Repository layout

```text
datalab_parse.py          submit a PDF and preserve paginated parser output
datalab_ground_toc.py     extract/ground the real ToC and emit physical pages
datalab_dfs_search.py     build the ToC tree and run JEV-ordered DFS
evaluated_dfs.py          accumulate evidence, draft, judge, resume, and audit
openai_grounded_answer.py produce a structured answer with exact citations
tests/                    deterministic traversal, coverage, and citation tests
docs/traces/              standalone interactive recorded-run explainers
```

## Setup

Python 3.12 was used for the prototype.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in the ignored `.env` file:

```env
TYPESAFE_API_KEY=...
DATALAB_API_KEY=...
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-6.1-sol
```

## Run the experiment

### 1. Parse a text-based PDF

```bash
python datalab_parse.py document.pdf \
  --output runs/document \
  --mode balanced
```

The parser response is saved locally so later experiments do not need to parse
the same PDF again.

### 2. Extract and ground its ToC

```bash
python datalab_ground_toc.py \
  runs/document/document.json \
  --output runs/document/grounded
```

This produces:

- `pages.jsonl`: one record per physical PDF page;
- `toc_entries.json`: real ToC entries grounded to physical pages; and
- `grounding_report.json`: offset votes, verification counts, and failures.

The current grounding method combines heading similarity with a dominant
printed-page/PDF-page offset. It worked for the demo document, but it is one of
the experiment's most obvious areas for improvement.

### 3. Search, draft, evaluate, and resume

```bash
python datalab_dfs_search.py \
  runs/document/grounded \
  "What are the four functions in the AI RMF Core?" \
  --answer \
  --json-output runs/document/search-trace.json
```

The JSON output records branch rankings, page judgments, evidence, drafts,
answer evaluations, backtracking, coverage, and the final status.

## Recorded trace explainers

The two HTML files are standalone snapshots. They require no server or API keys;
open them directly in a browser.

- [Visual token and control flow](docs/traces/jev-token-flow-trace.html) follows
  the query, branch order, page checks, evidence, draft, rejection, and resumed
  DFS as data moves through the system.
- [Exact payload trace](docs/traces/jev-exact-payload-trace.html) exposes the
  recorded inputs, normalized outputs, decision policy, search memory, and
  coverage through a Langfuse-style observation tree.

The latter is a Langfuse-style projection from recorded application events, not
a Langfuse export. Both explainers describe one run against the 48-page NIST AI
RMF 1.0 PDF and label the limits of the recorded data.

## Tests

```bash
python -m pytest -q
```

The deterministic tests cover ToC-tree construction, ordered DFS backtracking,
retention of low/zero-probability siblings, exhaustive page verification,
evidence accumulation, answer-gate continuation, and exact citation validation.

## What this prototype does not settle

- whether DFS is the best latency/cost tradeoff for large documents;
- whether parser-derived ToCs work reliably across diverse PDF layouts;
- how to handle scanned PDFs, diagrams, complex tables, or missing text;
- how thresholds should be calibrated on a representative evaluation set;
- how to compress large accumulated evidence without damaging recall;
- whether a best-first, beam, hybrid, or learned policy should replace DFS; or
- whether the answer writer and evaluator should use different models.

Those are evaluation questions, not conclusions hidden by the demo.

## Attribution and license

This work began from VectifyAI's open-source
[`jev-doc-search`](https://github.com/VectifyAI/jev-doc-search) example and was
substantially refocused into the direct-parser, exhaustive-DFS experiment above.
PageIndex remains the central architectural inspiration. The original Apache 2.0
license is preserved in [LICENSE](LICENSE).
