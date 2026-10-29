<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/header-dark.png">
  <img src="assets/header-light.png" width="900" alt="Long-document search with Jev and PageIndex">
</picture>


**Find the page that answers a question in a 300-page report with Jev's `Choice`. No vector DB or embeddings!**

[Jev](https://docs.typesafe.ai) answers multiple-choice questions: give it the options, and it returns a probability for each. "Which page answers this question?" is one of them, and it works well, until the document outgrows what Jev can read at once. [PageIndex](https://github.com/VectifyAI/PageIndex) removes that limit by turning the document into a hierarchical tree representation. Jev picks a node, then one of its children, and so on down the tree, choosing among a handful of options each time, however long the document.

## Page search with Jev's `Choice`

Page search can be framed as a multiple-choice question. The question is "which page answers this?", and the options are the pages themselves: each page is one option, described by its own text. Jev reads every option and returns a probability for each page; the most likely page is the answer. We call this flat page search: one [`Choice`](https://docs.typesafe.ai/primitives/choice), one option per page.

<img src="assets/page-search.gif" width="900" alt="Each page is one option of a Choice; Jev returns a probability for every page">

To try it on a short PDF of your own, such as an employee handbook, run [page_search.py](page_search.py) (see [Setup](#setup) for installing and API keys):

```bash
python page_search.py handbook.pdf "How many vacation days do new employees get?"
```

It prints the most likely page and its text. The core of it is a single call:

```python
from pypdf import PdfReader
from typesafe_sdk import Choice, TypeSafeClient

typesafe = TypeSafeClient(model="jev-1.13.0")

pages = [page.extract_text() for page in PdfReader("handbook.pdf").pages]
question = "How many vacation days do new employees get?"

r = typesafe.system_one(
    state={"question": question},
    questions={"page": Choice(
        instructions="Which page contains the answer to the question?",
        criteria={f"p{i}": text for i, text in enumerate(pages, 1)},
    )},
)
print(r.answers["page"].choice)  # the most likely page, e.g. 'p7'
```

In the code:

- **`state`** is what Jev reads before deciding. Here it is just the question.
- **`Choice`** is the decision. Its options go in `criteria`: one key per page (`p1`, `p2`, …), each described by that page's text.
- **The answer** names the most likely page in `choice`, with every page's probability in `probabilities`, summing to 1.

### Why flat page search breaks on long documents

Flat page search hits two limits, one after the other:

1. **Too many tokens.** Every page's text goes into the request as an option, and `state` plus the `Choice` must fit in 32k tokens. That runs out after a few dozen pages. NVIDIA's [10-K for fiscal 2026](https://d18rn0p25nwr6d.cloudfront.net/CIK-0001045810/e361e58a-7483-44f5-bc62-a9080ae6ec72.pdf) has only 93 pages, but its text is about 76k tokens, more than twice what fits.
2. **Too many options.** A `Choice` takes at most 255 options, so one option per page stops at 255 pages. Some annual reports are longer than that: Citigroup's [10-K for 2025](https://www.citigroup.com/rcs/citigpa/storage/public/citi-2025-10-k-2-20-26.pdf) has 318 pages.

[PageIndex](https://github.com/VectifyAI/PageIndex) solves both at once. It turns the flat choice over pages into a tree: the document splits into sections, each with a title and a short summary, and each section into its pages. Jev then makes one small choice per level instead of one huge one, so every `Choice` has only a handful of options and fewer tokens.

## Tree search with PageIndex

### 1. Build the tree with PageIndex

PageIndex builds the tree from the document's own structure. The root is the whole document, its children are sections, and each section has a title, a summary, and a page range, down to the pages.

<img src="assets/tree-index.gif" width="990" alt="PageIndex turns a 300-page document into a three-level tree: the document, its sections, their pages">

Submit the PDF and read its tree:

```python
import os

from pageindex import PageIndexClient

pageindex = PageIndexClient(api_key=os.environ["PAGEINDEX_API_KEY"])

doc_id = pageindex.submit_document("NVIDIA_10K.pdf", wait=True)["doc_id"]
tree = pageindex.get_document_structure(doc_id)
```

`tree` is a list of nodes, each with its `title`, its page range (`start_index` to `end_index`), a `summary` of those pages, and its children in `nodes`. For the NVIDIA 10-K it looks like this (abridged):

```jsonc
[
  // … 4 more
  {
    "title": "Item 1. Business",
    "node_id": "0004",
    "start_index": 4,
    "end_index": 12,
    "summary": "This section covers NVIDIA's overall business overview…",
    "nodes": [
      {
        "title": "Our Company",
        "node_id": "0005",
        "start_index": 4,
        "end_index": 5,
        "summary": "This text provides an overview of NVIDIA as a pioneer in…"
      }
      // … 16 more
    ]
  }
  // … 25 more
]
```

### 2. Search the tree with Jev

Jev starts at the root and asks one `Choice` per level: which of these children most likely contains the answer? Each option reads `"title. summary"`. The chosen section's children become the next menu, until the search reaches pages.

<img src="assets/tree-search.gif" width="990" alt="Jev searches the tree with one Choice per level: a section, then a page">

Continuing from the tree above, the search takes two steps.

**First, pick a section.** One `Choice` over the top-level sections, each described by its title and summary.

```python
from pageindex.utils import get_node
from typesafe_sdk import Choice, TypeSafeClient

typesafe = TypeSafeClient(model="jev-1.13.0")
question = "What was NVIDIA's total revenue for fiscal year 2026?"

sections = {n["node_id"]: f"{n['title']}. {n.get('summary', '')}" for n in tree}

r = typesafe.system_one(
    state={"question": question},
    questions={"section": Choice(
        instructions="Which section most likely contains the answer to the question?",
        criteria=sections,
    )},
)
section = get_node(tree, r.answers["section"].choice)  # the picked node_id, e.g. "0004"
```

**Then, pick a page inside it.** One `Choice` over the pages of that section, `start_index` to `end_index`, as in flat page search.

```python
start_index, end_index = section["start_index"], section["end_index"]
pages = pageindex.get_page_content(doc_id, f"{start_index}-{end_index}")
r = typesafe.system_one(
    state={"question": question},
    questions={"page": Choice(
        instructions="Which page contains the answer to the question?",
        criteria={f"p{p['page_index']}": p["markdown"] for p in pages},
    )},
)
print(r.answers["page"].choice)  # the most likely page
```

This is a *simplified version* to show the idea; the next section covers the full tree search.

## The full tree search

The two-step search above is the simplest version. Going further, three changes make it general and sturdier: *deeper trees*, *top-K search*, and *checking with `Noul`*.

**Deeper trees.** A real tree has more than two levels: sections have subsections, which can have their own. The search is the same step repeated: one `Choice` over the children of the section just picked, until a section has no subsections, then one over its pages. Each level keeps the menu short, however long the document. The tree can go one level further, below pages: once a page is picked, one more `Choice` over its lines finds the exact line, as in TypeSafe's [line-by-line search](https://docs.typesafe.ai/cookbooks/semantic_find) cookbook.

**Top-K search.** Taking only the most likely option at each step is brittle: if the right section or page ranks second, it is lost. Instead, keep the top K: the K most likely keys of `probabilities`, not just `choice`. In the tree this is beam search: keep the K best paths at each level, scored by the geometric mean of their step probabilities, as in TypeSafe's [hierarchical classification](https://docs.typesafe.ai/cookbooks/hierarchical_classification) cookbook.

**Checking with `Noul`.** A `Choice` is relative: its probabilities sum to 1, so it always names a winner, even when no option answers the question. A [`Noul`](https://docs.typesafe.ai/primitives/noul) is a yes/no question with its own probability, so each candidate page can be judged on its own, with its full text in `state`, and kept or dropped by a threshold.

[tree_search.py](tree_search.py) puts all three together:

1. **Sections.** A beam of 3 goes down a tree of any depth. At each section, its opening pages, before its first subsection, are an option too.
2. **Pages.** One `Choice` over the pages of each of the 3 sections the beam ends in picks the candidates. A section too long for one request is split into windows that fit.
3. **Check.** One `Noul` per candidate, up to 16. Pages at 0.5 or above are kept, or the best 2 if none is.

Run it on a PDF and a question:

```bash
python tree_search.py NVIDIA_10K.pdf "What was NVIDIA's total revenue for fiscal year 2026?"
```

It uploads the PDF, builds the tree, prints its `doc_id`, then the sections the search ended in and the pages it kept. To ask another question about the same document, pass the `doc_id` instead of the PDF, so it is not uploaded again:

```bash
python tree_search.py pi-... "What was NVIDIA's gross margin for fiscal year 2026?"
```

On the two annual reports, uploaded to the cloud PageIndex:

| Question | Pages | Answer found on | Correct |
| --- | --- | --- | --- |
| What was NVIDIA's total revenue for fiscal year 2026? | 93 | p37, p51 | ✓ |
| What was Citigroup's net income for 2025? | 318 | p12, p16, p134, p135 | ✓ |

Both answers are right: they are the figures in each report's consolidated statement of income (NVIDIA p51, Citigroup p134).

## Setup

```bash
pip install -r requirements.txt
export TYPESAFE_API_KEY="..."
export PAGEINDEX_API_KEY="..."
```

You can get a TypeSafe key from the [TypeSafe console](https://console.typesafe.ai) and a PageIndex key from the [PageIndex dashboard](https://dash.pageindex.ai/).

## Parse and ground a PDF with Datalab

Put `DATALAB_API_KEY=...` in the ignored `.env` file. Parse a document into
paginated Markdown and structured JSON:

```bash
python datalab_parse.py document.pdf --output run/document --mode balanced
```

Then turn the structured result into one record per physical page and ground the
document's ToC entries to those records:

```bash
python datalab_ground_toc.py run/document/document.json --output run/document/grounded
```

The grounded output contains:

- `pages.jsonl`: physical `page_id`, one-based PDF page number, bounding box, and cleaned Markdown.
- `toc_entries.json`: printed ToC page, grounded PDF page, matched heading, block ID, and bounding box.
- `grounding_report.json`: inferred offset, offset votes, verification counts, and failed pages.

Search the grounded document with the summary-free ordered DFS:

```bash
python datalab_dfs_search.py run/document/grounded \
  "What are the four functions in the AI RMF Core?" \
  --answer \
  --json-output run/document/search-trace.json
```

At each ToC node, JEV `Choice` orders only the immediate children. The code
preserves every sibling and searches them in that local order. Every page in a
leaf range is screened with independent `Noul` questions, and the best candidates
are checked with their adjacent pages. A branch is abandoned only when its page
evidence fails verification; a high or low `Choice` probability never removes it.

With `--answer`, `EvaluatedDFSSearcher` checks every nonempty page in each visited
leaf (the legacy top-K candidate limit applies only without `--answer`). It caches
page checks, retains useful source pages across sections, and calls Sol to draft
when the evidence bundle grows. Exact citation validation and a JEV answer gate
run before stopping. JEV sees the original question, draft, and source pages; it
classifies the answer and independently checks completeness and source support.
Both Nouls must reach 0.85, the writer must report fully answered, and the verdict
must be direct or explicitly negative. This is a demo threshold, not a guarantee.

A rejected draft resumes the same recursive DFS stack. Missing-information text
comes from the writer's limitations field; JEV returns typed judgments, not prose.
Every unique page has separate screening and verification coverage. Empty,
missing, and truncated text never counts as fully checked. After exhausting DFS,
any text pages missed by ToC traversal are checked before reporting exhaustion.
All-page coverage means model checks over parsed text, not proof of absence.

Accumulated evidence is retained verbatim. If it exceeds the JEV context budget,
the run stops as incomplete rather than silently discarding evidence. Configure
the ignored `.env`:

```env
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-6.1-sol
```

## Recorded trace explainers

Two standalone HTML snapshots document the evaluated DFS pipeline without
requiring API keys or a running server:

- [`docs/traces/jev-token-flow-trace.html`](docs/traces/jev-token-flow-trace.html)
  follows the query, branch ordering, page checks, retained evidence, drafting,
  evaluation, and DFS continuation as a visual flow.
- [`docs/traces/jev-exact-payload-trace.html`](docs/traces/jev-exact-payload-trace.html)
  exposes the recorded request payloads and typed responses at each hop, with a
  Langfuse-style observation tree and expandable exact values.

These are recorded demo traces, not live Langfuse exports. Open either file
directly in a browser; its data, styling, and interactions are embedded.

## Watch the search as a chat

The standalone UI replays the historical golden run when opened directly. To ask new
questions and stream the real JEV hops into the interface, start the local server:

```bash
python jev_chat_server.py
```

If dependencies are cached through uv, use
`uv run --offline --with-requirements requirements.txt python jev_chat_server.py`.

Then open `http://127.0.0.1:8765`. Every `Choice`, descent, page screen,
verification, and backtrack is recorded as structured JSON. Use **Export** in the
UI to download the run. `Choice` rows are explicitly local rankings; all siblings
remain visible and recoverable until an answer passes evaluation. The memory
panel shows unique page coverage, retained evidence, the latest verdict and
missing information. "Replay fee test" loads the recorded live licensing
regression without another model call.

## License

[Apache 2.0](LICENSE)
