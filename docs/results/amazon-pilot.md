# Amazon 2017 10-K pilot comparison

Run date: 2026-11-04

This is an engineering smoke test, not a publishable benchmark result. It uses
three answerable PageIndex-OSS-Benchmark questions over the same 85-page Amazon
2017 10-K. All three gold pages fall inside the same nested ToC path:

`PART II → Item 8 → Notes to Consolidated Financial Statements`

The three methods share the same Datalab parse, grounded ToC, JEV page screen
and verification questions, Sol answer writer, exact-citation validation, and
JEV answer gate. Only the section-traversal policy changes.

| strategy | cases | failures | answer accuracy | section recall | pages screened | pages verified | JEV requests | Sol calls | latency (s) | query cost ($) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Exhaustive | 3 | 0 | 1.000 | 1.000 | 85.0 | 85.0 | 171.0 | 1.0 | 26.02 | 0.065692 |
| JEV DFS | 3 | 0 | 1.000 | 1.000 | 31.7 | 8.7 | 45.0 | 1.7 | 17.51 | 0.033452 |
| Sol ToC selection | 3 | 0 | 1.000 | 1.000 | 31.0 | 8.0 | 40.0 | 1.0 | 10.47 | 0.026636 |

Answer accuracy was judged with MMLongBench-Doc-V2's semantic-equivalence
judge using `gpt-5.6-luna`; all nine answers were equivalent to their reference
answers. Query cost excludes the one-time Datalab parse, which cost $0.34 for
this document. Estimates use the public prices observed on 2026-11-04.

## What this pilot shows

- The comparison harness works end to end.
- JEV DFS correctly backtracked from a plausible sibling on two questions and
  then reached the gold notes section.
- Exhaustive evaluation provides the coverage ceiling but used about 3.8× the
  JEV requests of DFS in this pilot.
- One-shot Sol ToC selection was fastest and cheapest on these easy questions.
- “Pages screened” and “pages verified” are reported separately. A page-level
  JEV screen still reads page text even when the page never reaches the more
  expensive adjacent-context verification stage.

## What this pilot does not show

All three questions are answerable, concern conventional text, and land in the
same ToC branch. The sample cannot establish a general recall advantage or the
value of exhaustive fallback. The next cohort needs different document types,
answers distributed across branches, misleading ToC choices, and manually
audited unanswerable questions.

