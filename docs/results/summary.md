# DFS vs exhaustive vs Sol: twenty questions over four PDFs

Completed **152/180** planned runs (up to three repetitions per question and approach).

Each approach gets a question and a prepared PDF and must return the source pages that answer it. Recall is the share of annotated answer pages returned; precision is the share of returned pages that are annotated answer pages. Pages read counts every page whose text a model saw.

| Approach | Mean recall | All answer pages returned | Pages read | Median calls | Mean est. cost | Median time | Pages returned | Precision | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| JEV DFS | 98.3% | 58/60 | 62 | 15.5 | $0.00447 | 4.79s | 5.3 | 36% | 0 |
| Exhaustive JEV | 98.3% | 58/60 | 116 | 16 | $0.00558 | 5.32s | 7.3 | 20% | 0 |
| Sol agent | 82.5% | 22/32 | 10 | 3 | $0.05614 | 12.54s | 1.4 | 88% | 2 |

Questions are weighted equally; repetitions measure variation, not extra questions. Runs that failed on network or credit errors are excluded from the quality and time columns and counted under Failures. Every approach sends one request at a time, so time follows the work done. Time is query to returned pages, from error-free runs. Cost uses the pinned rates in evaluation/pricing.py.

Runs come from more than one run folder, each checked against its own frozen plan:

- `retrieval-v7-missing-085-2026-11-10` (retrieval-v7-missing-085): Exhaustive JEV, JEV DFS
- `retrieval-v3-one-judgment-2026-11-09` (retrieval-v3-one-judgment): Sol agent

## Per document

| Document | Approach | Median question time | Mean recall | Full recall runs |
|---|---|---:|---:|---:|
| AMAZON_2017_10K.pdf | JEV DFS | 2.86s | 96.7% | 14/15 |
| AMAZON_2017_10K.pdf | Exhaustive JEV | 5.04s | 100.0% | 15/15 |
| AMAZON_2017_10K.pdf | Sol agent | 14.51s | 70.0% | 5/8 |
| BESTBUY_2023_10K.pdf | JEV DFS | 2.57s | 100.0% | 15/15 |
| BESTBUY_2023_10K.pdf | Exhaustive JEV | 4.73s | 93.3% | 13/15 |
| BESTBUY_2023_10K.pdf | Sol agent | 10.44s | 90.0% | 6/9 |
| ACTIVISIONBLIZZARD_2019_10K.pdf | JEV DFS | 15.45s | 96.7% | 14/15 |
| ACTIVISIONBLIZZARD_2019_10K.pdf | Exhaustive JEV | 10.03s | 100.0% | 15/15 |
| ACTIVISIONBLIZZARD_2019_10K.pdf | Sol agent | 37.12s | 80.0% | 5/8 |
| User_Manual_1500S_Classic_EN.pdf | JEV DFS | 2.24s | 100.0% | 15/15 |
| User_Manual_1500S_Classic_EN.pdf | Exhaustive JEV | 5.43s | 100.0% | 15/15 |
| User_Manual_1500S_Classic_EN.pdf | Sol agent | 9.85s | 90.0% | 6/7 |

## Scope

- Same prepared page text, grounded ToC and 8-page / 12k-token read and output limits for every approach. JEV neighbor pages count toward the read limit.
- JEV asks one judgment per candidate page, read with its neighbors as context; only judged pages are returned. DFS checks one window at a time and stops once JEV judges that nothing the question needs is missing from the selected pages (score ≥ 0.85); exhaustive checks every page. Sol walks the same ToC alone, with no JEV. No approach runs requests in parallel.
- Recall counts annotated pages only. Annotations can miss valid alternative pages, and extra returned pages are not penalized by recall, so read it together with pages returned and precision.
- Four PDFs (three financial reports and one manual) and twenty answerable questions: a convenience sample, not a representative one. Latency is client-observed and includes network time.

## Review annotation misses before drawing quality conclusions

The original scores above are unchanged. Source inspection found repeated or alternative evidence for several questions. This review is by the assistant, not independent human adjudication; it does not provide an adjusted accuracy score.

- **AMAZON_2017_10K.pdf / Repeated source text:** The cost-of-sales components on annotated p44 are also stated on p27. Returning p27 can satisfy the evidence need while receiving zero exact annotated-page recall.
- **AMAZON_2017_10K.pdf / Annotated set is not minimal:** PDF p40 contains cash of 20,522 and liabilities of 57,883 + 24,743 + 20,975. Those inputs yield approximately 0.198; annotated p50 is not necessary for that calculation.
- **BESTBUY_2023_10K.pdf / Equivalent statement location:** The reference uses operating income on p25 and balance-sheet inputs on p39. Operating income 1,795 is repeated on p40; p39 supplies assets 15,803 and current liabilities 8,979. The returned pair p39+p40 supports 1,795 / (15,803 − 8,979) ≈ 0.263.
- **User_Manual_1500S_Classic_EN.pdf / Annotated set is not minimal:** The reference includes p24 and p25, but p25 explicitly identifies preparation using the manual insert and says max. 16 g. That page alone contains the requested fact.
- **ACTIVISIONBLIZZARD_2019_10K.pdf / Repeated financial inputs:** The reference uses product-development/R&D spending on p47 and total assets on p69. Spending 998 also appears on p70 and assets 19,845 also appear on p30. These listed alternative page pairs contain the inputs for approximately 5.03%.
- **BESTBUY_2023_10K.pdf / Equivalent inputs requiring arithmetic:** The annotated p39 states net goodwill 1,383. PDF p53 supplies total gross goodwill 2,058 and cumulative impairment 675, which yield 1,383. This is useful alternative evidence, although it requires arithmetic rather than a direct lookup.
- **ACTIVISIONBLIZZARD_2019_10K.pdf / Possible genuine evidence gap:** Some returned bundles include operating income but omit p114, where the separate interest expense of 90 is stated. The net interest-and-other-income line is a different number. A high sufficiency score alone does not establish that the denominator was retrieved; these runs need closer review.

Open the standalone HTML's source-review details to inspect excerpts and the original run payloads.
