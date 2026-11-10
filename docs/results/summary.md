# DFS vs exhaustive vs Sol: twenty questions over four PDFs

Completed **180/180** planned runs (up to three repetitions per question and approach).

Each approach gets a question and a prepared PDF and must return the source pages that answer it. Answer supported is the share of runs whose pages contain everything needed for the gold answer: every annotated page, or a reviewed alternative page set (below). Page recall is the share of annotated answer pages returned; precision is the share of returned pages that are annotated answer pages. Pages read counts every page whose text a model saw.

| Approach | Answer supported | Mean page recall | All annotated pages returned | Pages read | Median calls | Mean est. cost | Median time | Pages returned | Precision | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| JEV DFS | 98.3% | 98.3% | 58/60 | 62 | 15.5 | $0.00447 | 4.79s | 5.3 | 36% | 0 |
| Exhaustive JEV | 100.0% | 98.3% | 58/60 | 116 | 16 | $0.00558 | 5.32s | 7.3 | 20% | 0 |
| Sol agent | 100.0% | 87.5% | 46/60 | 10 | 3.5 | $0.05792 | 12.36s | 1.4 | 92% | 0 |

Questions are weighted equally; repetitions measure variation, not extra questions. Runs that failed on network or credit errors are excluded from the quality and time columns and counted under Failures. Every approach sends one request at a time, so time follows the work done. Time is query to returned pages, from error-free runs. Cost uses the pinned rates in evaluation/pricing.py.

Runs come from more than one run folder, each checked against its own frozen plan:

- `retrieval-v7-missing-085-2026-11-10` (retrieval-v7-missing-085): Exhaustive JEV, JEV DFS
- `sol-final-2026-11-10` (retrieval-v7-missing-085): Sol agent

## Per document

| Document | Approach | Median question time | Mean recall | Full recall runs |
|---|---|---:|---:|---:|
| AMAZON_2017_10K.pdf | JEV DFS | 2.86s | 96.7% | 14/15 |
| AMAZON_2017_10K.pdf | Exhaustive JEV | 5.04s | 100.0% | 15/15 |
| AMAZON_2017_10K.pdf | Sol agent | 12.92s | 83.3% | 11/15 |
| BESTBUY_2023_10K.pdf | JEV DFS | 2.57s | 100.0% | 15/15 |
| BESTBUY_2023_10K.pdf | Exhaustive JEV | 4.73s | 93.3% | 13/15 |
| BESTBUY_2023_10K.pdf | Sol agent | 9.97s | 90.0% | 12/15 |
| ACTIVISIONBLIZZARD_2019_10K.pdf | JEV DFS | 15.45s | 96.7% | 14/15 |
| ACTIVISIONBLIZZARD_2019_10K.pdf | Exhaustive JEV | 10.03s | 100.0% | 15/15 |
| ACTIVISIONBLIZZARD_2019_10K.pdf | Sol agent | 30.51s | 86.7% | 11/15 |
| User_Manual_1500S_Classic_EN.pdf | JEV DFS | 2.24s | 100.0% | 15/15 |
| User_Manual_1500S_Classic_EN.pdf | Exhaustive JEV | 5.43s | 100.0% | 15/15 |
| User_Manual_1500S_Classic_EN.pdf | Sol agent | 10.53s | 90.0% | 12/15 |

## Scope

- Same prepared page text, grounded ToC and 8-page / 12k-token read and output limits for every approach. JEV neighbor pages count toward the read limit.
- JEV asks one judgment per candidate page, read with its neighbors as context; only judged pages are returned. DFS checks one window at a time and stops once JEV judges that nothing the question needs is missing from the selected pages (score ≥ 0.85); exhaustive checks every page. Sol walks the same ToC alone, with no JEV. No approach runs requests in parallel.
- Recall counts annotated pages only. Annotations can miss valid alternative pages, and extra returned pages are not penalized by recall, so read it together with pages returned and precision.
- Four PDFs (three financial reports and one manual) and twenty answerable questions: a convenience sample, not a representative one. Latency is client-observed and includes network time.

## Alternative evidence

Source inspection found repeated or alternative evidence for several questions. Each alternative page set below was checked against the gold answer; these sets count toward Answer supported. Page recall still uses the original annotations. The review is by the assistant, not independent human adjudication.

- **AMAZON_2017_10K.pdf / Repeated source text:** The cost-of-sales components on annotated p44 are also stated on p27. Returning p27 can satisfy the evidence need while receiving zero exact annotated-page recall.
- **AMAZON_2017_10K.pdf / Annotated set is not minimal:** PDF p40 contains cash of 20,522 and liabilities of 57,883 + 24,743 + 20,975. Those inputs yield approximately 0.198; annotated p50 is not necessary for that calculation.
- **BESTBUY_2023_10K.pdf / Equivalent statement location:** The reference uses operating income on p25 and balance-sheet inputs on p39. Operating income 1,795 is repeated on p40; p39 supplies assets 15,803 and current liabilities 8,979. The returned pair p39+p40 supports 1,795 / (15,803 − 8,979) ≈ 0.263.
- **User_Manual_1500S_Classic_EN.pdf / Annotated set is not minimal:** The reference includes p24 and p25, but p25 explicitly identifies preparation using the manual insert and says max. 16 g. That page alone contains the requested fact.
- **ACTIVISIONBLIZZARD_2019_10K.pdf / Repeated financial inputs:** The reference uses product-development/R&D spending on p47 and total assets on p69. Spending 998 also appears on p70 and assets 19,845 also appear on p30. These listed alternative page pairs contain the inputs for approximately 5.03%.
- **BESTBUY_2023_10K.pdf / Equivalent inputs requiring arithmetic:** The annotated p39 states net goodwill 1,383. PDF p53 supplies total gross goodwill 2,058 and cumulative impairment 675, which yield 1,383. This is useful alternative evidence, although it requires arithmetic rather than a direct lookup.
- **ACTIVISIONBLIZZARD_2019_10K.pdf / Repeated financial inputs:** Gold answer 17.85 = operating income 1,607 / interest expense 90. Operating income is on p41 and p70; the interest expense of 90 is on p114 and, as 86 interest plus 4 amortization, on p98. The net interest-and-other line (-26) on p37/p70 is a different number and does not supply the denominator.

Open the standalone HTML's source-review details to inspect excerpts and the original run payloads.
