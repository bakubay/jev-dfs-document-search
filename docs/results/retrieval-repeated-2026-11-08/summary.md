# Repeated retrieval experiment: four PDFs, twenty questions

Completed **180/180** planned runs. Three repetitions per question and policy.

The four PDFs come from PageIndex-OSS-Benchmark. Eight questions are from its text-lookup subset; twelve additional existing annotations come from the pinned MMLongBench-Doc-V2 source dataset. These include table, enumeration and derived-answer questions. We score retrieval of the annotated pages, not answer calculation.

| Policy | Median of question-median time | Mean annotated-page recall | All annotated pages returned | Median calls per question | Mean estimated cost | API failures |
|---|---:|---:|---:|---:|---:|---:|
| JEV DFS | 2.52s | 88.3% | 48/60 | 8 | $0.00222 | 0 |
| Exhaustive JEV | 1.83s | 86.7% | 46/60 | 17 | $0.00681 | 0 |
| Sol agent | 9.37s | 85.0% | 45/60 | 3 | $0.05471 | 0 |

Time is query to returned full-page evidence, including routing and stopping decisions. Preprocessing, ground-truth scoring and downstream answer generation are excluded. Error-free retrievals contribute to latency medians; all recorded attempts contribute to recall and failure counts. Step/context-limit stops remain visible and are not silently removed. Cost is estimated from the existing pinned rates; failed-request usage can be unavailable.

## Per document

| Document | Policy | Median question time | Mean page recall | Full recall runs |
|---|---|---:|---:|---:|
| AMAZON_2017_10K.pdf | JEV DFS | 2.09s | 90.0% | 12/15 |
| AMAZON_2017_10K.pdf | Exhaustive JEV | 1.89s | 90.0% | 12/15 |
| AMAZON_2017_10K.pdf | Sol agent | 10.00s | 70.0% | 9/15 |
| BESTBUY_2023_10K.pdf | JEV DFS | 2.53s | 90.0% | 12/15 |
| BESTBUY_2023_10K.pdf | Exhaustive JEV | 1.72s | 76.7% | 10/15 |
| BESTBUY_2023_10K.pdf | Sol agent | 8.12s | 90.0% | 12/15 |
| ACTIVISIONBLIZZARD_2019_10K.pdf | JEV DFS | 5.31s | 73.3% | 9/15 |
| ACTIVISIONBLIZZARD_2019_10K.pdf | Exhaustive JEV | 2.56s | 80.0% | 9/15 |
| ACTIVISIONBLIZZARD_2019_10K.pdf | Sol agent | 25.04s | 90.0% | 12/15 |
| User_Manual_1500S_Classic_EN.pdf | JEV DFS | 1.63s | 100.0% | 15/15 |
| User_Manual_1500S_Classic_EN.pdf | Exhaustive JEV | 1.59s | 100.0% | 15/15 |
| User_Manual_1500S_Classic_EN.pdf | Sol agent | 7.77s | 90.0% | 12/15 |

## Scope and reproducibility

- Same prepared page text, grounded ToC and 8-page / 12k-token read and output limits. Neighbor pages count toward limits.
- JEV uses the frozen four-judgment combined-window implementation. The two-judgment proposal was not applied.
- Each repeat shuffles all question/policy pairs; queries run sequentially. Exhaustive has eight internal window workers; DFS retains leaf-local scheduling. Sol navigates alone, with no JEV or answer writer.
- Original ToC hierarchy is retained. This is a convenience cohort of three financial reports and one manual, not a representative sample of PDFs.
- All twenty questions are answerable. This cohort does not establish absent-evidence reliability, image understanding or correctness on arbitrary PDFs.
- Exact annotated-page recall can undercount valid evidence on alternative/repeated pages. Full recall is not answer accuracy. Repetitions are not additional independent questions.
- Client-observed time includes network/provider/SDK time. No region-controlled or server-only timing is available.
- This was not an isolated benchmark host: local report development/validation occurred during collection. Queries themselves ran one at a time.
- The report embeds every completed run with exact requests, parsed responses, source code and a downloadable JSON record. Missing attempts, if any, are explicitly listed.

Sources: [PageIndex-OSS-Benchmark](https://github.com/VectifyAI/PageIndex-OSS-Benchmark), [MMLongBench-Doc-V2](https://github.com/VectifyAI/MMLongBench-Doc-V2). Both revisions and input hashes are pinned in manifest.json.

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
