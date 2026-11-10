# Parameter exploration

How the shipped JEV DFS settings were reached. The main report
([`docs/results/REPORT.md`](../results/REPORT.md)) uses only the final setup.
Recall here is annotated-page recall. All runs: the same 20 questions; JEV `jev-1.13.0`; page score cutoff 0.35;
8 pages / 12,000 tokens per read and in the returned evidence.

| Run | What changed | DFS page recall | 1-page / 2-page | Pages read | Requests | Cost | Time | Read whole doc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | 8 requests in parallel; stop question *"is the evidence enough?"* ≥ 0.85 | 87.5% | 100% / 72% | 24 | 6 | $0.0019 | 2.4 s¹ | 3 of 35 |
| B | one request at a time; stop cutoff raised to 0.95 | 99.2% | 100% / 98% | 81 | 38 | $0.0059 | 21.1 s² | 32 of 60 |
| C | stop cutoff back to 0.85 | 87.5% | 100% / 72% | 24 | 7 | $0.0018 | 6.1 s² | 3 of 60 |
| D | stop question reworded to *"is anything the question needs missing?"*, cutoff 0.7 | 87.5% | 100% / 72% | 30 | 7 | $0.0024 | 2.4 s | 4 of 60 |
| **E (shipped)** | same question, cutoff 0.85 | **98.3%** | 100% / 96% | 62 | 15.5 | $0.0045 | 4.8 s | 20 of 60 |

Exhaustive JEV over the same runs: recall 95–100%, 116 pages read, about
$0.006 per question. From run D on it no longer asks a stop question.

¹ Requests within one retrieval ran in parallel, so this time is not comparable.
² Network latency was much higher during these sessions; compare pages read and cost.

## What we learned

- **One-page answers are never the problem.** Every setting found them 100% of
  the time. Every miss is a second input of a ratio or comparison.
- **The stop question decides two-page recall.** At 0.95, DFS rarely felt sure
  and read the whole document in half the runs, so it was barely cheaper than
  exhaustive. At 0.85 with the *"enough?"* wording, it stopped after the first
  input.
- **Asking what is missing works better than asking whether it is enough.**
  On 247 recorded stop decisions replayed against the annotations, the
  *missing* wording separated complete from incomplete evidence better (AUC 0.95
  vs 0.91). At 0.7, though, the early stops still scored 0.70–0.85, so recall
  only improved once the cutoff went back to 0.85.
- **The cutoff is a cost/recall dial.** 0.7 is about half the reading and cost
  for 72% two-page recall; 0.85 is 96% at about 80% of exhaustive's cost. Pass
  `sufficiency_threshold=0.7` to `JevDFSRetriever` for the cheaper setting.
- **Parallel requests were dropped** so all three approaches are timed the same
  way. `evaluation.run_repeated --parallel N` still runs several questions at once
  to shorten benchmark runs.

Because the cutoff was chosen on these same 20 questions, the shipped numbers
may be slightly optimistic.
