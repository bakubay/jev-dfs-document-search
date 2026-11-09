# Start with one question, not the scoreboard

Open the standalone `jev-retrieval-study.html` report and keep **Follow one search** selected.
Choose **Coffee manual → How many coffee quality levels can the coffee machine be set to? → JEV DFS → Run 1**.

1. **First choice:** JEV ranks “Barista pad — coffee strength” at 0.81. The lower-ranked branches stay available.
2. **First page check:** targets p30–31 are checked using source pages p29–32. The right panel shows the selected evidence.
3. **Why it continued:** the selected bundle receives 0.50, below the 0.85 stop threshold. The search backtracks to the saved root order.
4. Continue with Next. JEV enters Software, checks Operating options, and again gets an insufficient bundle (0.49).
5. It then checks Beverages, targets p35–40 with source p34–41. Page 39 lists quality levels 1 through 7. The selected bundle scores 0.94 and the search returns it.
6. **Returned pages:** the timer stops. Only then does the report compare returned pages with the held-out reference p39.

Switch to **Exhaustive JEV** for the same question. Its request bars overlap because up to eight windows are checked concurrently. It does not choose ToC branches.
Then switch to **Sol agent**: inspect its chosen target, received raw pages, and final selected page IDs. Sol supplies navigation actions; no JEV call is involved.

The three labels to distinguish are:

- **Received:** source text was delivered in a completed model call, possibly as neighboring context.
- **Checked:** JEV judged that page as a candidate. A neighboring page can be received without yet being a target.
- **Selected:** the page is in the current bounded evidence bundle. Selection can replace earlier pages, so receiving the reference page is not the same as returning it.

The application returns original page text, not a generated answer. Source text, exact prompts and responses, and responsible code are expandable at each event.

Use **Compare the results** only after following that case. The primary metric is exact annotated-page recall. Some documents repeat evidence or annotate more pages than the minimum required; the evidence-review notes explain observed examples without changing their original scores.
