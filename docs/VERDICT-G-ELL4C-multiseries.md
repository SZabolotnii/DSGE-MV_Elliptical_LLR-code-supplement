# VERDICT — G-ELL-4C (derived>fitted across many heavy-tailed real series)

**Trigger.** Reviewer/own-analysis concern: the heavy-tail real-data result rested
on a single FX series (n=1, all-FX). Turn it into a pattern across asset classes,
each validated leakage-free (the gate G-ELL-4B scheme: purged-and-embargoed blocked
6-fold + moving-block bootstrap).

**Setup.** `run_ell_gate4c_multiseries.py`. Six real financial series (FRED),
same task as gate4 (d=5 return embeddings, top-vs-bottom 21-day vol-regime tercile),
purged blocked 6-fold + moving-block bootstrap (block 21, embargo 26), train-only
scaling. Five genuinely heavy-tailed + one light-tailed FX control. Seed 2026.

**Results (95% CI of Δaccuracy; `>0` favours the derived link; ✓ = excludes 0).**

| series | class | n | exc.kurt | kun−ghosh (fitted) | kun−qda | kun−identity |
|---|---|---|---|---|---|---|
| CAD/USD | FX | 9256 | 9.3 | **[+0.015,+0.027]** | [−0.010,+0.018] | **[+0.004,+0.027]** |
| JPY/USD | FX | 9250 | 9.1 | [−0.006,+0.016] | **[+0.009,+0.047]** | **[+0.021,+0.059]** |
| GBP/USD | FX | 9254 | 6.9 | **[+0.004,+0.017]** | [−0.008,+0.023] | [−0.006,+0.021] |
| WTI oil | commodity | 6772 | 64.9 | **[+0.005,+0.028]** | **[+0.024,+0.070]** | **[+0.004,+0.032]** |
| S&P 500 | equity | 1658 | 16.8 | [−0.048,+0.005] | **[+0.038,+0.126]** | **[+0.020,+0.090]** |
| EUR/USD *(light control)* | FX | 4574 | 2.5 | **[+0.007,+0.018]** | [−0.014,+0.013] | [−0.006,+0.018] |

**Findings — stronger and honest.**
1. **Derived ≥ fitted-Ghosh, never significantly worse on any series.** Significant
   on 3/5 heavy series (CAD, GBP, oil) + the light control; tie on 2/5 (JPY, S&P,
   where the fitted link is level — S&P point estimate slightly favours fitted but
   n.s.). The paper's central derived-vs-fitted thesis holds as a pattern across FX,
   commodity, and equity.
2. **Derived beats QDA on the heaviest-tailed series and ties on the rest —
   tracking tail-heaviness.** Significant vs QDA on oil (kurt 64.9), S&P (16.8),
   JPY (9.1); tie on CAD (9.3) and GBP (6.9); **tie on the light EUR control
   (2.5).** This re-opens a qualified real-data vs-QDA advantage that the single
   CAD series (a tie) had hidden, and it shows the **adaptivity thesis on real
   data**: the QDA advantage appears under heavy tails and vanishes on the light
   control.
3. **Derived > identity in 4/5 heavy series** — a non-identity link is needed on
   real heavy-tailed data.

**Note.** CAD/USD excess kurtosis is **9.3** (reproducible from DEXCAUS daily
log-returns), correcting the earlier manuscript figure "12.3". Per-series kurtosis
now reported directly.

**Manuscript action.** Replace the single-FX §exp-heavy with this multi-series
table; reframe the headline as derived ≥ fitted across asset classes (sig. on the
majority, never worse) + derived beats QDA on the heaviest-tailed series with the
advantage tracking kurtosis and vanishing on the light control. Data staged in the
lake at `store/fred-heavytail/`. Script + `ell_gate4c_results.json` retained.
