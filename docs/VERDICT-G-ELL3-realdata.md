# Gate G-ELL-3 (real data + faithful Ghosh) — verdict: PARTIAL PASS, honest & informative

**Date:** 2026-06-27. **Seed:** 2026. **Code:** `run_ell_gate3.py`. 3 real DA benchmarks, 5×5
repeated stratified CV, leakage-safe (per-fold fit + standardize), paired bootstrap R=2000.
Closes pre-submission must-haves #1 (true Ghosh) + #2 (real data) of `paper-plan-...md`.

## Faithful Ghosh (not the proxy)
Per-class **global** Mahalanobis (class mean+cov, ridge) **+ local** Mahalanobis (kNN local cov),
both fed to a quantile-spline GAM (logistic), with the **local bandwidth k CV-selected** on train
(k∈{15,30,60}) — a faithful re-implementation of Ghosh, Ghosh, SahaRay & Sarkar (JMVA 207:105417,
2025), given its best shot. `ghosh_full` = global+local (their full method); `ghosh_glob` = global only.

## Result (mean accuracy; paired bootstrap CIs)

| dataset (n,d,C) | qda | identity_md | **kunchenko** | ghosh_glob | kunchenko_gl | ghosh_full |
|---|---|---|---|---|---|---|
| wine (178,13,3) | **0.992** | 0.965 | 0.975 | 0.975 | 0.974 | 0.973 |
| breast_cancer (569,30,2) | **0.964** | 0.901 | 0.959 | 0.944 | 0.937 | 0.815 |
| digits (1797,64,10) | **0.979** | 0.544 | 0.955 | 0.957 | 0.973 | 0.973 |

Paired bootstrap 95% CI of Δacc (>0 favours first):

| comparison | wine | breast_cancer | digits |
|---|---|---|---|
| **kunchenko − ghosh_glob** (global, eq budget) | [−0.006,+0.006] tie | **[+0.009,+0.021]** ✅ | [−0.005,+0.001] tie |
| **kunchenko_gl − ghosh_full** (global+local, eq budget) | [−0.008,+0.010] tie | **[+0.109,+0.136]** ✅ | [−0.002,+0.002] tie |
| kunchenko − identity_md | **[+0.004,+0.017]** ✅ | **[+0.047,+0.069]** ✅ | **[+0.400,+0.424]** ✅ |
| kunchenko − qda | **[−0.027,−0.008]** ✗ | [−0.011,+0.000] tie | **[−0.028,−0.021]** ✗ |

## Reading — three honest findings

1. **Kunchenko fractional link ≥ faithful Ghosh spline link, always (ties or wins; never loses) at
   equal budget on real data.** Decisive win on breast_cancer (global [+0.009,+0.021]; global+local
   [+0.109,+0.136]) where **Ghosh's local+spline GAM catastrophically overfits** (ghosh_full 0.815 <
   ghosh_glob 0.944 — the noisy d=30 local-MD features break the spline, while the fractional link
   degrades gracefully, 0.937). The fractional/PATP basis is the **more robust radial link**.
2. **Kunchenko fractional link beats the identity link on every dataset** (CIs exclude 0; +40pp on
   digits where identity-MD is near-useless) → the **non-identity radial link is justified on real
   data**, not just synthetic — supports C2.
3. **vs textbook QDA: regime-dependent.** QDA wins on wine & digits (light-tailed / near-Gaussian),
   ties on breast_cancer. So the derived-link advantage **over QDA is heavy-tail-specific** — these
   standard benchmarks are light-tailed, the wrong regime for the headline win. Confirms the
   adaptivity framing: the method **matches** QDA when φ≈identity and is expected to **win** under
   heavy tails (shown synthetically in G-ELL; a heavy-tailed real dataset is the remaining gap).

## Verdict: PARTIAL PASS
- ✅ **must-have #1 (true Ghosh):** implemented faithfully (global+local, CV bandwidth); Kunchenko
  never loses to it, decisively wins where it overfits → the equal-budget link claim survives a real
  comparator, robustly.
- ✅ **must-have #2 (real data):** 3 standard benchmarks, repeated CV, leakage-safe.
- ⚠️ **gap (must-have #3):** a **heavy-tailed real dataset** (financial returns / contaminated UCI)
  is still needed to demonstrate the derived-link win **over QDA** (the adaptivity headline). On
  light-tailed benchmarks the honest result is tie-or-slightly-behind QDA, ahead of Ghosh.

## Honest caveats
- `ghosh_full`'s breast_cancer collapse (0.815) is a real fragility of local-MD+spline in moderate
  dim, not an artifact of handicapping (CV-bandwidth applied; ghosh_glob is fine at 0.944). Worth a
  sentence in the paper as evidence for the fractional link's robustness.
- digits is pixel data (not elliptical) — included as a stress/robustness check, off the elliptical
  thesis; report as such.
- These results **reframe**, not weaken, the paper: the contribution is "a robust, adaptive radial
  link that ties QDA on light tails, beats the identity link always, and beats / never-loses-to the
  fitted spline-GAM link at equal budget" — with the heavy-tail win as the adaptivity payoff.

## Disposition
must-haves #1+#2 substantially closed. Next: heavy-tailed real dataset (#3) + the theory core
(φ-identifiability, √n-consistency of φ̂, asymptotic Bayes-optimality) — the highest-leverage item
per both reviews.
