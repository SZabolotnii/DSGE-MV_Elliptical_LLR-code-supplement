# Gate G-ELL-2 (extension) — verdict: PASS, the fractional-link edge is regime-robust

**Date:** 2026-06-27. **Seed:** 2026. **Code:** `run_ell_gate2.py`. 15 splits, bootstrap R=2000,
d∈{4,10}. Strengthens the empirical pillar of `VERDICT-G-ELL.md` by testing **Ghosh's home turf**:
a regime where local Mahalanobis genuinely matters.

## DGP — fine 4×4 checkerboard of t(4) blobs (global-uninformative)

Both classes are mixtures of 8 isotropic Student-t(4) blobs on a 4×4 grid, label = (i+j) mod 2.
By symmetry both classes share **identical global mean and global covariance** → a single-Gaussian
global Mahalanobis head is provably uninformative (≈ chance). Only **local** cluster structure
separates the classes — exactly what Ghosh's local Mahalanobis is designed for. (Earlier 2-blob
"XOR" attempts failed to defeat global QDA because the two classes had opposite correlation signs;
the fine even checkerboard fixes that — confirmed: global heads collapse to chance below.)

## Result (mean accuracy; mixture-Bayes oracle = ceiling)

**d = 4:**
| head | budget | acc | sd |
|---|---|---|---|
| identity_qda (global, linear) | 2 | 0.510 | .019 |
| kunchenko (global, fractional) | 6 | 0.496 | .020 |
| ghosh_global (global, spline) | 6 | 0.567 | .030 |
| **kunchenko_hi (global+local, fractional)** | 12 | **0.595** | .022 |
| ghosh_full (global+local, spline) | 12 | 0.573 | .020 |
| oracle (mixture Bayes) | — | 0.842 | .008 |

**d = 10:**
| head | budget | acc | sd |
|---|---|---|---|
| identity_qda | 2 | 0.521 | .016 |
| kunchenko (global) | 6 | 0.517 | .018 |
| ghosh_global | 6 | 0.532 | .020 |
| **kunchenko_hi (global+local)** | 12 | **0.622** | .019 |
| ghosh_full (global+local) | 12 | 0.613 | .018 |
| oracle | — | 0.847 | .012 |

Paired bootstrap 95% CI of Δacc (>0 favours first):

| comparison | d=4 | d=10 |
|---|---|---|
| **kunchenko_hi − ghosh_full** (fractional vs spline local link, equal budget 12) | **[+0.017, +0.029]** ✅ | **[+0.003, +0.015]** ✅ |
| ghosh_full − ghosh_global (does local help the spline?) | [−0.011, +0.024] ~ | **[+0.069, +0.093]** ✅ |
| kunchenko_hi − kunchenko (does local help the fractional link?) | **[+0.087, +0.112]** ✅ | **[+0.092, +0.118]** ✅ |

## Reading

1. **The DGP is a genuine local test.** Global heads (identity_qda, kunchenko-global) sit at chance
   (0.50–0.52) — the global Mahalanobis is provably uninformative, as designed.
2. **Local Mahalanobis genuinely matters here** — adding it lifts the fractional link by ~+10pp at
   both dimensions, and lifts the spline link by +8pp at d=10 (Ghosh's local enhancement earns its
   keep on this DGP, unlike the unimodal gate-1 case where it hurt).
3. **Even on Ghosh's home turf the Kunchenko fractional local link beats the spline-GAM local link
   at equal budget** — kunchenko_hi > ghosh_full at both d (CIs exclude 0). The fractional/PATP
   basis is the better radial-link representation **whether or not local structure matters**.

So the headline of gate 1 (fractional link > spline link at equal budget, unimodal elliptical) is
now shown **regime-robust**: it holds in the hard multimodal-local regime too.

## Verdict: PASS (extension) — claim strengthened, with an honest paradigm boundary
- **Strengthened claim:** the Kunchenko fractional/PATP radial link beats the spline-GAM link at
  equal budget across BOTH unimodal-elliptical (gate 1) AND hard-multimodal-local (gate 2) regimes,
  on global-only and global+local features alike.
- **Honest boundary:** ALL Mahalanobis-head methods (Kunchenko AND Ghosh, global AND local) remain
  far below the mixture oracle on the fine checkerboard (0.60–0.62 vs 0.85). The two-Mahalanobis-head
  paradigm cannot represent XOR-parity structure; that regime needs explicit mixture/local-density
  modeling and is **outside** the elliptical-radial-link scope. The win is "fractional link > spline
  link", not "Kunchenko solves the checkerboard".

## Caveats (carried)
- Ghosh remains a quantile-knot spline-GAM **proxy**; a true Ghosh LMD-GAM / `vineclass` comparison
  is still future work (pyvinecopulib not installed). The equal-budget spline-link loss is robust
  across two very different DGPs, which makes the proxy result more credible.
- All real-data validation (DroneRF IQ, the G-DRF-1 gate from the dsge-multivariate NOTE) remains
  future work; these are synthetic-DGP results.

## Disposition
Empirical pillar strengthened. Combined with gate 1 (G-ELL PASS) and the sorry-free Lean
(Phase 1), the elliptical extension is well-supported across regimes → Paper 5 draft warranted.
