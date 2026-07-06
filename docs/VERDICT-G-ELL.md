# Gate G-ELL — verdict: PASS (the elliptical case is real and non-tautological)

**Date:** 2026-06-26. **Seed:** 2026. **Code:** `run_ell_gate.py`. **Spec (pre-registered):**
`SPEC-G-ELL.md`. Elapsed 5.7s. Decides Phase 1 (Lean elliptical/multivariate) + Phase 2 (Paper 5).

## G-ELL-1 — identity-link failure ✅ PASS

R² of the **linear** fit of the true LLR on (D₀², D₁²) — i.e. how well the *identity link*
(Gaussian/Ghosh order-2) reproduces the log-likelihood-ratio:

| family | R²_lin | reading |
|---|---|---|
| gaussian | **1.0000** | identity link **exact** (control — Λ is affine in the radii) |
| t(ν=30) | 0.9515 | near-Gaussian |
| t(ν=8) | 0.6591 | identity link degrading |
| **t(ν=4)** | **0.2399** | identity link captures only 24% → **demonstrably insufficient** |
| t(ν=2.5) | 0.0091 | identity link useless |

PASS-1 (gaussian ≥0.999 AND t4 ≤0.95): **met** (1.000 / 0.240). The Gaussian↔Ghosh identity-link
equivalence is **special to the Gaussian carrier**; for elliptical t the LLR is genuinely
nonlinear in the Mahalanobis radii → the GSA↔Ghosh comparison stops being tautological.

## G-ELL-2 — Kunchenko captured-fraction of the elliptical LLR ✅ PASS

κ_m = ‖P_m Λ̊‖²/‖Λ̊‖² on the fractional/PATP radial basis, m = 1…5:

| family | m=1 | m=2 | m=3 | m=4 | m=5 |
|---|---|---|---|---|---|
| gaussian | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 |
| t(ν=8) | 0.659 | 0.995 | 0.996 | 0.998 | 0.999 |
| **t(ν=4)** | 0.240 | **0.950** | 0.986 | 0.996 | 0.998 |
| t(ν=2.5) | 0.009 | 0.707 | 0.824 | 0.912 | *0.025* |

PASS-2 (κ_m monotone, ≥0.95 by m≤5 for t4): **met** (κ₂=0.950). Gaussian **terminates at m=1**
(matches Paper 4's termination theorem). The fractional Kunchenko basis captures the elliptical
log-link with ~2 terms — the analytic-link claim. *Honest boundary:* at ν=2.5 (near the
2nd-moment-divergence edge) the high-order radial projection becomes ill-conditioned (κ collapses
at m=5) — consistent with the Pillar-A finding that the extreme tail needs the moment-free route.

## G-ELL-3 — Kunchenko-head vs Ghosh-GAM ✅ PASS (d=10 clear; d=4 marginal)

t(ν=4), 15 repeated leakage-safe splits, equal budget, bootstrap R=2000. Both "heads" are
additive-radial-link logistic classifiers on the same global Mahalanobis inputs — differing only
in the **basis** (Kunchenko fractional/PATP vs Ghosh quantile-knot B-spline). `_hi`/`full` add
local (kNN) Mahalanobis.

**d = 4:**
| head | budget | acc | sd |
|---|---|---|---|
| identity_qda | 2 | 0.7324 | .023 |
| **kunchenko** | 6 | **0.7372** | .026 |
| ghosh_global | 6 | 0.7322 | .028 |
| kunchenko_hi | 12 | 0.7218 | .023 |
| ghosh_full | 12 | 0.7191 | .030 |
| oracle | — | 0.7484 | .020 |

**d = 10:**
| head | budget | acc | sd |
|---|---|---|---|
| identity_qda | 2 | 0.7867 | .027 |
| **kunchenko** | 6 | **0.7921** | .026 |
| ghosh_global | 6 | 0.7723 | .041 |
| kunchenko_hi | 12 | 0.7436 | .024 |
| ghosh_full | 12 | 0.7508 | .030 |
| oracle | — | 0.8118 | .024 |

Paired bootstrap 95% CI of Δacc (>0 favours first):

| comparison | d=4 | d=10 |
|---|---|---|
| **kunchenko − ghosh_global** (equal budget 6) | [+0.0003, +0.0098] ✅ | **[+0.0112, +0.0292]** ✅ |
| kunchenko_hi − ghosh_full (equal budget 12) | [−0.0021, +0.0076] tie | [−0.0165, +0.0036] tie |
| kunchenko − identity_qda | [−0.0003, +0.0095] ~tie | **[+0.0020, +0.0085]** ✅ |

PASS-3 (Kunchenko ≥ Ghosh AND Kunchenko > identity): **met at d=10** (both CIs exclude 0),
**marginal at d=4** (beats Ghosh-spline significantly; ties identity-QDA). Reading:
- **The fractional/PATP radial link beats the B-spline link at equal budget** — significantly,
  and the margin **grows with dimension** (d=4 +0.5pp → d=10 +2.0pp). This is the headline: for
  elliptical heavy-tailed Mahalanobis structure the Kunchenko basis is a better link than splines.
- **Kunchenko ≈ oracle** (d=10: 0.792 vs 0.812) — the analytic-style link nears the Bayes ceiling.
- **Local Mahalanobis HURTS here** (`_hi`/`full` < global): a single elliptical component per
  class → local cov only adds variance. So Ghosh's global+local enhancement gives no edge on
  *unimodal elliptical* classes (honest; on mixture/multi-modal classes it would matter — not tested).

## Overall verdict: PASS
All three checks pass (G-ELL-3 clear at d=10, marginal at d=4). The elliptical case makes the
GSA↔Ghosh↔DSGE unification **non-tautological** and gives a **real, equal-budget empirical edge**
for the Kunchenko fractional/PATP radial link over the spline-GAM link, growing with dimension and
nearing the oracle. → **Proceed to Phase 1 (Lean) + Phase 2 (Paper 5).**

## Honest caveats (recorded)
- **Ghosh is a PROXY** (quantile-knot spline-GAM on global+local Mahalanobis). A win vs proxy is
  *provisional* pending the true Ghosh LMD-GAM / `vineclass`; but the spline link's loss at equal
  budget is informative and the comparison is fair (quantile knots, equal feature count).
- **DGP = single elliptical component per class** → favours global Mahalanobis (why local hurts).
  Scope the "beats Ghosh-full" claim to unimodal elliptical; multi-modal is future work.
- **ν=2.5** radial-projection conditioning breaks at high order → extreme tail = moment-free
  (CF) territory, not the fractional basis. Consistent with Ku-2D Pillar A.
- d=4 edge over identity-QDA is not significant; the radial-nonlinearity payoff **grows with d**.

## Disposition
Theory extension warranted. Phase 1: Lean — (i) multivariate-Gaussian quadratic-form membership
(span of quadratic monomials, reusing the dimension-free bridge thms 1–10), (ii) elliptical
radial-link theorem Λ ∈ span{1, ψ(D₀), ψ(D₁)} with non-identity link. Phase 2: separate paper
(Paper 5), NOT a revision of Paper 4 (IMAIAI-2026-109 is under editorial review).
