# Gate G-ELL-5 (rate simulation — numerical verification of the theory core) — verdict: PASS

**Date:** 2026-06-27. **Seed:** 2026. **Code:** `run_ell_gate5_rates.py`. **Results:** `ell_gate5_results.json`.
Numerically verifies `THEORY-CORE.md` against a **known** ground-truth elliptical model where the true
link φ, true LLR Λ, and Bayes risk R* are all closed-form: two-class Student-t, X|0 ~ t_ν(0, I),
X|1 ~ t_ν(μ₁, a·I) in ℝ^d (a = the covariance-contrast knob; ‖μ₁‖ = 1.5). Every regime below is chosen
to sit **inside the formal CLT scope** (ν = 8 > 6 ⇒ the full M = 3 radial basis {1, 0.5, 1.5} has finite
2nd moments, A1; ν > 4 ⇒ Σ̂ is √n, A3), except the explicitly-labelled illustrative tail (ν < 6) in C2.

## (A) Theorem 3 — √n-consistency of the radial logistic M-estimator (d=4, ν=8, a=1, in scope)

TRUE whitening + fixed feature scaler; β* fit on N = 120k; β̂_n on growing n; 16 reps.

| n | 1000 | 2000 | 4000 | 8000 | 16000 | 32000 |
|---|---|---|---|---|---|---|
| ‖β̂_n − β*‖ | 3.53 | 2.30 | 1.61 | 0.92 | 0.56 | 0.58 |

**log-log slope = −0.57** (theory predicts **−0.5**; the last node flattens at the ℓ₂-regularization
floor). ✅ Confirms β̂_n is √n-consistent (the error is O(n^{−1/2}), not flat / not slower).

## (B) Theorem 5 / Corollary 6 — QDA plateau vs derived-link vanishing (ν=8, a=3, IN SCOPE)

Full ESTIMATED pipeline (μ̂, Σ̂ from train); excess risk = R(clf) − R* on a 40k test set; 10 reps.
**Bayes risk R* = 0.1714.**

| n | 400 | 800 | 1600 | 3200 | 6400 | 12800 |
|---|---|---|---|---|---|---|
| excess QDA (affine link) | 0.0212 | 0.0188 | 0.0185 | 0.0190 | 0.0178 | **0.0177** |
| excess derived link (m=3) | 0.0287 | 0.0146 | 0.0074 | 0.0027 | 0.0014 | **0.0005** |

**Textbook Corollary 6, in scope:** QDA's excess risk **plateaus at a constant ≈ 0.018** (its affine
radial link cannot represent the logarithmic t-link; the gap does not shrink with n), while the
**derived link's excess risk vanishes** 0.029 → 0.0005. ✅

## (C) The two levers of the QDA gap (refined Cor 6 / Heuristic 6′), fixed n = 6400, d = 10

**C1 — covariance contrast (ν = 8, IN SCOPE):**

| a (scale ratio) | 1.0 | 2.0 | 3.0 | 4.0 | 6.0 |
|---|---|---|---|---|---|
| gap = excess_QDA − excess_derived | **−0.0001** | 0.0231 | 0.0187 | 0.0132 | 0.0045 |

Equal covariance (a=1) ⇒ pure location shift ⇒ near the boundary D₀²≈D₁², φ(D₁²)−φ(D₀²) is first-order
linear ⇒ QDA optimal, **gap ≈ 0**. Covariance heterogeneity breaks the cancellation ⇒ gap opens
(peak ≈ a=2–3); at a=6 the classes nearly separate (R*≈0.09) and it shrinks. **This is why the
equal-Σ toy hides the effect and the real per-class-Σ benchmarks (breast_cancer, FX) show it.**

**C2 — tail-heaviness (a = 3; ν ≥ 6 in scope, ν < 6 illustrative):**

| ν | 12 | 8 | 6 | 5 | 4 | 3 |
|---|---|---|---|---|---|---|
| excess QDA | 0.0091 | 0.0199 | 0.0332 | 0.0468 | 0.0678 | 0.1158 |
| excess derived (m=3) | 0.0016 | 0.0016 | 0.0026 | 0.0038 | 0.0050 | 0.0217 |
| **gap** | 0.0075 | 0.0182 | 0.0306 | 0.0429 | 0.0628 | 0.0941 |

The gap grows **monotonically as the tail heavies** (Heuristic 6′), clearly even in the in-scope range
ν ≥ 6. The derived link stays near the sieve floor throughout; QDA degrades steeply.

## Verdict: PASS
All three theoretical predictions hold numerically against a closed-form ground truth, **in the formal
CLT scope**: (A) the n^{−1/2} estimation rate (Theorem 3); (B) the constant-QDA-gap-vs-vanishing-derived-
link dichotomy (Theorem 5 / Corollary 6); (C) the two structural levers (covariance heterogeneity +
tail-heaviness) that set the gap's magnitude — the precise mechanism behind the G-ELL-4 ν-sweep and the
CAD/USD heavy-tail win, and the refinement that the advantage requires non-trivial Σ-contrast and/or
heavy tails (it vanishes for equal-Σ location shift). With `THEORY-CORE.md` (pen-and-paper) and the
sorry-free `EllipticalUnification.lean` (structural layer), the theory core (Task А) is in place and
numerically corroborated: identifiability → √n-consistency+CLT → iterated-limit Bayes-optimality.

## Honest caveats
- (A) slope −0.57 is over the full n-range; the point is the rate class (n^{−1/2}), confirmed; the last
  node flattens at the mild-ℓ₂ floor. The clean rate uses d=4 (the gap demos use d=10) — part A tests the
  estimator rate, which is independent of the gap.
- (B)/(C) the derived-link residual (~0.0005–0.002 in scope) is the fixed-m sieve floor (Theorem 5 sends
  it to 0 only as m→∞); finite-m here, reported as such — not an over-claim of exact 0.
- C2's ν = 5, 4, 3 rows are **beyond** the full-basis moment scope (ν > 6) and the A3 scope (ν > 4); shown
  as the illustrative heavy-tail continuation, consistent with the in-scope ν ≥ 6 trend.
- single generator family (Student-t); a misspecified-generator rate check is the optional roadmap item.
