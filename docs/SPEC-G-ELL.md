# Gate G-ELL — does the *elliptical* case make the GSA↔Ghosh↔DSGE unification non-tautological?

**Pre-registered 2026-06-26. Seed family fixed to 2026. Criterion written BEFORE numbers.**
Decides whether to invest in (Phase 1) Lean elliptical/multivariate formalization and
(Phase 2) a separate follow-up paper (Paper 5) to Ku-LSU (Paper 4, IMAIAI-2026-109, under review).

## Motivation / the non-tautology argument

Paper 4 proves: in the **Gaussian** order-2 case the Kunchenko/GSA head IS Ghosh's
Mahalanobis classifier with the **identity link** (Lean `gauss_llr_eq_mahala_affine`; gate
A3(b) |Ghosh-head − LLR| = 2.7e-15). That identity is **tautological** — same object, two names.

For an **elliptical, non-Gaussian** family with radial generator ϕ
(log p_c(x) = ϕ(D_c²) − ½log|Σ_c| + const, D_c² = (x−μ_c)ᵀΣ_c⁻¹(x−μ_c)), the LLR
Λ(x) = ϕ(D₁²) − ϕ(D₀²) + const is a **NONLINEAR** function of the two Mahalanobis radii →
the identity link **fails** and a genuine **GAM (non-identity link)** is required. There the
two approaches genuinely differ: **Kunchenko derives the link analytically** from ϕ (moment-
matched projection in stochastic-polynomial space), **Ghosh fits it nonparametrically** (spline
GAM). That difference is testable. Worked case — multivariate Student-t(ν):
ϕ(u) = −((ν+d)/2)·log(1 + u/ν); Gaussian is the ν→∞ limit where ϕ becomes affine (control).

## Pre-registered checks & PASS criteria

**G-ELL-1 — identity-link failure (analytic, no MC needed).**
Sample x over the pooled support; compute true Λ(x) and (D₀², D₁²). Fit Λ ~ a + b·D₀² + c·D₁²
(OLS). Report R²_lin.
- PASS-1 iff: **Gaussian control R²_lin ≥ 0.999** (identity link exact) AND **Student-t(ν=4)
  R²_lin ≤ 0.95** (identity link demonstrably insufficient), with the gap widening as ν↓.
- This is the necessary condition for the comparison to be non-tautological.

**G-ELL-2 — Kunchenko captured-fraction of the elliptical LLR.**
Project the centered Λ onto a radial basis {ψ_k(r)} (fractional/PATP powers of the Mahalanobis
radius, the Kunchenko space) of growing order m; compute κ_m = ‖P_m Λ̊‖²/‖Λ̊‖².
- PASS-2 iff: κ_m is **monotone non-decreasing in m** AND **κ_m ≥ 0.95 by m ≤ 5** for
  Student-t(ν=4) (the analytic link is captured by a low-order Kunchenko projection).

**G-ELL-3 — Kunchenko-head vs Ghosh-GAM (the empirical, non-tautological benchmark).**
2-class multivariate Student-t(ν=4), classes differ in μ and Σ. d ∈ {4, 10}. Equal feature
budget. **Leakage-safe**: train heads on train split only; **15 repeated splits**;
**bootstrap R=2000** on the paired accuracy difference. Heads (all consume the same two
Mahalanobis heads D₀², D₁² estimated from train):
- **Kunchenko-head**: analytic radial link (project LLR onto the fractional Kunchenko basis,
  coefficients from moment-matched F·K=Y) — cheap, no link-fitting.
- **Ghosh-GAM** (proxy): global + local (kNN) Mahalanobis → smooth additive **spline-logistic**
  GAM (SplineTransformer + LogisticRegression). PROXY for Ghosh LMD-GAM (arXiv 2402.08283 /
  JMVA 2025) — a *win vs proxy is provisional* pending the true `vineclass`/Ghosh implementation;
  a *loss vs proxy is decisive*.
- **identity-QDA** (control): linear-in-(D₀²,D₁²) head (Gaussian assumption) — should trail.
- **true-LLR oracle** (upper bound): the analytic Λ — not a competitor, the ceiling.
- PASS-3 iff: **Kunchenko-head ≥ Ghosh-GAM** (paired Δacc bootstrap 95% CI excludes 0 in
  Kunchenko's favour, OR statistical tie) **AND** Kunchenko-head **> identity-QDA** (CI excludes
  0) — i.e. the analytic link both matches the fitted GAM and beats the identity link. A clean
  loss to Ghosh-GAM, or no gain over identity-QDA, is an honest **FAIL/boundary** (record it).

## Overall verdict
- **PASS** (G-ELL-1 ∧ G-ELL-2 ∧ G-ELL-3) → the elliptical extension is real and non-tautological:
  proceed to Phase 1 (Lean) + Phase 2 (Paper 5).
- **Partial** (1∧2 pass, 3 boundary) → theory holds but no empirical edge over Ghosh-GAM →
  reframe as "analytic-link = fitted-GAM at lower cost" (efficiency claim) or hold.
- **FAIL** (1 or 2 fails) → identity link suffices / Kunchenko doesn't capture → do NOT spin a paper.

## Discipline (monorepo CLAUDE.md)
Equal budget; leakage-safe split; bootstrap R≥2000 before any headline; cond(F) per class
(Tikhonov ridge, fractional basis for conditioning); criterion frozen above BEFORE numbers;
negatives reported, not hidden. Ghosh-GAM is a PROXY — flag provisional wins.
