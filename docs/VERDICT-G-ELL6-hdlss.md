# VERDICT — G-ELL-6 (HDLSS / d≫n probe)

**Question.** Does the derived radial link keep its equal-budget advantage when the
dimension approaches/exceeds the per-class sample size, with a ridge-regularised
per-class scatter?

**Setup.** `run_ell_gate6_hdlss.py`. Two elliptical Student-$t$ classes, $\nu=8$
(inside the full-basis moment scope $\nu>6$), covariance contrast $a=3$ (so the link
matters), $n_{\text{tr}}=40$/class, $n_{\text{te}}=1000$, 12 reps. All heads consume the
**same** global Mahalanobis radii from a ridge-regularised per-class scatter
(`G3.class_stats`, RIDGE=1e-3); they differ only in the radial-link basis. Equal budget:
`identity` (raw radii), `kunchenko` (fixed fractional-power basis, $M=3$), `ghosh`
(quantile-spline GAM on the same global radii = `ghosh_glob`).

**Result (test accuracy, `ell_gate6_results.json`).**

| d   | d/n | identity | kunchenko (derived) | ghosh (adaptive spline) |
|-----|-----|----------|---------------------|--------------------------|
| 20  | 0.5 | 0.740    | 0.695               | 0.672 |
| 60  | 1.5 | 0.520    | 0.513               | 0.624 |
| 120 | 3.0 | 0.501    | 0.501               | 0.668 |
| 200 | 5.0 | 0.500    | 0.500               | 0.631 |

**Finding — NEGATIVE for the derived link at d≫n, and honest.**
As $d/n$ grows past 1, the global Mahalanobis radii — estimated from a rank-deficient,
ridge-loaded scatter — **saturate**, and the *fixed* fractional-power (derived) basis
collapses to chance, exactly tracking the identity link. The *data-adaptive*
quantile-spline link retains signal ($\approx0.63$–$0.67$). Normalising the radius by $d$
before the power basis does **not** rescue it (checked: still $0.50$ at $d=120,200$), so
this is not a scale-saturation artifact — it is the fixed, non-adaptive basis failing once
the radii lose resolution.

**Mechanism.** With $n=40\ll d$ the sample covariance is rank-deficient; the ridge term
dominates, so $D_c(x)\approx\|x-\hat\mu_c\|^2/\text{ridge}$ and the two radii become
near-collinear and near-constant in ratio. A fixed power basis of two collinear, saturated
numbers is rank-deficient after standardisation; quantile-spline knots adapt to the
residual within-class quantile structure and keep a usable direction.

**Implication for the paper (scope, not defect).** The structural and identifiability
results (Prop. bridge/dichotomy, Lemma ident) are dimension-free and unaffected — they are
algebraic identities in the radii. What is a **moderate-$d$ phenomenon** is the *practical*
derived-link advantage: it rests on well-estimated radii, which require $n\gg d$. The
honest scope is therefore $d\ll n$ (our benchmarks: $d\le64$). Recovering a $d\gg n$
advantage needs (i) a high-dimensional scatter (regularised/robust, not raw sample
covariance) and (ii) a data-adaptive or moment-free radius basis. Left to future work.

**Disposition.** Not promoted to a headline experiment (the mechanism is a
radius-estimation breakdown that deserves its own study, not a results subsection here).
Used to **sharpen** Discussion limitation (iv) from "untested" to a precise,
evidence-backed scope boundary. Script + `ell_gate6_results.json` retained in the repo as
provenance.
