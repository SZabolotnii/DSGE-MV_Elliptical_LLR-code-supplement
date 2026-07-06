# VERDICT — G-ELL-4B (FX temporal-dependence-robust re-validation)

**Trigger.** PaperMentor CRITICAL (experiments.tex:100): the FX features are
overlapping $d=5$ return embeddings and the label is a 21-day trailing-vol
tercile — both temporally autocorrelated. Random stratified CV + an i.i.d. paired
bootstrap leak adjacent near-identical rows across train/test and understate the
CIs, so the original "beats QDA, bootstrap-significant" may be inflated.

**Method.** `run_ell_gate4b_blocked.py`. Two temporal-aware schemes, both with
train-only scaling, an embargo of `vol_win+d=26` filtered samples, and a
**moving-block bootstrap** (block = 21) on pooled out-of-sample paired
correctness:
- **forward-chaining** (expanding window, rolling-origin) — adds a 1971→2026
  distribution-shift penalty on top of leakage removal;
- **purged-and-embargoed blocked 6-fold** (López de Prado) — isolates *adjacency
  leakage* while letting train see all regimes (the fair in-distribution test).

**Results (OOS accuracy; moving-block 95% CI of paired Δacc).**

| comparison | purged 6-fold | forward-chaining | robust verdict |
|---|---|---|---|
| kunchenko − qda        | +0.004 [−0.010, +0.018] n.s. | +0.019 [−0.004, +0.042] n.s. | **TIE** |
| kunchenko − ghosh_glob | +0.020 **[+0.015, +0.027]**  | −0.003 [−0.011, +0.003] n.s. | derived > fitted (leakage-free) |
| kunchenko − identity   | +0.015 **[+0.004, +0.027]**  | +0.027 **[+0.014, +0.042]**  | **derived > identity** |

OOS accuracy (purged 6-fold): qda 0.825, identity 0.814, kunchenko 0.829,
ghosh_glob 0.809. (i.i.d. 5×5 CV gave ~0.85 for kunchenko and a small significant
edge over QDA — that edge is the leakage artifact.)

**Finding — the referee is right; the vs-QDA FX win does not survive; the central
thesis does.**
1. **kunchenko vs QDA: TIE** under both temporal-aware schemes. The original
   "beats QDA on real heavy-tailed data, bootstrap-significant" was inflated by
   temporal leakage and is **withdrawn**. The vs-QDA heavy-tail advantage rests on
   the controlled rate simulation + real-covariance ν-sweep (no temporal leakage).
2. **kunchenko vs the fitted Ghosh link: derived WINS** once adjacency leakage is
   removed (purged 6-fold, CI [+0.015,+0.027]); it washes out only under the
   harsher forward distribution shift. The paper's *central* claim — a *derived*
   link beats a *fitted* link at equal budget — therefore **holds on real
   heavy-tailed data**, leakage-free.
3. **kunchenko vs the identity link: derived WINS** in both schemes — a
   non-identity link is needed even on real heavy-tailed data.

**Manuscript actions.** §exp-heavy: report the temporal-robust validation as
primary, withdraw "beats QDA … all bootstrap-significant", reframe the robust FX
message as derived > fitted + identity (vs-QDA = tie; vs-QDA advantage from
simulation). Mirror the softening in abstract, intro C4, conclusion. Script +
`ell_gate4b_results.json` retained as provenance.
