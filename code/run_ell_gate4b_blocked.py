"""G-ELL-4B-blocked  — temporal-dependence-robust FX validation.

Referee point (PaperMentor CRITICAL, experiments.tex:100): the FX features are
overlapping d=5 return embeddings and the label is a 21-day trailing-vol tercile,
so both are temporally autocorrelated. Random stratified CV + an i.i.d. paired
bootstrap leak temporally-adjacent (near-identical) rows across train/test and
understate the CIs. This script re-evaluates the primary heavy-tail result with:

  (1) forward-chaining (expanding-window) splits — train on the past, test on the
      future — with an EMBARGO gap of vol_win+d=26 filtered samples between train
      and test, killing the label-window + embedding overlap at the boundary;
  (2) a MOVING-BLOCK bootstrap (block length = vol_win = 21) on the pooled
      out-of-sample paired correctness differences, which respects autocorrelation
      instead of assuming i.i.d. draws.

If the derived-link advantage survives this, the FX claim is robust; if it
shrinks, we report it honestly. Heads reuse the faithful run_ell_gate3 internals.
Seed 2026.
"""
import json
import numpy as np
from numpy.linalg import inv
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
import run_ell_gate as G
import run_ell_gate3 as G3
import run_ell_gate4 as G4

SEED = 2026
VOL_WIN = 21
D = 5
EMBARGO = VOL_WIN + D          # 26 filtered samples between train and test
N_SPLITS = 8
BLOCK = VOL_WIN                # moving-block length for the bootstrap
B = 2000
HEADS = ["qda", "identity_md", "kunchenko", "ghosh_glob"]


def predict_head(Xtr, ytr, Xte, kind, classes):
    """Mirror G3.eval_head but RETURN predictions (need per-sample correctness)."""
    if kind == "qda":
        return QDA(reg_param=0.05).fit(Xtr, ytr).predict(Xte)
    mu, Si = G3.class_stats(Xtr, ytr, classes)
    Dg_tr, Dg_te = G3.global_md(Xtr, mu, Si, classes), G3.global_md(Xte, mu, Si, classes)
    if kind == "identity_md":
        Ftr, Fte = Dg_tr, Dg_te
    elif kind == "kunchenko":
        Ftr, Fte = G3.frac(Dg_tr), G3.frac(Dg_te)
    elif kind == "ghosh_glob":
        Ftr, Fte = G3.spline(Dg_tr, Dg_te)
    else:
        raise ValueError(kind)
    sc = StandardScaler().fit(Ftr)
    clf = LogisticRegression(max_iter=3000, C=10.0).fit(sc.transform(Ftr), ytr)
    return clf.predict(sc.transform(Fte))


def forward_chunks(n, n_splits, embargo):
    """Expanding-window forward-chaining test blocks with an embargo gap."""
    fold = n // (n_splits + 1)
    for k in range(1, n_splits + 1):
        tr_end = fold * k
        te_lo = tr_end + embargo
        te_hi = min(tr_end + fold, n)
        if te_lo >= te_hi:
            continue
        yield np.arange(0, tr_end), np.arange(te_lo, te_hi)


def purged_blocks(n, k_folds, embargo):
    """Purged-and-embargoed blocked K-fold (López de Prado): contiguous test block,
    train = complement minus an embargo window on both sides. Isolates adjacency
    leakage from forward distribution shift (train sees all regimes)."""
    edges = np.linspace(0, n, k_folds + 1).astype(int)
    for k in range(k_folds):
        lo, hi = edges[k], edges[k + 1]
        te = np.arange(lo, hi)
        keep = np.ones(n, bool)
        keep[max(0, lo - embargo):min(n, hi + embargo)] = False
        tr = np.where(keep)[0]
        yield tr, te


def moving_block_boot(diff, block, B, rng):
    """Moving-block bootstrap CI for the mean of an autocorrelated paired diff."""
    diff = diff[~np.isnan(diff)]
    T = len(diff)
    nb = int(np.ceil(T / block))
    starts_pool = np.arange(0, T - block + 1)
    means = np.empty(B)
    for b in range(B):
        starts = rng.choice(starts_pool, size=nb, replace=True)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:T]
        means[b] = diff[idx].mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def evaluate(Xk, yk, classes, splits, scheme):
    n = len(yk)
    correct = {h: [] for h in HEADS}
    n_test, folds_used = 0, 0
    for tri, tei in splits:
        ytr, yte = yk[tri], yk[tei]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue                                   # skip degenerate single-class block
        folds_used += 1
        sc = StandardScaler().fit(Xk[tri])             # train-only scaling (leakage-safe)
        Xtr, Xte = sc.transform(Xk[tri]), sc.transform(Xk[tei])
        for h in HEADS:
            try:
                pred = predict_head(Xtr, ytr, Xte, h, classes)
                correct[h].append((pred == yte).astype(float))
            except Exception:
                correct[h].append(np.full(len(yte), np.nan))
        n_test += len(tei)
    corr = {h: np.concatenate(correct[h]) for h in HEADS}
    acc = {h: float(np.nanmean(corr[h])) for h in HEADS}
    rng = np.random.default_rng(SEED)
    pairs = {"kunchenko - qda": ("kunchenko", "qda"),
             "kunchenko - ghosh_glob": ("kunchenko", "ghosh_glob"),
             "kunchenko - identity_md": ("kunchenko", "identity_md")}
    ci = {}
    for name, (a, b) in pairs.items():
        diff = corr[a] - corr[b]
        lo, hi = moving_block_boot(diff, BLOCK, B, rng)
        ci[name] = {"mean_diff": round(float(np.nanmean(diff)), 4), "ci": [round(lo, 4), round(hi, 4)]}
    print(f"\n[{scheme}]  n={n}, OOS rows={n_test}, folds used={folds_used}, "
          f"embargo={EMBARGO}, block={BLOCK}")
    print(f"  {'head':<14}{'OOS acc':>9}")
    for h in HEADS:
        print(f"  {h:<14}{acc[h]:>9.4f}")
    print("  moving-block bootstrap 95% CI of paired acc diff (>0 favours derived link):")
    for name, v in ci.items():
        sig = "" if v["ci"][0] > 0 or v["ci"][1] < 0 else "  (n.s.)"
        print(f"    {name:<26} mean={v['mean_diff']:+.4f}  CI={v['ci']}{sig}")
    return {"scheme": scheme, "n_oos_test": int(n_test), "folds_used": folds_used,
            "acc_oos": acc, "ci": ci}


def main():
    Xk, yk = G4.build_fx(d=D, vol_win=VOL_WIN)          # time-ordered, middle tercile dropped
    classes = np.unique(yk)
    n = len(yk)
    out = {"seed": SEED, "n": int(n), "d": D, "embargo": EMBARGO, "block": BLOCK, "B": B,
           "results": {}}
    out["results"]["forward_chaining"] = evaluate(
        Xk, yk, classes, forward_chunks(n, N_SPLITS, EMBARGO),
        "forward-chaining expanding window + embargo")
    out["results"]["purged_kfold"] = evaluate(
        Xk, yk, classes, purged_blocks(n, 6, EMBARGO),
        "purged-and-embargoed blocked 6-fold (isolates adjacency leakage)")
    json.dump(out, open("ell_gate4b_results.json", "w"), indent=2)
    print("\nwrote ell_gate4b_results.json")


if __name__ == "__main__":
    main()
