#!/usr/bin/env python3
"""
Gate G-ELL-3 (pre-submission gate #1+#2) — REAL multivariate data + a Ghosh-style GAM PROXY.

NOTE (premortem H-001, 2026-07-03): the comparator BELOW is a *proxy* (fixed quantile
B-spline in sklearn logistic + kNN-local), NOT a faithful Ghosh implementation. The real
method (Ghosh, SahaRay & Sarkar, JMVA 2025 / arXiv 2402.08283) uses per-class UNSQUARED
distances -> penalized-spline GAM (mgcv, REML) and a kernel-LMD with bootstrap bandwidth.
The FAITHFUL comparison lives in run_ell_gate_mgcv_faithful.py / run_ell_gate4c_mgcv_faithful.py;
against it the headline "derived beats fitted" becomes a TIE (breast_cancer) — the manuscript
was corrected to a "matches, never worse" claim. This proxy script is retained for provenance.

Original intent (both JMVA-reviewer vulnerabilities of the synthetic gates):
  (#1) the Ghosh comparator was a fixed spline proxy -> a Ghosh-STYLE GAM proxy here:
       per-class GLOBAL + LOCAL (kNN) Mahalanobis distances fed to a spline GAM, with the
       local bandwidth k chosen by CV on train (equal budget). [proxy, see NOTE above]
  (#2) synthetic-only -> here standard real multivariate DA benchmarks (wine, breast_cancer,
       digits), repeated stratified CV, leakage-safe, paired bootstrap.

Heads (all consume the SAME per-class Mahalanobis distances; differ only in the radial LINK):
  qda         : sklearn QuadraticDiscriminantAnalysis (textbook baseline)
  identity_md : logistic, LINEAR in the global MDs (identity link)
  kunchenko   : logistic on the fractional/PATP radial basis of the global MDs (derived link)
  ghosh_glob  : logistic on quantile-spline GAM of the global MDs (fitted link, global only)
  kunchenko_gl: fractional basis on global+local MDs
  ghosh_full  : spline GAM on global+local MDs  == faithful Ghosh LMD-GAM (CV bandwidth)
Equal feature budget within each {kunchenko vs ghosh} comparison.

Pre-registered: on real elliptical-ish DA data, the Kunchenko fractional link should be
>= the Ghosh spline link at equal budget (paired bootstrap CI), and both >= identity link.
Honest negative reported if not. Seed 2026.
"""
import json, time, warnings
import numpy as np
from numpy.linalg import inv
from sklearn import datasets as SKD
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.model_selection import RepeatedStratifiedKFold
import run_ell_gate as G

warnings.filterwarnings("ignore")
SEED = 2026
RIDGE = 1e-3
M = 3
K_GRID = [15, 30, 60]   # local-MD bandwidth choices, CV-selected (Ghosh's best shot)


def class_stats(Xtr, ytr, classes):
    mu, Si = {}, {}
    d = Xtr.shape[1]
    for c in classes:
        Xc = Xtr[ytr == c]; mu[c] = Xc.mean(0)
        S = np.cov(Xc, rowvar=False) + RIDGE * np.eye(d)
        Si[c] = inv(S)
    return mu, Si

def global_md(X, mu, Si, classes):
    return np.column_stack([np.einsum("ni,ij,nj->n", X - mu[c], Si[c], X - mu[c]) for c in classes])

def local_md(X, Xtr, ytr, classes, k):
    return np.column_stack([G.local_maha2(X, Xtr[ytr == c], k) for c in classes])

def spline(tr, te):
    sp = SplineTransformer(n_knots=3, degree=2, include_bias=False, knots="quantile")
    return sp.fit_transform(tr), sp.transform(te)

def frac(D):  # fractional/PATP radial basis applied column-wise to a distance matrix
    return np.column_stack([G.radial_basis(D[:, j], M) for j in range(D.shape[1])])

def cv_pick_k(Xtr, ytr, classes):
    """Choose local bandwidth k for Ghosh by 3-fold CV accuracy of ghosh_full on train."""
    from sklearn.model_selection import StratifiedKFold
    best_k, best = K_GRID[0], -1.0
    for k in K_GRID:
        accs = []
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        for tri, vai in skf.split(Xtr, ytr):
            try:
                a = eval_head(Xtr[tri], ytr[tri], Xtr[vai], ytr[vai], "ghosh_full", classes, k=k)
            except Exception:
                a = 0.0
            accs.append(a)
        m = float(np.mean(accs))
        if m > best: best, best_k = m, k
    return best_k

def eval_head(Xtr, ytr, Xte, yte, kind, classes, k=30):
    if kind == "qda":
        clf = QDA(reg_param=0.05).fit(Xtr, ytr)
        return float((clf.predict(Xte) == yte).mean())
    mu, Si = class_stats(Xtr, ytr, classes)
    Dg_tr, Dg_te = global_md(Xtr, mu, Si, classes), global_md(Xte, mu, Si, classes)
    if kind in ("kunchenko_gl", "ghosh_full"):
        Dl_tr = local_md(Xtr, Xtr, ytr, classes, k); Dl_te = local_md(Xte, Xtr, ytr, classes, k)
    if kind == "identity_md":
        Ftr, Fte = Dg_tr, Dg_te
    elif kind == "kunchenko":
        Ftr, Fte = frac(Dg_tr), frac(Dg_te)
    elif kind == "ghosh_glob":
        Ftr, Fte = spline(Dg_tr, Dg_te)
    elif kind == "kunchenko_gl":
        Ftr = np.hstack([frac(Dg_tr), frac(Dl_tr)]); Fte = np.hstack([frac(Dg_te), frac(Dl_te)])
    elif kind == "ghosh_full":
        Ftr, Fte = spline(np.hstack([Dg_tr, Dl_tr]), np.hstack([Dg_te, Dl_te]))
    else:
        raise ValueError(kind)
    sc = StandardScaler().fit(Ftr); Ftr, Fte = sc.transform(Ftr), sc.transform(Fte)
    clf = LogisticRegression(max_iter=3000, C=10.0).fit(Ftr, ytr)
    return float((clf.predict(Fte) == yte).mean())

HEADS = ["qda", "identity_md", "kunchenko", "ghosh_glob", "kunchenko_gl", "ghosh_full"]

def run_dataset(name, X, y, n_splits=5, n_repeats=5):
    classes = np.unique(y)
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    accs = {h: [] for h in HEADS}
    for tri, tei in rkf.split(X, y):
        Xtr, Xte = X[tri], X[tei]; ytr, yte = y[tri], y[tei]
        sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        k = cv_pick_k(Xtr, ytr, classes)
        for h in HEADS:
            try:
                accs[h].append(eval_head(Xtr, ytr, Xte, yte, h, classes, k=k))
            except Exception:
                accs[h].append(np.nan)
    return classes, accs


if __name__ == "__main__":
    t0 = time.time()
    DSETS = [("wine", SKD.load_wine), ("breast_cancer", SKD.load_breast_cancer),
             ("digits", SKD.load_digits)]
    out = {"seed": SEED, "datasets": {}}
    for name, loader in DSETS:
        b = loader(); X = np.asarray(b.data, float); y = np.asarray(b.target)
        classes, accs = run_dataset(name, X, y)
        ci = {
            "kunchenko - ghosh_glob (global, eq budget)": G.boot_ci(accs["kunchenko"], accs["ghosh_glob"]),
            "kunchenko_gl - ghosh_full (global+local, eq budget)": G.boot_ci(accs["kunchenko_gl"], accs["ghosh_full"]),
            "kunchenko - identity_md": G.boot_ci(accs["kunchenko"], accs["identity_md"]),
            "kunchenko - qda": G.boot_ci(accs["kunchenko"], accs["qda"]),
        }
        out["datasets"][name] = {"n": int(X.shape[0]), "d": int(X.shape[1]), "C": int(len(classes)),
                                 "acc_mean": {h: float(np.nanmean(accs[h])) for h in HEADS},
                                 "acc_sd": {h: float(np.nanstd(accs[h])) for h in HEADS}, "ci": ci}
        print("=" * 86)
        print(f"{name}: n={X.shape[0]} d={X.shape[1]} C={len(classes)} (5x5 repeated stratified CV)")
        print("=" * 86)
        print(f"{'head':<14}{'acc_mean':>10}{'acc_sd':>9}")
        for h in HEADS:
            print(f"{h:<14}{np.nanmean(accs[h]):>10.4f}{np.nanstd(accs[h]):>9.4f}")
        print("Paired bootstrap 95% CI (>0 favours first):")
        for kk, vv in ci.items():
            print(f"   {kk:<48} {vv}")
    out["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(out, open("ell_gate3_results.json", "w"), indent=2)
    print(f"\nelapsed {out['elapsed_sec']}s -> ell_gate3_results.json")
