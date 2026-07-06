"""Faithful-comparator check (premortem H-001): re-run the load-bearing binary
head-to-head against the REAL Ghosh global MD-GAM (mgcv penalized-spline GAM, REML,
per-class UNSQUARED distances) instead of the sklearn quantile-B-spline proxy.

The derived (kunchenko) / identity / qda heads are IDENTICAL to run_ell_gate3, on the
SAME 5x5 repeated-stratified folds (seed 2026). Only the fitted-link comparator changes:
  ghosh_glob  (proxy)   = SplineTransformer degree-2 quantile knots -> logistic  [old]
  ghosh_mdgam (faithful)= mgcv gam(y~s(d0)+s(d1), binomial, REML) on unsquared d  [new]

If kunchenko stays >= ghosh_mdgam (paired bootstrap CI), the manuscript's
"faithful re-implementation" claim is substantiated against the actual method and the
number can be reported honestly. Binary datasets only (mgcv binomial); breast_cancer is
the decisive Table-2 win. Seed 2026.
"""
import json, os, subprocess, tempfile, time, warnings
import numpy as np
from sklearn import datasets as SKD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.model_selection import RepeatedStratifiedKFold
import run_ell_gate as G
import run_ell_gate3 as G3

warnings.filterwarnings("ignore")
SEED = 2026
HERE = os.path.dirname(os.path.abspath(__file__))
RSCRIPT = os.path.join(HERE, "mgcv_mdgam.R")


def _mgcv_prob(Dg_tr, ybin, Dg_te):
    """One faithful binomial MD-GAM fit (mgcv, REML); returns P(y=1) on test.
    Distance columns are z-scored with train stats (monotone; only improves GAM
    conditioning for separable classes, does not change link flexibility)."""
    m, s = Dg_tr.mean(0), Dg_tr.std(0) + 1e-9
    Dg_tr, Dg_te = (Dg_tr - m) / s, (Dg_te - m) / s
    with tempfile.TemporaryDirectory() as td:
        cols = [f"d{j}" for j in range(Dg_tr.shape[1])]
        np.savetxt(f"{td}/tr.csv", np.column_stack([Dg_tr, ybin]), delimiter=",",
                   comments="", header=",".join(cols + ["y"]))
        np.savetxt(f"{td}/te.csv", Dg_te, delimiter=",", comments="", header=",".join(cols))
        r = subprocess.run(["Rscript", RSCRIPT, f"{td}/tr.csv", f"{td}/te.csv", f"{td}/out.csv"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("mgcv failed: " + r.stderr[-400:])
        arr = np.loadtxt(f"{td}/out.csv", delimiter=",", skiprows=1)
        return np.atleast_2d(arr)[:, 0]   # prob column


def ghosh_mdgam_acc(Dg_tr, ytr, Dg_te, yte):
    """Faithful Ghosh global MD-GAM; UNSQUARED per-class distances. Binary -> one fit;
    J>2 -> one-vs-rest binomial MD-GAMs, argmax of the per-class probabilities."""
    classes = np.unique(ytr)
    if len(classes) == 2:
        pred = (_mgcv_prob(Dg_tr, ytr, Dg_te) > 0.5).astype(int)
    else:
        P = np.column_stack([_mgcv_prob(Dg_tr, (ytr == c).astype(int), Dg_te) for c in classes])
        pred = classes[P.argmax(1)]
    return float((pred == yte).mean())


def eval_simple(Xtr, ytr, Xte, yte, kind, classes):
    """derived / identity / qda heads, identical to gate3 (global squared MDs)."""
    if kind == "qda":
        return float((QDA(reg_param=0.05).fit(Xtr, ytr).predict(Xte) == yte).mean())
    mu, Si = G3.class_stats(Xtr, ytr, classes)
    Dg_tr, Dg_te = G3.global_md(Xtr, mu, Si, classes), G3.global_md(Xte, mu, Si, classes)
    if kind == "identity_md":
        Ftr, Fte = Dg_tr, Dg_te
    elif kind == "kunchenko":
        Ftr, Fte = G3.frac(Dg_tr), G3.frac(Dg_te)
    sc = StandardScaler().fit(Ftr)
    clf = LogisticRegression(max_iter=3000, C=10.0).fit(sc.transform(Ftr), ytr)
    return float((clf.predict(sc.transform(Fte)) == yte).mean())


def run_dataset(name, X, y):
    classes = np.unique(y)
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    heads = ["qda", "identity_md", "kunchenko", "ghosh_mdgam"]
    accs = {h: [] for h in heads}
    for tri, tei in rkf.split(X, y):
        Xtr, Xte = X[tri], X[tei]; ytr, yte = y[tri], y[tei]
        sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        for h in ("qda", "identity_md", "kunchenko"):
            accs[h].append(eval_simple(Xtr, ytr, Xte, yte, h, classes))
        # faithful MD-GAM on UNSQUARED per-class distances (sqrt of the squared MDs)
        mu, Si = G3.class_stats(Xtr, ytr, classes)
        Dg_tr = np.sqrt(np.clip(G3.global_md(Xtr, mu, Si, classes), 0, None))
        Dg_te = np.sqrt(np.clip(G3.global_md(Xte, mu, Si, classes), 0, None))
        accs["ghosh_mdgam"].append(ghosh_mdgam_acc(Dg_tr, ytr, Dg_te, yte))
    ci = {
        "kunchenko - ghosh_mdgam (FAITHFUL, global)": G.boot_ci(accs["kunchenko"], accs["ghosh_mdgam"]),
        "kunchenko - identity_md": G.boot_ci(accs["kunchenko"], accs["identity_md"]),
        "kunchenko - qda": G.boot_ci(accs["kunchenko"], accs["qda"]),
    }
    return heads, accs, ci


if __name__ == "__main__":
    t0 = time.time()
    out = {"seed": SEED, "comparator": "faithful mgcv global MD-GAM (REML)", "datasets": {}}
    # Elliptical, on-thesis benchmarks only. digits (10-class non-elliptical pixel data)
    # is dropped from the FAITHFUL head-to-head: it was already a tie vs the proxy and a
    # 10-class faithful GAM is disproportionate and off-thesis (kept as a proxy stress).
    for name, loader in [("wine", SKD.load_wine), ("breast_cancer", SKD.load_breast_cancer)]:
        b = loader(); X = np.asarray(b.data, float); y = np.asarray(b.target)
        heads, accs, ci = run_dataset(name, X, y)
        out["datasets"][name] = {"n": int(X.shape[0]), "d": int(X.shape[1]),
                                 "acc_mean": {h: float(np.nanmean(accs[h])) for h in heads},
                                 "acc_sd": {h: float(np.nanstd(accs[h])) for h in heads}, "ci": ci}
        print("=" * 74)
        print(f"{name}: n={X.shape[0]} d={X.shape[1]} (5x5 CV) — FAITHFUL mgcv MD-GAM")
        print("=" * 74)
        for h in heads:
            print(f"{h:<14}{np.nanmean(accs[h]):>10.4f}  sd {np.nanstd(accs[h]):>7.4f}")
        print("Paired bootstrap 95% CI (>0 favours first):")
        for k, v in ci.items():
            print(f"   {k:<44} {v}")
    out["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(out, open("ell_gate_mgcv_faithful_results.json", "w"), indent=2)
    print(f"\nelapsed {out['elapsed_sec']}s -> ell_gate_mgcv_faithful_results.json")
