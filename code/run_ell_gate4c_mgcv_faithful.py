"""Faithful-comparator check on the 6 heavy-tailed series (premortem H-001).

Same purged-and-embargoed blocked 6-fold + moving-block bootstrap as G-ELL-4C
(reuses run_ell_gate4b_blocked), same seed 2026, same kunchenko/qda heads. Only the
fitted comparator changes: the sklearn quantile-B-spline proxy -> the REAL Ghosh global
MD-GAM (mgcv penalized-spline, REML, per-class UNSQUARED distances). Tests whether the
central multi-series claim "derived link is never significantly worse than the fitted
link" survives against the actual method (ties are acceptable for that claim).
"""
import json, os, subprocess, tempfile
import numpy as np
from scipy.stats import kurtosis
import run_ell_gate3 as G3
import run_ell_gate4b_blocked as B
from run_ell_gate4c_multiseries import SERIES, STORE, load_returns, build_fx, D

HERE = os.path.dirname(os.path.abspath(__file__))
RSCRIPT = os.path.join(HERE, "mgcv_mdgam.R")


def mgcv_preds(Dg_tr_unsq, ytr, Dg_te_unsq):
    with tempfile.TemporaryDirectory() as td:
        cols = [f"d{j}" for j in range(Dg_tr_unsq.shape[1])]
        np.savetxt(f"{td}/tr.csv", np.column_stack([Dg_tr_unsq, ytr]), delimiter=",",
                   comments="", header=",".join(cols + ["y"]))
        np.savetxt(f"{td}/te.csv", Dg_te_unsq, delimiter=",", comments="", header=",".join(cols))
        r = subprocess.run(["Rscript", RSCRIPT, f"{td}/tr.csv", f"{td}/te.csv", f"{td}/out.csv"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("mgcv failed: " + r.stderr[-300:])
        return np.loadtxt(f"{td}/out.csv", delimiter=",", skiprows=1).astype(int)


def eval_series_faithful(X, y):
    classes = np.unique(y)
    corr = {"kunchenko": [], "ghosh_mdgam": [], "qda": []}
    for tri, tei in B.purged_blocks(len(y), 6, B.EMBARGO):
        ytr, yte = y[tri], y[tei]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue
        sc = G3.StandardScaler().fit(X[tri])
        Xtr, Xte = sc.transform(X[tri]), sc.transform(X[tei])
        corr["kunchenko"].append((B.predict_head(Xtr, ytr, Xte, "kunchenko", classes) == yte).astype(float))
        corr["qda"].append((B.predict_head(Xtr, ytr, Xte, "qda", classes) == yte).astype(float))
        mu, Si = G3.class_stats(Xtr, ytr, classes)
        Dtr = np.sqrt(np.clip(G3.global_md(Xtr, mu, Si, classes), 0, None))
        Dte = np.sqrt(np.clip(G3.global_md(Xte, mu, Si, classes), 0, None))
        corr["ghosh_mdgam"].append((mgcv_preds(Dtr, ytr, Dte) == yte).astype(float))
    corr = {h: np.concatenate(v) for h, v in corr.items()}
    rng = np.random.default_rng(B.SEED)
    out = {"acc": {h: round(float(np.nanmean(corr[h])), 4) for h in corr}}
    for name, (a, b) in {"kun-ghosh_mdgam": ("kunchenko", "ghosh_mdgam"),
                         "kun-qda": ("kunchenko", "qda")}.items():
        lo, hi = B.moving_block_boot(corr[a] - corr[b], B.BLOCK, B.B, rng)
        out[name] = [round(lo, 4), round(hi, 4)]
    return out


def main():
    res = {"seed": B.SEED, "comparator": "faithful mgcv global MD-GAM (REML)", "series": {}}
    print(f"{'series':<10}{'exk':>7}  {'kun':>6}{'mdgam':>7}{'qda':>6}  "
          f"{'kun-ghosh_mdgam':>20}{'kun-qda':>20}")
    nworse = ties = 0
    for label, f, cls, heavy in SERIES:
        r = load_returns(f"{STORE}/{f}")
        X, y = build_fx(r)
        o = eval_series_faithful(X, y)
        res["series"][label] = {"heavy": heavy, **o}
        cg, cq = o["kun-ghosh_mdgam"], o["kun-qda"]
        print(f"{label:<10}{float(kurtosis(r)):>7.1f}  "
              f"{o['acc']['kunchenko']:>6.3f}{o['acc']['ghosh_mdgam']:>7.3f}{o['acc']['qda']:>6.3f}  "
              f"{str(cg):>20}{str(cq):>20}")
        if heavy:
            if cg[1] < 0:
                nworse += 1
            elif cg[0] <= 0 <= cg[1]:
                ties += 1
    res["summary"] = {"heavy_derived_sig_worse_than_faithful": nworse, "heavy_ties": ties}
    print(f"\nvs FAITHFUL mgcv MD-GAM: derived significantly WORSE on {nworse}/5 heavy series, "
          f"tie on {ties}/5")
    json.dump(res, open("ell_gate4c_mgcv_faithful_results.json", "w"), indent=2)
    print("wrote ell_gate4c_mgcv_faithful_results.json")


if __name__ == "__main__":
    main()
