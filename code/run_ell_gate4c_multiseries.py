"""G-ELL-4C  — derived>fitted across MANY heavy-tailed real series (temporal-robust).

Turns the single-FX heavy-tail result into a pattern across asset classes (FX,
commodity, equity), answering the "n=1 / all-FX" reviewer concern. Each series is
validated with the same leakage-free scheme as gate G-ELL-4B: purged-and-embargoed
blocked 6-fold + moving-block bootstrap (block=vol_win) on pooled out-of-sample
paired correctness. Same task as gate4: d=5 return embeddings, binary
top-vs-bottom volatility-regime tercile label (21-day trailing realized vol).

Reports per series: n, excess kurtosis, OOS accuracy per head, and the three
temporal-robust CIs (derived−fitted, derived−identity, derived−QDA). A light-tailed
FX control (EUR/USD) is included to show the advantage tracks tail-heaviness.
Heads reuse the faithful run_ell_gate3 internals. Seed 2026.
"""
import json, os
import numpy as np
from scipy.stats import kurtosis
from sklearn.preprocessing import StandardScaler
import run_ell_gate4b_blocked as B

STORE = os.environ.get("GELL_FRED_DATA_DIR", "data/fred-heavytail")
VOL_WIN, D, K_FOLDS = 21, 5, 6
SERIES = [   # (label, file, asset class, heavy?)
    ("CAD/USD", "DEXCAUS.csv",  "FX",        True),
    ("JPY/USD", "DEXJPUS.csv",  "FX",        True),
    ("GBP/USD", "DEXUSUK.csv",  "FX",        True),
    ("WTI oil", "DCOILWTICO.csv","commodity", True),
    ("S&P 500", "SP500.csv",    "equity",    True),
    ("EUR/USD", "DEXUSEU.csv",  "FX",        False),  # light-tailed control
]


def load_returns(path):
    v = np.genfromtxt(path, delimiter=",", skip_header=1, usecols=1)
    v = v[np.isfinite(v)]; v = v[v > 0]
    return np.diff(np.log(v))


def build_fx(r, d=D, vol_win=VOL_WIN):
    N = len(r)
    rv = np.array([r[max(0, i - vol_win):i].std() for i in range(N)])
    X, lab = [], []
    for i in range(vol_win, N - d):
        X.append(r[i:i + d]); lab.append(rv[i])
    X = np.asarray(X); lab = np.asarray(lab)
    lo, hi = np.quantile(lab, [1 / 3, 2 / 3])
    keep = (lab <= lo) | (lab >= hi)
    return X[keep], (lab[keep] >= hi).astype(int)


def eval_series(path):
    r = load_returns(f"{STORE}/{path}")
    exk = float(kurtosis(r, fisher=True))
    Xk, yk = build_fx(r)
    classes = np.unique(yk)
    res = B.evaluate(Xk, yk, classes, B.purged_blocks(len(yk), K_FOLDS, B.EMBARGO),
                     f"purged blocked {K_FOLDS}-fold")
    res["excess_kurtosis"] = round(exk, 2)
    res["n_pairs"] = int(len(yk))
    return res


def sig(ci):
    return "+" if ci[0] > 0 else ("-" if ci[1] < 0 else "0")  # + favours derived, 0 tie


def main():
    out = {"seed": B.SEED, "scheme": "purged-and-embargoed blocked 6-fold + moving-block bootstrap",
           "d": D, "vol_win": VOL_WIN, "series": {}}
    print(f"{'series':<10}{'class':<10}{'n':>7}{'exk':>7}  "
          f"{'kun':>6}{'ghosh':>6}{'qda':>6}  {'kun-ghosh':>22}{'kun-id':>22}{'kun-qda':>22}")
    win_fit = win_id = tie_qda = nheavy = 0
    for label, f, cls, heavy in SERIES:
        r = eval_series(f)
        out["series"][label] = {"asset_class": cls, "heavy": heavy, **r}
        acc, ci = r["acc_oos"], r["ci"]
        cg, cid, cq = ci["kunchenko - ghosh_glob"]["ci"], ci["kunchenko - identity_md"]["ci"], ci["kunchenko - qda"]["ci"]
        print(f"{label:<10}{cls:<10}{r['n_pairs']:>7}{r['excess_kurtosis']:>7.1f}  "
              f"{acc['kunchenko']:>6.3f}{acc['ghosh_glob']:>6.3f}{acc['qda']:>6.3f}  "
              f"{str(cg):>22}{str(cid):>22}{str(cq):>22}")
        if heavy:
            nheavy += 1
            win_fit += sig(cg) == "+"; win_id += sig(cid) == "+"; tie_qda += sig(cq) == "0"
    out["summary"] = {"n_heavy": nheavy, "derived_beats_fitted": win_fit,
                      "derived_beats_identity": win_id, "ties_qda": tie_qda}
    print(f"\nHEAVY-TAILED ({nheavy} series): derived>fitted in {win_fit}/{nheavy}, "
          f">identity in {win_id}/{nheavy}, ties QDA in {tie_qda}/{nheavy}")
    json.dump(out, open("ell_gate4c_results.json", "w"), indent=2)
    print("wrote ell_gate4c_results.json")


if __name__ == "__main__":
    main()
