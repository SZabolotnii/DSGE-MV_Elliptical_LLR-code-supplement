#!/usr/bin/env python3
"""
Gate G-ELL-4 (pre-submission gate #3 = heavy-tailed data; + the adaptivity ν-sweep).

Closes the remaining empirical gap from G-ELL-3: show the DERIVED Kunchenko radial link beats
textbook Gaussian QDA WHERE THE THEORY PREDICTS — under heavy tails — and that the advantage is
*adaptive* (vanishes as tails → Gaussian). Two parts:

(A) ν-SWEEP on REAL covariance structure (the adaptivity curve; Review-LLM's #2 framing item).
    Per-class (μ_c, Σ_c) are estimated from a REAL dataset (breast_cancer); classes are then drawn as
    multivariate Student-t with those real second-order parameters and shrinking ν (ν→∞ = Gaussian).
    Covariance held ≈ Σ_c across ν (shape = Σ·(ν−2)/ν) so only the TAIL changes — an apples-to-apples
    Gaussian-vs-heavy comparison on real-data structure. Expect: derived link ≈ QDA at ν→∞, and
    increasingly beats QDA as ν↓. Turns regime-dependence into adaptivity.

(B) GENUINELY-REAL heavy-tailed financial data: CAD/USD daily FX (FRED DEXCAUS, 1971–, ~14k obs;
    daily FX log-returns are a textbook heavy-tailed elliptical-t). d-dim return embeddings, 2-class
    volatility-regime label (top vs bottom tercile of a rolling realized-vol window; middle dropped),
    repeated CV, bootstrap. Tests whether the derived t-link beats Gaussian QDA on real heavy tails.

Heads reuse run_ell_gate3 (faithful). Seed 2026.
"""
import json, os, time, warnings
import numpy as np
from scipy.stats import multivariate_t, multivariate_normal
import run_ell_gate as G
import run_ell_gate3 as G3
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn import datasets as SKD

warnings.filterwarnings("ignore")
SEED = 2026
DEXCAUS = os.environ.get("GELL_DEXCAUS_CSV", "data/DEXCAUS.csv")


def rng_of(s): return np.random.default_rng(s)

# ---------- (A) nu-sweep on real covariance ----------
def real_class_params(name="breast_cancer"):
    b = SKD.load_breast_cancer() if name == "breast_cancer" else SKD.load_wine()
    X = StandardScaler().fit_transform(np.asarray(b.data, float)); y = np.asarray(b.target)
    classes = np.unique(y); P = {}
    for c in classes:
        Xc = X[y == c]; P[c] = (Xc.mean(0), np.cov(Xc, rowvar=False) + 1e-3 * np.eye(X.shape[1]))
    return classes, P

def draw(P, c, n, nu, rng):
    mu, S = P[c]
    if not np.isfinite(nu):
        return multivariate_normal(mu, S, allow_singular=True).rvs(n, random_state=rng)
    shape = S * (nu - 2.0) / nu   # so Cov(t) ~ S  (matched covariance; only tails differ)
    return multivariate_t(mu, shape, df=nu).rvs(n, random_state=rng)

def nu_sweep(n_tr=400, n_te=400, reps=10, shrink=0.5):
    classes, P = real_class_params("breast_cancer")
    # Place the experiment in the INFORMATIVE SNR regime: at full mean-separation the classes
    # saturate (~99% for every link, differences invisible). Shrink the between-class mean
    # separation toward the grand mean by `shrink` (covariance Σ_c kept REAL & untouched) so
    # Bayes error is ~10-15% and link quality drives a visible gap. Honest knob, documented.
    mus = np.array([P[c][0] for c in classes]); gm = mus.mean(0)
    P = {c: (gm + shrink * (P[c][0] - gm), P[c][1]) for c in classes}
    HEADS = ["qda", "identity_md", "kunchenko", "ghosh_glob"]
    grid = [np.inf, 16.0, 8.0, 5.0, 4.0, 3.0, 2.5]
    out = {}
    for nu in grid:
        accs = {h: [] for h in HEADS}
        for r in range(reps):
            rng = rng_of(SEED + 17 * (r + 1) + int(0 if not np.isfinite(nu) else nu * 10))
            Xtr = np.vstack([draw(P, c, n_tr, nu, rng) for c in classes])
            ytr = np.concatenate([np.full(n_tr, c) for c in classes]).astype(int)
            Xte = np.vstack([draw(P, c, n_te, nu, rng) for c in classes])
            yte = np.concatenate([np.full(n_te, c) for c in classes]).astype(int)
            sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            for h in HEADS:
                try: accs[h].append(G3.eval_head(Xtr, ytr, Xte, yte, h, classes, k=40))
                except Exception: accs[h].append(np.nan)
        key = "inf" if not np.isfinite(nu) else f"{nu}"
        out[key] = {h: float(np.nanmean(accs[h])) for h in HEADS}
        out[key]["ci_kun_minus_qda"] = G.boot_ci(accs["kunchenko"], accs["qda"])
    return out

# ---------- (B) real CAD/USD heavy-tailed ----------
def load_returns():
    v = np.genfromtxt(DEXCAUS, delimiter=",", skip_header=1, usecols=1)
    v = v[np.isfinite(v)]; v = v[v > 0]
    return np.diff(np.log(v))   # daily log-returns

def build_fx(d=5, vol_win=21):
    r = load_returns(); N = len(r)
    rv = np.array([r[max(0, i - vol_win):i].std() for i in range(N)])  # trailing realized vol
    X, lab = [], []
    for i in range(vol_win, N - d):
        X.append(r[i:i + d]); lab.append(rv[i])
    X = np.asarray(X); lab = np.asarray(lab)
    lo, hi = np.quantile(lab, [1 / 3, 2 / 3])
    keep = (lab <= lo) | (lab >= hi)
    Xk = X[keep]; yk = (lab[keep] >= hi).astype(int)   # 1 = high-vol regime, 0 = low-vol
    return Xk, yk

def fx_bench(d=5, n_splits=5, n_repeats=5):
    X, y = build_fx(d=d); classes = np.unique(y)
    X = StandardScaler().fit_transform(X)
    HEADS = ["qda", "identity_md", "kunchenko", "ghosh_glob"]
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    accs = {h: [] for h in HEADS}
    for tri, tei in rkf.split(X, y):
        Xtr, Xte = X[tri], X[tei]; ytr, yte = y[tri], y[tei]
        sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        for h in HEADS:
            try: accs[h].append(G3.eval_head(Xtr, ytr, Xte, yte, h, classes, k=40))
            except Exception: accs[h].append(np.nan)
    ci = {"kunchenko - qda": G.boot_ci(accs["kunchenko"], accs["qda"]),
          "kunchenko - ghosh_glob": G.boot_ci(accs["kunchenko"], accs["ghosh_glob"]),
          "kunchenko - identity_md": G.boot_ci(accs["kunchenko"], accs["identity_md"])}
    return {"n": int(len(y)), "d": int(d), "n_high": int(y.sum()),
            "acc_mean": {h: float(np.nanmean(accs[h])) for h in HEADS},
            "acc_sd": {h: float(np.nanstd(accs[h])) for h in HEADS}, "ci": ci}


if __name__ == "__main__":
    t0 = time.time(); out = {"seed": SEED}
    print("=" * 78); print("(A) nu-sweep on REAL breast_cancer covariance (adaptivity curve)"); print("=" * 78)
    A = nu_sweep(); out["nu_sweep"] = A
    print(f"{'nu':>6}{'qda':>9}{'identity':>9}{'kunchenko':>11}{'ghosh':>9}   CI(kun-qda)")
    for k, v in A.items():
        print(f"{k:>6}{v['qda']:>9.4f}{v['identity_md']:>9.4f}{v['kunchenko']:>11.4f}{v['ghosh_glob']:>9.4f}   {v['ci_kun_minus_qda']}")
    print("\n" + "=" * 78); print("(B) REAL CAD/USD daily FX, heavy-tailed, vol-regime classification"); print("=" * 78)
    B = fx_bench(); out["fx"] = B
    print(f"n={B['n']} d={B['d']} high-vol={B['n_high']} (5x5 CV)")
    print(f"{'head':<14}{'acc':>9}{'sd':>8}")
    for h in ["qda", "identity_md", "kunchenko", "ghosh_glob"]:
        print(f"{h:<14}{B['acc_mean'][h]:>9.4f}{B['acc_sd'][h]:>8.4f}")
    print("Paired bootstrap 95% CI (>0 favours first):")
    for k, v in B["ci"].items(): print(f"   {k:<26} {v}")
    out["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(out, open("ell_gate4_results.json", "w"), indent=2)
    print(f"\nelapsed {out['elapsed_sec']}s -> ell_gate4_results.json")
