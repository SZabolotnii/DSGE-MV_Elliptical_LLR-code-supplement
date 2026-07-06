#!/usr/bin/env python3
"""
Gate G-ELL-2 (extension) — the FAIR test for Ghosh's local Mahalanobis.

G-ELL (gate 1) used a single elliptical component per class, where a global Mahalanobis suffices
and Ghosh's local part only added variance (it lost). The honest caveat: that DGP favoured the
global link. This extension uses MULTIMODAL (mixture) classes arranged XOR-like so that:
  * both class GLOBAL means coincide (~0) -> global Mahalanobis is near-useless,
  * the discriminative structure is LOCAL -> Ghosh's local Mahalanobis SHOULD help.
Question: when local genuinely matters, does the Kunchenko fractional link on global+local
(kunchenko_hi) still match/beat the spline-GAM on global+local (ghosh_full)? And do the local
heads now beat the global-only heads (confirming the DGP is a fair local test)?

Reuses gate-1 heads/verifiers from run_ell_gate.py (fit_eval, radial_basis, local_maha2, boot_ci).
Pre-registered (extends SPEC-G-ELL.md). Seed family = 2026. 15 splits, bootstrap R=2000, d in {4,10}.
PASS (extension): (i) local heads > global-only here (DGP is a real local test); (ii) kunchenko_hi
>= ghosh_full (fractional link competitive WHEN local matters); else honest boundary.
"""
import json, time, warnings
import numpy as np
from scipy.stats import multivariate_t
from scipy.special import logsumexp
import run_ell_gate as G

warnings.filterwarnings("ignore")
SEED = 2026
NU = 4.0


def rng_of(s): return np.random.default_rng(s)

def mixture_classes(d, rng, grid=4, a=1.6, scale=0.18):
    """Fine GxG checkerboard in the first 2 dims: blob at cell (i,j), label = (i+j) mod 2,
    ISOTROPIC same-scale blobs. For even G both classes have G²/2 blobs arranged symmetrically →
    IDENTICAL global mean (0) AND global covariance → a single-Gaussian global Mahalanobis head is
    provably uninformative (≈ chance); ONLY local cluster structure separates the classes. This is
    the decisive fair test of Ghosh's local Mahalanobis (its home turf)."""
    off = (grid - 1) / 2.0
    S = np.eye(d) * scale
    comps0, comps1 = [], []
    for i in range(grid):
        for j in range(grid):
            c = np.zeros(d); c[0] = (i - off) * a; c[1] = (j - off) * a
            (comps0 if (i + j) % 2 == 0 else comps1).append([1.0, c, S])
    for comps in (comps0, comps1):
        wsum = sum(w for w, _, _ in comps)
        for t in comps: t[0] /= wsum
    return [tuple(t) for t in comps0], [tuple(t) for t in comps1]

def sample_mixture(comps, n, rng):
    ks = rng.multinomial(n, [w for w, _, _ in comps])
    parts = [multivariate_t(mu, S, df=NU).rvs(k, random_state=rng)
             for (w, mu, S), k in zip(comps, ks) if k > 0]
    X = np.vstack([p.reshape(-1, comps[0][1].shape[0]) for p in parts])
    rng.shuffle(X)
    return X

def mix_logpdf(comps, X):
    L = np.column_stack([np.log(w) + multivariate_t(mu, S, df=NU).logpdf(X) for w, mu, S in comps])
    return logsumexp(L, axis=1)

def mix_oracle_acc(Xte, yte, comps0, comps1):
    Lam = mix_logpdf(comps1, Xte) - mix_logpdf(comps0, Xte)
    return float(((Lam > 0).astype(int) == yte).mean())


def run(d, n_tr=500, n_te=500, n_splits=15, k=None):
    k = k or max(50, 10 * d)
    HEADS = ["identity_qda", "kunchenko", "ghosh_global", "kunchenko_hi", "ghosh_full"]
    accs = {h: [] for h in HEADS}; accs["oracle"] = []; budgets = {}
    for s in range(n_splits):
        rng = rng_of(SEED + 211 * (s + 1))
        comps0, comps1 = mixture_classes(d, rng)
        Xtr = np.vstack([sample_mixture(comps0, n_tr, rng), sample_mixture(comps1, n_tr, rng)])
        ytr = np.r_[np.zeros(n_tr), np.ones(n_tr)].astype(int)
        Xte = np.vstack([sample_mixture(comps0, n_te, rng), sample_mixture(comps1, n_te, rng)])
        yte = np.r_[np.zeros(n_te), np.ones(n_te)].astype(int)
        for h in HEADS:
            a, b = G.fit_eval(Xtr, ytr, Xte, yte, h, m=3, k=k)
            accs[h].append(a); budgets[h] = b
        accs["oracle"].append(mix_oracle_acc(Xte, yte, comps0, comps1))
    return accs, budgets


if __name__ == "__main__":
    t0 = time.time(); out = {"seed": SEED, "dgp": "xor-multimodal-t4", "by_d": {}}
    for d in [4, 10]:
        print("=" * 84)
        print(f"G-ELL-2 extension — XOR multimodal t(4), d={d}, 15 splits, leakage-safe, equal budget")
        print("=" * 84)
        accs, budgets = run(d)
        ci = {
            "kunchenko_hi - ghosh_full (local, equal budget)": G.boot_ci(accs["kunchenko_hi"], accs["ghosh_full"]),
            "ghosh_full - ghosh_global (does local help?)": G.boot_ci(accs["ghosh_full"], accs["ghosh_global"]),
            "kunchenko_hi - kunchenko (does local help frac?)": G.boot_ci(accs["kunchenko_hi"], accs["kunchenko"]),
        }
        out["by_d"][f"d{d}"] = {"acc_mean": {h: float(np.mean(v)) for h, v in accs.items()},
                                "acc_sd": {h: float(np.std(v)) for h, v in accs.items()},
                                "budgets": budgets, "ci": ci}
        print(f"{'head':<16}{'budget':>7}{'acc_mean':>10}{'acc_sd':>9}")
        for h in ["identity_qda", "kunchenko", "ghosh_global", "kunchenko_hi", "ghosh_full", "oracle"]:
            print(f"{h:<16}{str(budgets.get(h,'-')):>7}{np.mean(accs[h]):>10.4f}{np.std(accs[h]):>9.4f}")
        print("Paired bootstrap 95% CI (>0 favours first):")
        for kk, vv in ci.items():
            print(f"   {kk:<48} {vv}")
    out["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(out, open("ell_gate2_results.json", "w"), indent=2)
    print(f"\nelapsed {out['elapsed_sec']}s -> ell_gate2_results.json")
