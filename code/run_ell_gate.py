#!/usr/bin/env python3
"""
Gate G-ELL — does the ELLIPTICAL case make GSA<->Ghosh<->DSGE unification non-tautological?
Pre-registered criteria in SPEC-G-ELL.md. Seed family = 2026.

G-ELL-1: identity-link failure   (analytic R^2 of linear fit of true LLR on (D0^2,D1^2))
G-ELL-2: Kunchenko captured-fraction kappa_m of the elliptical LLR (fractional radial basis)
G-ELL-3: Kunchenko-head vs Ghosh-GAM benchmark (equal budget, leakage-safe, 15 splits, bootstrap)
"""
import json, time, warnings
import numpy as np
from numpy.linalg import inv, lstsq, slogdet
from scipy.stats import multivariate_t, multivariate_normal
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
SEED = 2026
RIDGE = 1e-6
RAD_POWERS = [1.0, 0.5, 1.5, 2.0, 2.5, 0.25]   # p=1 first -> Gaussian terminates at m=1


def rng_of(s): return np.random.default_rng(s)

def rand_spd(d, rng, cond_target=4.0):
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    eigs = np.linspace(1.0, cond_target, d)
    return (Q * eigs) @ Q.T

def make_classes(d, rng, sep=1.4):
    mu0 = np.zeros(d)
    mu1 = np.full(d, sep / np.sqrt(d))
    S0 = rand_spd(d, rng, 4.0)
    S1 = rand_spd(d, rng, 6.0) * 1.3
    return mu0, mu1, S0, S1

def maha2(X, mu, S):
    Si = inv(S + RIDGE * np.eye(S.shape[0])); Xc = X - mu
    return np.einsum("ni,ij,nj->n", Xc, Si, Xc)

def r2(y, yhat):
    yc = y - y.mean(); ss = (yc @ yc)
    return 1.0 - ((y - yhat) @ (y - yhat)) / ss if ss > 0 else 0.0

def lin_fit_r2(Lam, D0, D1):
    A = np.column_stack([np.ones_like(D0), D0, D1])
    beta, *_ = lstsq(A, Lam, rcond=None)
    return r2(Lam, A @ beta)

def radial_basis(D, m):
    """m fractional/PATP powers of a Mahalanobis radius (no bias). The Kunchenko radial link."""
    return np.column_stack([np.sign(D) * np.abs(D) ** p for p in RAD_POWERS[:m]])

def kappa_features(D0, D1, m):
    return np.column_stack([np.ones_like(D0), radial_basis(D0, m), radial_basis(D1, m)])

def kappa_m(Lam, D0, D1, m):
    F = kappa_features(D0, D1, m)
    beta, *_ = lstsq(F, Lam, rcond=None)
    return r2(Lam, F @ beta)


# ---------------- G-ELL-1 & G-ELL-2 (analytic, true params) ----------------
def gell_1_2(d=6, n=40000):
    rng = rng_of(SEED)
    mu0, mu1, S0, S1 = make_classes(d, rng)
    res = {"d": d, "n": n, "r2_lin": {}, "kappa": {}}
    # sample pooled support
    def pooled(sampler0, sampler1):
        X = np.vstack([sampler0(n // 2), sampler1(n - n // 2)])
        return X
    # ---- Gaussian control ----
    g0 = multivariate_normal(mu0, S0); g1 = multivariate_normal(mu1, S1)
    Xg = pooled(lambda k: g0.rvs(k, random_state=rng), lambda k: g1.rvs(k, random_state=rng))
    Lg = g1.logpdf(Xg) - g0.logpdf(Xg)
    D0g, D1g = maha2(Xg, mu0, S0), maha2(Xg, mu1, S1)
    res["r2_lin"]["gaussian"] = float(lin_fit_r2(Lg, D0g, D1g))
    res["kappa"]["gaussian"] = [float(kappa_m(Lg, D0g, D1g, m)) for m in range(1, 6)]
    # ---- Student-t sweep ----
    for nu in [30.0, 8.0, 4.0, 2.5]:
        t0 = multivariate_t(mu0, S0, df=nu); t1 = multivariate_t(mu1, S1, df=nu)
        Xt = pooled(lambda k: t0.rvs(k, random_state=rng), lambda k: t1.rvs(k, random_state=rng))
        Lt = t1.logpdf(Xt) - t0.logpdf(Xt)
        D0t, D1t = maha2(Xt, mu0, S0), maha2(Xt, mu1, S1)
        res["r2_lin"][f"t{nu}"] = float(lin_fit_r2(Lt, D0t, D1t))
        res["kappa"][f"t{nu}"] = [float(kappa_m(Lt, D0t, D1t, m)) for m in range(1, 6)]
    return res


# ---------------- G-ELL-3 (benchmark) ----------------
def local_maha2(Xq, Xtrain_c, k):
    nn = NearestNeighbors(n_neighbors=min(k, len(Xtrain_c))).fit(Xtrain_c)
    _, idx = nn.kneighbors(Xq)
    out = np.empty(len(Xq))
    d = Xq.shape[1]
    for i in range(len(Xq)):
        nb = Xtrain_c[idx[i]]; mu = nb.mean(0)
        C = np.cov(nb, rowvar=False) + RIDGE * np.eye(d)
        v = Xq[i] - mu
        out[i] = v @ inv(C) @ v
    return out

def features_global(X, mu0, S0, mu1, S1):
    return maha2(X, mu0, S0), maha2(X, mu1, S1)

def fit_eval(Xtr, ytr, Xte, yte, kind, m=3, k=40):
    mu0 = Xtr[ytr == 0].mean(0); mu1 = Xtr[ytr == 1].mean(0)
    S0 = np.cov(Xtr[ytr == 0], rowvar=False); S1 = np.cov(Xtr[ytr == 1], rowvar=False)
    D0tr, D1tr = features_global(Xtr, mu0, S0, mu1, S1)
    D0te, D1te = features_global(Xte, mu0, S0, mu1, S1)
    clf = LogisticRegression(max_iter=2000, C=10.0)
    # Ghosh GAM proxy: quantile-knot B-splines (fair on heavy tails), m functions per radius
    def spline(cols_tr, cols_te):
        sp = SplineTransformer(n_knots=3, degree=2, include_bias=False, knots="quantile")
        return sp.fit_transform(cols_tr), sp.transform(cols_te)
    if kind == "identity_qda":
        Ftr = np.column_stack([D0tr, D1tr]); Fte = np.column_stack([D0te, D1te])
    elif kind == "kunchenko":   # fractional radial link on global Mahalanobis
        Ftr = np.column_stack([radial_basis(D0tr, m), radial_basis(D1tr, m)])
        Fte = np.column_stack([radial_basis(D0te, m), radial_basis(D1te, m)])
    elif kind == "ghosh_global":  # spline link on global Mahalanobis (equal budget to kunchenko)
        Ftr, Fte = spline(np.column_stack([D0tr, D1tr]), np.column_stack([D0te, D1te]))
    elif kind in ("ghosh_full", "kunchenko_hi"):
        L0tr = local_maha2(Xtr, Xtr[ytr == 0], k); L1tr = local_maha2(Xtr, Xtr[ytr == 1], k)
        L0te = local_maha2(Xte, Xtr[ytr == 0], k); L1te = local_maha2(Xte, Xtr[ytr == 1], k)
        if kind == "ghosh_full":  # spline on global + local Mahalanobis (the actual Ghosh method)
            Ftr, Fte = spline(np.column_stack([D0tr, D1tr, L0tr, L1tr]),
                              np.column_stack([D0te, D1te, L0te, L1te]))
        else:  # kunchenko_hi: fractional link on global+local (equal budget to ghosh_full)
            Ftr = np.column_stack([radial_basis(D0tr, m), radial_basis(D1tr, m),
                                   radial_basis(L0tr, m), radial_basis(L1tr, m)])
            Fte = np.column_stack([radial_basis(D0te, m), radial_basis(D1te, m),
                                   radial_basis(L0te, m), radial_basis(L1te, m)])
    else:
        raise ValueError(kind)
    clf.fit(Ftr, ytr)
    return float((clf.predict(Fte) == yte).mean()), Ftr.shape[1]

def oracle_acc(Xte, yte, mu0, S0, mu1, S1, nu):
    t0 = multivariate_t(mu0, S0, df=nu); t1 = multivariate_t(mu1, S1, df=nu)
    Lam = t1.logpdf(Xte) - t0.logpdf(Xte)
    return float(((Lam > 0).astype(int) == yte).mean())

def gell_3(d=4, nu=4.0, n_tr=400, n_te=400, n_splits=15, k=40):
    HEADS = ["identity_qda", "kunchenko", "ghosh_global", "kunchenko_hi", "ghosh_full"]
    accs = {h: [] for h in HEADS}; accs["oracle"] = []; budgets = {}
    for s in range(n_splits):
        rng = rng_of(SEED + 101 * (s + 1))
        mu0, mu1, S0, S1 = make_classes(d, rng)
        t0 = multivariate_t(mu0, S0, df=nu); t1 = multivariate_t(mu1, S1, df=nu)
        Xtr = np.vstack([t0.rvs(n_tr, random_state=rng), t1.rvs(n_tr, random_state=rng)])
        ytr = np.r_[np.zeros(n_tr), np.ones(n_tr)].astype(int)
        Xte = np.vstack([t0.rvs(n_te, random_state=rng), t1.rvs(n_te, random_state=rng)])
        yte = np.r_[np.zeros(n_te), np.ones(n_te)].astype(int)
        for h in HEADS:
            a, b = fit_eval(Xtr, ytr, Xte, yte, h, m=3, k=k)
            accs[h].append(a); budgets[h] = b
        accs["oracle"].append(oracle_acc(Xte, yte, mu0, S0, mu1, S1, nu))
    return accs, budgets

def boot_ci(a, b, R=2000, seed=SEED):
    rng = rng_of(seed); a = np.asarray(a); b = np.asarray(b); n = len(a); d = []
    for _ in range(R):
        idx = rng.integers(0, n, n); d.append((a[idx] - b[idx]).mean())
    return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]


if __name__ == "__main__":
    t0 = time.time()
    out = {"seed": SEED}
    print("=" * 78); print("G-ELL-1/2  (analytic, true params)"); print("=" * 78)
    a12 = gell_1_2()
    out["gell12"] = a12
    print("R^2 of linear fit of true LLR on (D0^2,D1^2)  [identity link]:")
    for k, v in a12["r2_lin"].items():
        print(f"   {k:<10} R2_lin = {v:.4f}")
    print("Kunchenko captured-fraction kappa_m (fractional radial basis), m=1..5:")
    for k, v in a12["kappa"].items():
        print(f"   {k:<10} " + " ".join(f"{x:.3f}" for x in v))

    out["gell3"] = {}
    for d in [4, 10]:
        print("\n" + "=" * 78)
        print(f"G-ELL-3  benchmark (t(4), d={d}, 15 splits, leakage-safe, equal budget)")
        print("=" * 78)
        accs, budgets = gell_3(d=d, k=max(40, 8 * d))
        ci = {
            "kunchenko - ghosh_global (equal budget)": boot_ci(accs["kunchenko"], accs["ghosh_global"]),
            "kunchenko_hi - ghosh_full (equal budget)": boot_ci(accs["kunchenko_hi"], accs["ghosh_full"]),
            "kunchenko - identity_qda": boot_ci(accs["kunchenko"], accs["identity_qda"]),
        }
        out["gell3"][f"d{d}"] = {
            "acc_mean": {h: float(np.mean(v)) for h, v in accs.items()},
            "acc_sd": {h: float(np.std(v)) for h, v in accs.items()},
            "budgets": budgets, "ci": ci}
        print(f"{'head':<16}{'budget':>7}{'acc_mean':>10}{'acc_sd':>9}")
        for h in ["identity_qda", "kunchenko", "ghosh_global", "kunchenko_hi", "ghosh_full", "oracle"]:
            b = budgets.get(h, "-")
            print(f"{h:<16}{str(b):>7}{np.mean(accs[h]):>10.4f}{np.std(accs[h]):>9.4f}")
        print("Paired bootstrap 95% CI of accuracy difference (>0 favours first):")
        for kk, vv in ci.items():
            print(f"   {kk:<42} {vv}")
    out["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(out, open("ell_gate_results.json", "w"), indent=2)
    print(f"\nelapsed {out['elapsed_sec']}s -> ell_gate_results.json")
