#!/usr/bin/env python3
"""
Gate G-ELL-5 — RATE SIMULATION verifying the theory core (THEORY-CORE.md) against a KNOWN
ground-truth elliptical model (R*, the true link φ, and the true LLR Λ all computable exactly).

Ground truth: two-class elliptical Student-t, X|0 ~ t_ν(0, I), X|1 ~ t_ν(μ1, a·I) in ℝ^d, equal
priors. True radial link φ_ν(t) = −((ν+d)/2)·log(1+t/ν); D_c² uses the class's own scale;
Λ(x)=φ(D1²)−φ(D0²)−½·d·log(a); Bayes rule sign(Λ); Bayes risk R* on a huge fixed test set.
The scale ratio a = |Σ1|^{1/d}/|Σ0|^{1/d} is the **covariance-contrast** knob.

(A) THEOREM 3 (√n-consistency) — IN SCOPE at ν=8 (>6 ⇒ full M=3 basis {1,0.5,1.5} has finite 2nd
    moments, A1; ν>4 ⇒ A3). Equal scale (a=1). TRUE whitening + fixed scaler ⇒ β̂_n → β* at n^{-1/2};
    check log-log slope ≈ −0.5.

(B) THEOREM 5 / excess risk — IN SCOPE at ν=8, with covariance contrast a=3 (where the radial-link
    curvature bites). Full ESTIMATED pipeline; excess risk vs n. Prediction: textbook QDA (affine
    link) plateaus at a CONSTANT positive approximation gap; the derived link (m=3) drops to its
    sieve floor.

(C) The two LEVERS of the QDA gap (refined Corollary 6 + Heuristic 6′), both at fixed n, d, IN SCOPE
    where noted: C1 covariance-contrast sweep a∈{1,2,3,4,6} at ν=8; C2 tail sweep ν∈{12,8,6,5,4,3}
    at a=3. Shows the gap is ~0 for equal Σ (a=1), opens with covariance heterogeneity, and grows as
    the tail heavies (ν↓) — the mechanism behind G-ELL-4 and the real-data gates.

Seed 2026. Heads reuse run_ell_gate{,3}.
"""
import json, time, warnings
import numpy as np
from scipy.stats import multivariate_t
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
import run_ell_gate as G
import run_ell_gate3 as G3

warnings.filterwarnings("ignore")
SEED = 2026
D = 10
SEP = 1.5
N_TEST = 40000
N_REF = 120000


def rng_of(s): return np.random.default_rng(s)

def mu1_of(d): return np.full(d, SEP / np.sqrt(d))

def sample(n, d, nu, a, rng):
    n0 = n // 2; n1 = n - n0
    X0 = multivariate_t(np.zeros(d), np.eye(d), df=nu).rvs(n0, random_state=rng)
    X1 = multivariate_t(mu1_of(d), a * np.eye(d), df=nu).rvs(n1, random_state=rng)
    X = np.vstack([X0, X1]); y = np.concatenate([np.zeros(n0), np.ones(n1)]).astype(int)
    return X, y

def phi(t, nu, d): return -((nu + d) / 2.0) * np.log1p(t / nu)

def true_llr(X, d, nu, a):
    d0 = (X ** 2).sum(1); d1 = ((X - mu1_of(d)) ** 2).sum(1) / a
    return phi(d1, nu, d) - phi(d0, nu, d) - 0.5 * d * np.log(a)

def true_radial_feats(X, d, a, m):
    d0 = (X ** 2).sum(1); d1 = ((X - mu1_of(d)) ** 2).sum(1) / a
    return np.hstack([G.radial_basis(d0, m), G.radial_basis(d1, m)])

def bayes(d, nu, a, rng):
    Xte, yte = sample(N_TEST, d, nu, a, rng)
    bp = (true_llr(Xte, d, nu, a) > 0).astype(int)
    return Xte, yte, float((bp != yte).mean())

def fit_kun(Xtr, ytr, Xte, m):                 # full ESTIMATED pipeline (μ̂,Σ̂ from train)
    cl = np.array([0, 1]); mu, Si = G3.class_stats(Xtr, ytr, cl)
    Dtr = G3.global_md(Xtr, mu, Si, cl); Dte = G3.global_md(Xte, mu, Si, cl)
    Ftr = np.hstack([G.radial_basis(Dtr[:, j], m) for j in range(2)])
    Fte = np.hstack([G.radial_basis(Dte[:, j], m) for j in range(2)])
    sc = StandardScaler().fit(Ftr)
    return LogisticRegression(max_iter=5000, C=10.0).fit(sc.transform(Ftr), ytr).predict(sc.transform(Fte))

def excess(d, nu, a, n, reps=10, seed0=0):
    Xte, yte, R = bayes(d, nu, a, rng_of(SEED + 700 + seed0))
    qa, ka = [], []
    for r in range(reps):
        Xtr, ytr = sample(n, d, nu, a, rng_of(SEED + 5 * r + seed0 + n))
        qa.append(float((QDA(reg_param=0.05).fit(Xtr, ytr).predict(Xte) != yte).mean()))
        ka.append(float((fit_kun(Xtr, ytr, Xte, 3) != yte).mean()))
    return R, float(np.mean(qa)) - R, float(np.mean(ka)) - R

# ---------- (A) sqrt(n) rate, nu=8, a=1, true whitening ----------
# Low dim (d_A=4) for a clean asymptotic rate: with d=10 the small-n logistic fit is noisy and the
# log-log slope is inflated by a pre-asymptotic transient. The rate claim concerns the estimator
# (Thm 3), independent of the gap demonstration (B/C, which need d, a contrast).
def part_A(nu=8.0, a=1.0, d_A=4, m=3, reps=16, C=1e4):
    rng = rng_of(SEED)
    Xr, yr = sample(N_REF, d_A, nu, a, rng); Fr = true_radial_feats(Xr, d_A, a, m)
    scaler = StandardScaler().fit(Fr)
    beta = LogisticRegression(max_iter=5000, C=C).fit(scaler.transform(Fr), yr).coef_.ravel()
    ns = [1000, 2000, 4000, 8000, 16000, 32000]; errs = []
    for n in ns:
        e = []
        for r in range(reps):
            X, y = sample(n, d_A, nu, a, rng_of(SEED + 1000 + 7 * r + n))
            b = LogisticRegression(max_iter=5000, C=C).fit(scaler.transform(true_radial_feats(X, d_A, a, m)), y).coef_.ravel()
            e.append(float(np.linalg.norm(b - beta)))
        errs.append(float(np.mean(e)))
    slope, _ = np.polyfit(np.log(ns), np.log(errs), 1)
    return {"nu": nu, "a": a, "d": d_A, "ns": ns, "errs": [round(x, 4) for x in errs],
            "loglog_slope": round(float(slope), 3), "predicted": -0.5, "beta_dim": int(beta.size)}


if __name__ == "__main__":
    t0 = time.time(); out = {"seed": SEED, "d": D, "sep": SEP}
    print("=" * 80); print("(A) THEOREM 3 — sqrt(n) rate of the radial logistic M-estimator (d=4, nu=8, a=1, IN SCOPE)"); print("=" * 80)
    A = part_A(); out["theorem3_rate"] = A
    print(f"  n:        {A['ns']}")
    print(f"  ||b-b*||: {A['errs']}")
    print(f"  log-log slope = {A['loglog_slope']}  (theory predicts -0.5; beta_dim={A['beta_dim']})")

    print("\n" + "=" * 80); print("(B) THEOREM 5 — excess risk vs n: QDA plateau vs derived link (nu=8, a=3, IN SCOPE)"); print("=" * 80)
    ns = [400, 800, 1600, 3200, 6400, 12800]; eq, ek = [], []; R0 = None
    for n in ns:
        R, q, k = excess(D, 8.0, 3.0, n, reps=10, seed0=11); R0 = R; eq.append(round(q, 4)); ek.append(round(k, 4))
    out["theorem5_excess"] = {"nu": 8.0, "a": 3.0, "ns": ns, "R_star": round(R0, 4),
                              "excess_qda": eq, "excess_kun_m3": ek}
    print(f"  Bayes risk R* = {round(R0,4)}   (nu=8, a=3)")
    print(f"  n:              {ns}")
    print(f"  excess[   qda]: {eq}")
    print(f"  excess[kun_m3]: {ek}")

    print("\n" + "=" * 80); print("(C) THE TWO LEVERS of the QDA gap (refined Cor 6 / Heuristic 6')"); print("=" * 80)
    print("  C1 covariance-contrast sweep (nu=8 IN SCOPE, fixed n=6400):")
    print(f"  {'a':>5}{'R*':>9}{'exc_qda':>10}{'exc_kun':>10}{'gap':>9}")
    c1 = {}
    for a in [1.0, 2.0, 3.0, 4.0, 6.0]:
        R, q, k = excess(D, 8.0, a, 6400, reps=10, seed0=int(a * 13))
        c1[str(a)] = {"R_star": round(R, 4), "exc_qda": round(q, 4), "exc_kun": round(k, 4), "gap": round(q - k, 4)}
        print(f"  {a:>5}{round(R,4):>9}{round(q,4):>10}{round(k,4):>10}{round(q-k,4):>9}")
    print("  C2 tail sweep (a=3 covariance contrast, fixed n=6400; nu>=6 in scope, nu<6 illustrative):")
    print(f"  {'nu':>5}{'R*':>9}{'exc_qda':>10}{'exc_kun':>10}{'gap':>9}")
    c2 = {}
    for nu in [12.0, 8.0, 6.0, 5.0, 4.0, 3.0]:
        R, q, k = excess(D, nu, 3.0, 6400, reps=10, seed0=int(nu * 17))
        c2[str(nu)] = {"R_star": round(R, 4), "exc_qda": round(q, 4), "exc_kun": round(k, 4), "gap": round(q - k, 4)}
        print(f"  {nu:>5}{round(R,4):>9}{round(q,4):>10}{round(k,4):>10}{round(q-k,4):>9}")
    out["levers"] = {"cov_contrast_nu8": c1, "tail_sweep_a3": c2}

    print("\n" + "=" * 80); print("(D) THEOREM (approx) — sieve ablation: excess risk vs m (fixed n=6400, nu=3, a=3)"); print("=" * 80)
    Xte, yte, Rd = bayes(D, 3.0, 3.0, rng_of(SEED + 700 + 99))
    abl = {}
    print(f"  R*={round(Rd,4)};  {'m':>4}{'exc_kun':>10}")
    for m in [1, 2, 3, 4, 6]:
        ka = []
        for r in range(10):
            Xtr, ytr = sample(6400, D, 3.0, 3.0, rng_of(SEED + 5 * r + 99 + 6400))
            ka.append(float((fit_kun(Xtr, ytr, Xte, m) != yte).mean()))
        abl[str(m)] = round(float(np.mean(ka)) - Rd, 4)
        print(f"  {'':>4}{m:>4}{abl[str(m)]:>10}")
    out["sieve_ablation_nu3_a3"] = {"R_star": round(Rd, 4), "n": 6400, "excess_by_m": abl}

    out["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(out, open("ell_gate5_results.json", "w"), indent=2)
    print(f"\nelapsed {out['elapsed_sec']}s -> ell_gate5_results.json")
