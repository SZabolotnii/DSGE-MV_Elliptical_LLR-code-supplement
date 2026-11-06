"""Numerical check of the first-stage correction in the proof of Theorem `thm:clt` (theory.tex:83).

Proof-audit 2026-11-05 (wiki/outputs/proof-audit-elliptical-llr-jmva-2026-11-05.md): both arms say
the proof differentiates only the trailing factor of the score s = (sigma(b'Z(x;g)) - y) Z(x;g) and
omits E[sigma'(b*'Z) Z b*' dZ/dg]; hence the proof's claim "under correct (Gaussian) specification
delta == 0 and the naive sandwich holds" is false.

We compare three asymptotic covariances of sqrt(n)(beta_hat - beta*) with the Monte Carlo truth:
  V_naive = H^-1 Var(s) H^-1                    (first stage ignored)
  V_proof = H^-1 Var(s + G_proof psi) H^-1      (G_proof = E[(sigma - y) dZ/dg], the proof's factor)
  V_full  = H^-1 Var(s + G_full psi) H^-1       (G_full = d/dg E[s(beta*, g)], both product-rule terms)
psi = influence function of the per-class sample mean and (MLE) covariance, g = (mu_c, vech Sigma_c).

Cases (d = 2, pi = 1/2, mu0 = 0, mu1 = (1, .5), Sigma0 = I, Sigma1 = [[2, .3], [.3, 1]]):
  G1  Gaussian, sieve p = (1,)            -- the proof's own "correct specification, m = 1" case
  G3  Gaussian, sieve p = (1, .5, 1.5)    -- the implemented working sieve, still correctly specified
  T3  Student-t nu = 10 (same means/covariances), sieve p = (1, .5, 1.5) -- misspecified, (A1) holds

Run:  python check_thm_clt.py   (seed 20261105; writes check_thm_clt.json beside it)
"""
import json
import pathlib

import numpy as np

SEED = 20261105
OUT = pathlib.Path(__file__).with_suffix(".json")
N_POP = 4_000_000   # sample standing in for the population
N = 4000            # Monte Carlo sample size
R = 2000            # Monte Carlo replications
FD_H = 1e-4

MU = [np.array([0.0, 0.0]), np.array([1.0, 0.5])]
SIG = [np.eye(2), np.array([[2.0, 0.3], [0.3, 1.0]])]
VECH = [(0, 0), (0, 1), (1, 1)]


def sample(rng, n, kind, nu=None):
    y = rng.binomial(1, 0.5, n)
    X = np.empty((n, 2))
    for c in (0, 1):
        idx = np.flatnonzero(y == c)
        L = np.linalg.cholesky(SIG[c])
        z = rng.standard_normal((idx.size, 2))
        if kind == "t":  # scatter chosen so that the covariance equals SIG[c]
            w = rng.chisquare(nu, idx.size) / nu
            z = z / np.sqrt(w)[:, None] * np.sqrt((nu - 2) / nu)
        X[idx] = MU[c] + z @ L.T
    return X, y


def pack(mu, sig):
    g = []
    for c in (0, 1):
        g += list(mu[c]) + [sig[c][i, j] for i, j in VECH]
    return np.array(g)


def unpack(g):
    mu, sig = [], []
    for c in (0, 1):
        o = 5 * c
        mu.append(g[o:o + 2])
        s = np.array([[g[o + 2], g[o + 3]], [g[o + 3], g[o + 4]]])
        sig.append(s)
    return mu, sig


def features(X, g, powers):
    mu, sig = unpack(g)
    cols = [np.ones(len(X))]
    for c in (0, 1):
        dx = X - mu[c]
        d2 = np.einsum("ij,jk,ik->i", dx, np.linalg.inv(sig[c]), dx)
        cols += [d2 ** p for p in powers]
    return np.column_stack(cols)


def sigm(u):
    return 0.5 * (1.0 + np.tanh(0.5 * u))


def logistic_loss(Z, y, b):
    u = Z @ b
    return float(np.sum(np.logaddexp(0.0, u) - y * u))


def logistic_fit(Z, y, b0=None, iters=200):
    """Unregularized logistic MLE: Newton with step-halving on the loss (lstsq for the step)."""
    b = np.zeros(Z.shape[1]) if b0 is None else b0.copy()
    loss = logistic_loss(Z, y, b)
    if not np.isfinite(loss):
        b = np.zeros(Z.shape[1])
        loss = logistic_loss(Z, y, b)
    for _ in range(iters):
        p = sigm(Z @ b)
        grad = Z.T @ (p - y)
        Hm = (Z * (p * (1 - p))[:, None]).T @ Z
        step = np.linalg.lstsq(Hm, grad, rcond=None)[0]
        t = 1.0
        while t > 1e-8:
            nb = b - t * step
            nl = logistic_loss(Z, y, nb)
            if np.isfinite(nl) and nl <= loss + 1e-12:
                break
            t *= 0.5
        b, loss = nb, nl
        if np.max(np.abs(t * step)) < 1e-10:
            break
    return b


def influence(X, y, g):
    """psi_i for (mu_c, vech Sigma_c), c = 0, 1; class proportion estimated -> 1{y=c}/pi_c."""
    mu, sig = unpack(g)
    psi = np.zeros((len(X), 10))
    for c in (0, 1):
        ind = (y == c).astype(float)
        pic = ind.mean()
        dx = X - mu[c]
        o = 5 * c
        psi[:, o:o + 2] = ind[:, None] * dx / pic
        for k, (i, j) in enumerate(VECH):
            psi[:, o + 2 + k] = ind * (dx[:, i] * dx[:, j] - sig[c][i, j]) / pic
    return psi


def population(kind, powers, nu, rng):
    X, y = sample(rng, N_POP, kind, nu)
    g0 = pack(MU, SIG)
    Z = features(X, g0, powers)
    b = logistic_fit(Z, y)
    p = sigm(Z @ b)
    H = (Z * (p * (1 - p))[:, None]).T @ Z / len(y)
    s = Z * (p - y)[:, None]
    G_full = np.zeros((Z.shape[1], 10))
    G_proof = np.zeros((Z.shape[1], 10))
    for k in range(10):
        e = np.zeros(10)
        e[k] = FD_H
        Zp, Zm = features(X, g0 + e, powers), features(X, g0 - e, powers)
        sp = Zp * (sigm(Zp @ b) - y)[:, None]
        sm = Zm * (sigm(Zm @ b) - y)[:, None]
        G_full[:, k] = (sp - sm).mean(0) / (2 * FD_H)
        G_proof[:, k] = (((Zp - Zm) / (2 * FD_H)) * (p - y)[:, None]).mean(0)
    psi = influence(X, y, g0)
    Hi = np.linalg.inv(H)

    def sandwich(G):
        xi = s + (psi @ G.T if G is not None else 0.0)
        return Hi @ np.cov(xi, rowvar=False) @ Hi

    return dict(beta=b, H=H, G_full=G_full, G_proof=G_proof,
                V_naive=sandwich(None), V_proof=sandwich(G_proof), V_full=sandwich(G_full))


def monte_carlo(kind, powers, nu, beta_star, rng):
    draws = np.empty((R, len(beta_star)))
    for r in range(R):
        X, y = sample(rng, N, kind, nu)
        mu_h = [X[y == c].mean(0) for c in (0, 1)]
        sig_h = [np.cov(X[y == c], rowvar=False, ddof=0) for c in (0, 1)]
        Z = features(X, pack(mu_h, sig_h), powers)
        draws[r] = logistic_fit(Z, y, beta_star)
    dev = np.sqrt(N) * (draws - beta_star)
    return dev.mean(0), np.cov(dev, rowvar=False)


CASES = {
    "G1_gauss_p1": ("gauss", (1.0,), None),
    "G3_gauss_p1_.5_1.5": ("gauss", (1.0, 0.5, 1.5), None),
    "T3_t10_p1_.5_1.5": ("t", (1.0, 0.5, 1.5), 10.0),
}


def main():
    rng = np.random.default_rng(SEED)
    res = {"seed": SEED, "N_pop": N_POP, "n": N, "R": R, "cases": {}}
    for name, (kind, powers, nu) in CASES.items():
        pop = population(kind, powers, nu, rng)
        bias, V_emp = monte_carlo(kind, powers, nu, pop["beta"], rng)
        d_emp = np.diag(V_emp)
        row = dict(
            beta_star=pop["beta"].tolist(),
            G_full_maxabs=float(np.max(np.abs(pop["G_full"]))),
            G_proof_maxabs=float(np.max(np.abs(pop["G_proof"]))),
            sqrt_n_bias=bias.tolist(),
            var_emp=d_emp.tolist(),
            ratio_emp_over_naive=(d_emp / np.diag(pop["V_naive"])).tolist(),
            ratio_emp_over_proof=(d_emp / np.diag(pop["V_proof"])).tolist(),
            ratio_emp_over_full=(d_emp / np.diag(pop["V_full"])).tolist(),
            mc_se_of_ratio=float(np.sqrt(2.0 / (R - 1))),
        )
        res["cases"][name] = row
        f = lambda v: "[" + ", ".join(f"{x:.2f}" for x in v) + "]"
        print(f"\n== {name}  beta*={f(row['beta_star'])}")
        print(f"   max|G_full|={row['G_full_maxabs']:.3f}   max|G_proof|={row['G_proof_maxabs']:.3f}")
        print(f"   Var_emp / V_naive = {f(row['ratio_emp_over_naive'])}")
        print(f"   Var_emp / V_proof = {f(row['ratio_emp_over_proof'])}")
        print(f"   Var_emp / V_full  = {f(row['ratio_emp_over_full'])}   (MC s.e. ≈ {row['mc_se_of_ratio']:.3f})")
    OUT.write_text(json.dumps(res, indent=1))
    print(f"\nwritten {OUT}")


if __name__ == "__main__":
    main()
