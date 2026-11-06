"""Numerical check of the two refutations of Corollary `cor:dichotomy` (theory.tex:189).

Proof-audit 2026-11-05 (wiki/outputs/proof-audit-elliptical-llr-jmva-2026-11-05.md): both arms
claim that a positive score gap ||Lambda_1^* - Lambda|| near the Bayes boundary does NOT force a
QDA excess-risk floor, because excess risk depends only on sign disagreement.

For each one-dimensional two-class model we compute, by quadrature on a dense sinh grid:
  * Lambda(x) = log(pi1 p1 / pi0 p0)                       (true log-odds, from the densities)
  * Lambda_1^*  = L2(P_x) projection of Lambda on span{1, D0^2, D1^2}   (the corollary's object)
  * gap      = ||Lambda_1^* - Lambda||_{L2(P_x)}
  * P(disagree) and excess risk  E|2 eta - 1| 1{sign Lambda_1^* != sign Lambda}
  * the same two numbers for population QDA (moment-matched Gaussian per class)
  * the mean excess risk of plug-in QDA fitted on n samples, n = 200 ... 20000.

Cases:
  A  astra's counterexample: generator exp(-t^2) (density ∝ exp(-((x-mu)/s)^4)), mu = ±1, s = 2.
  B  gemini's idea with finite moments: Student-t nu=8, mu = ±1, common scale 1.
     (gemini's Cauchy version is not admissible: under Cauchy x^2 is not in L2(P_x), so the
      projection Lambda_1^* does not exist; nu=8 keeps E x^4 finite.)
  C  contrast, the regime the corollary is meant for: Student-t nu=8, mu0=0, mu1=1, scales 1 vs 3.
  D  sanity: Gaussian, mu0=0, mu1=1, scales 1 vs 2 (gap must be 0, excess 0).

Run:  python check_cor_dichotomy.py   (seed 20261105; writes check_cor_dichotomy.json beside it)
"""
import json
import pathlib

import numpy as np
from scipy import special, stats

SEED = 20261105
OUT = pathlib.Path(__file__).with_suffix(".json")

# dense grid x = sinh(u): resolves the boundary near 0 and reaches |x| ~ 1.5e3 for heavy tails
U = np.linspace(-8.0, 8.0, 2_000_001)
X = np.sinh(U)
W = np.cosh(U) * (U[1] - U[0])  # dx weights (rectangle rule; the integrands are smooth or bounded)


def quartic_pdf(x, mu, s):
    return 2.0 / (s * special.gamma(0.25)) * np.exp(-(((x - mu) / s) ** 4))


def quartic_logpdf(x, mu, s):
    return np.log(2.0 / (s * special.gamma(0.25))) - ((x - mu) / s) ** 4


def t_pdf(x, mu, s, nu):
    return stats.t.pdf((x - mu) / s, nu) / s


def t_logpdf(x, mu, s, nu):
    return stats.t.logpdf((x - mu) / s, nu) - np.log(s)


def t_sample(rng, n, mu, s, nu):
    return mu + s * rng.standard_t(nu, n)


def quartic_sample(rng, n, mu, s):
    # |Z|^4 ~ Gamma(1/4, 1)  for density ∝ exp(-z^4)
    g = rng.gamma(0.25, 1.0, n)
    return mu + s * np.sign(rng.uniform(-1, 1, n)) * g ** 0.25


CASES = {
    "A_quartic_equal_scatter": dict(
        kind="quartic", mu=(-1.0, 1.0), s=(2.0, 2.0),
        note="astra counterexample: phi(t) = const - t^2, analytic, non-affine"),
    "B_t8_equal_scatter": dict(
        kind="t", nu=8.0, mu=(-1.0, 1.0), s=(1.0, 1.0),
        note="gemini counterexample with Cauchy replaced by t8 so that Lambda_1^* exists"),
    "C_t8_unequal_scatter": dict(
        kind="t", nu=8.0, mu=(0.0, 1.0), s=(1.0, 3.0),
        note="contrast: covariance heterogeneity (the regime the corollary targets)"),
    "D_gauss_unequal_scatter": dict(
        kind="gauss", mu=(0.0, 1.0), s=(1.0, 2.0),
        note="sanity: Gaussian, Lambda is affine in the radii"),
}


def densities(case):
    mu, s = case["mu"], case["s"]
    if case["kind"] == "quartic":
        lp = [quartic_logpdf(X, mu[c], s[c]) for c in (0, 1)]
    elif case["kind"] == "t":
        lp = [t_logpdf(X, mu[c], s[c], case["nu"]) for c in (0, 1)]
    else:
        lp = [stats.norm.logpdf(X, mu[c], s[c]) for c in (0, 1)]
    return lp


def sampler(case, rng, n_c, c):
    mu, s = case["mu"][c], case["s"][c]
    if case["kind"] == "quartic":
        return quartic_sample(rng, n_c, mu, s)
    if case["kind"] == "t":
        return t_sample(rng, n_c, mu, s, case["nu"])
    return rng.normal(mu, s, n_c)


def excess_and_disagree(score, lam, wdiff, px):
    dis = np.sign(score) != np.sign(lam)
    return float(np.sum(wdiff * dis * W)), float(np.sum(px * dis * W))


def run_case(name, case, rng):
    pi = (0.5, 0.5)
    lp0, lp1 = densities(case)
    p0, p1 = np.exp(lp0), np.exp(lp1)
    px = pi[0] * p0 + pi[1] * p1
    lam = (np.log(pi[1]) + lp1) - (np.log(pi[0]) + lp0)
    wdiff = np.abs(pi[1] * p1 - pi[0] * p0)  # = p_x |2 eta - 1|
    mass = float(np.sum(px * W))

    # L2(P_x) projection on span{1, D0^2, D1^2}; d=1, D_c^2 = ((x - mu_c)/s_c)^2
    D = [((X - case["mu"][c]) / case["s"][c]) ** 2 for c in (0, 1)]
    B = np.vstack([np.ones_like(X), D[0], D[1]])
    gram = (B * px * W) @ B.T
    rhs = (B * px * W) @ lam
    # equal scatter makes {1, D0^2, D1^2} = {1, x, x^2}: full rank; use lstsq for safety
    coef, *_ = np.linalg.lstsq(gram, rhs, rcond=None)
    proj = coef @ B
    gap = float(np.sqrt(np.sum((proj - lam) ** 2 * px * W)))
    ex_proj, dis_proj = excess_and_disagree(proj, lam, wdiff, px)

    # population QDA = moment-matched Gaussian per class
    m = [float(np.sum(X * np.exp(l) * W)) for l in (lp0, lp1)]
    v = [float(np.sum((X - m[c]) ** 2 * np.exp((lp0, lp1)[c]) * W)) for c in (0, 1)]
    qda_pop = (stats.norm.logpdf(X, m[1], np.sqrt(v[1])) - stats.norm.logpdf(X, m[0], np.sqrt(v[0]))
               + np.log(pi[1] / pi[0]))
    ex_qda, dis_qda = excess_and_disagree(qda_pop, lam, wdiff, px)

    # plug-in QDA on n samples
    plug = {}
    for n in (200, 1000, 5000, 20000):
        exs = []
        for _ in range(200):
            n1 = rng.binomial(n, pi[1])
            xs = [sampler(case, rng, n - n1, 0), sampler(case, rng, n1, 1)]
            mm = [xs[c].mean() for c in (0, 1)]
            vv = [xs[c].var() for c in (0, 1)]
            sc = (stats.norm.logpdf(X, mm[1], np.sqrt(vv[1])) - stats.norm.logpdf(X, mm[0], np.sqrt(vv[0]))
                  + np.log(n1 / (n - n1)))
            exs.append(excess_and_disagree(sc, lam, wdiff, px)[0])
        exs = np.asarray(exs)
        plug[str(n)] = dict(mean=float(exs.mean()), se=float(exs.std(ddof=1) / np.sqrt(len(exs))))

    return dict(
        note=case["note"], total_mass=mass,
        proj_coef=[float(c) for c in coef], score_gap_L2=gap,
        proj_P_disagree=dis_proj, proj_excess_risk=ex_proj,
        qda_pop_moments=dict(mean=m, var=v), qda_pop_P_disagree=dis_qda, qda_pop_excess_risk=ex_qda,
        qda_plugin_excess_risk=plug,
    )


def main():
    rng = np.random.default_rng(SEED)
    res = {"seed": SEED, "grid": "x = sinh(u), u in [-8,8], 2e6+1 points", "cases": {}}
    for name, case in CASES.items():
        r = run_case(name, case, rng)
        res["cases"][name] = r
        print(f"\n== {name}: {r['note']}")
        print(f"   mass={r['total_mass']:.6f}  gap||L1*-L||={r['score_gap_L2']:.4f}")
        print(f"   projection: P(disagree)={r['proj_P_disagree']:.2e}  excess={r['proj_excess_risk']:.2e}")
        print(f"   pop. QDA  : P(disagree)={r['qda_pop_P_disagree']:.2e}  excess={r['qda_pop_excess_risk']:.2e}")
        print("   plug-in QDA excess: " + "  ".join(
            f"n={n}: {v['mean']:.2e}±{v['se']:.1e}" for n, v in r["qda_plugin_excess_risk"].items()))
    OUT.write_text(json.dumps(res, indent=1))
    print(f"\nwritten {OUT}")


if __name__ == "__main__":
    main()
