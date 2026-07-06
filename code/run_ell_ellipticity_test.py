"""Ellipticity battery for the G-ELL-4C real series (premortem H-002).

The whole radial-GAM Bayes story (Paper 5, THEORY-CORE) assumes each class is
ELLIPTICAL. The real-data wins (G-ELL-4C: FX / oil / equity d=5 return embeddings)
were reported without ever testing that assumption. This script tests it on the
*same* embeddings the classifier consumes, per class and pooled.

Three independent probes:
  (A) Mardia multivariate SKEWNESS  b_{1,d}.  Elliptical symmetry => population
      third-order skewness = 0, so a significant b_{1,d} is direct evidence AGAINST
      ellipticity (not merely against normality).  Stat n*b1/6 ~ chi2(df),
      df = d(d+1)(d+2)/6.                                   <-- the discriminating test
  (B) Mardia multivariate KURTOSIS  b_{2,d}.  Reported for context: heavy tails
      inflate this even for a *bona fide* elliptical-t, so a rejection here is
      EXPECTED and is NOT evidence against ellipticity.  z = (b2 - d(d+2))/sqrt(8d(d+2)/n).
  (C) Directional-kurtosis HOMOGENEITY.  A known characterization: for an elliptical
      law every 1-D projection of the whitened data has the SAME kurtosis.  We whiten
      by (mean, cov), take R random unit directions, compute each projection's excess
      kurtosis, and use spread = std over directions as the statistic.  The null band
      comes from a parametric bootstrap of a matched multivariate-t (df set so the mean
      projection excess-kurtosis matches, nu = 4 + 6/kbar), which IS elliptical -- so
      observed spread above the null 95th pct rejects ellipticity for a reason
      independent of skewness (anisotropic tails / tail dependence).

Seed 2026.  Uses the exact load_returns / build_fx from run_ell_gate4c_multiseries.
"""
import json
import numpy as np
from scipy.stats import kurtosis, chi2, norm
from run_ell_gate4c_multiseries import SERIES, STORE, load_returns, build_fx, D

RNG = np.random.default_rng(2026)
R_DIRS = 400        # random projection directions
B_BOOT = 300        # parametric-bootstrap replications for the null band


def _whiten(X):
    mu = X.mean(0)
    S = np.cov(X.T, bias=True)
    # symmetric inverse sqrt
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 1e-12, None)
    Winv = V @ np.diag(w ** -0.5) @ V.T
    return (X - mu) @ Winv, S


def mardia(X):
    """Mardia multivariate skewness/kurtosis with tests vs the MVN/elliptical nulls."""
    n, d = X.shape
    mu = X.mean(0)
    S = np.cov(X.T, bias=True)
    Sinv = np.linalg.pinv(S)
    Xc = X - mu
    G = Xc @ Sinv @ Xc.T                       # n x n Mahalanobis Gram
    b1 = (G ** 3).mean()                        # (1/n^2) sum_ij g_ij^3
    b2 = np.mean(np.diag(G) ** 2)               # (1/n) sum_i g_ii^2
    # skewness test (elliptical symmetry => b1 population = 0)
    df = d * (d + 1) * (d + 2) / 6
    A = n * b1 / 6.0
    p_skew = float(chi2.sf(A, df))
    # kurtosis test (heavy tails inflate b2 even under elliptical-t -> expected)
    z = (b2 - d * (d + 2)) / np.sqrt(8 * d * (d + 2) / n)
    p_kurt = float(2 * norm.sf(abs(z)))
    return dict(b1=float(b1), skew_stat=float(A), skew_df=float(df), p_skew=p_skew,
                b2=float(b2), kurt_z=float(z), p_kurt=p_kurt)


def dir_kurt_spread(Yw):
    """std of excess kurtosis over random unit directions of whitened data."""
    n, d = Yw.shape
    A = RNG.standard_normal((d, R_DIRS))
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    P = Yw @ A                                  # n x R_DIRS projections
    ks = kurtosis(P, axis=0, fisher=True)       # excess kurtosis per direction
    return float(np.std(ks)), float(np.mean(ks))


def elliptic_symmetry_test(X):
    """Probe C: directional-kurtosis homogeneity vs a matched elliptical-t null."""
    n, d = X.shape
    Yw, _ = _whiten(X)
    obs_spread, kbar = dir_kurt_spread(Yw)
    # matched elliptical null: multivariate-t with univariate excess kurtosis = kbar
    nu = 4.0 + 6.0 / kbar if kbar > 1e-6 else 50.0
    nu = float(np.clip(nu, 4.5, 200))
    null = np.empty(B_BOOT)
    for b in range(B_BOOT):
        z = RNG.standard_normal((n, d))
        g = RNG.chisquare(nu, size=n) / nu
        Yt = z / np.sqrt(g)[:, None]            # standard multivariate-t_nu (elliptical)
        Yt, _ = _whiten(Yt)
        null[b], _ = dir_kurt_spread(Yt)
    p = float((1 + np.sum(null >= obs_spread)) / (B_BOOT + 1))
    return dict(obs_spread=obs_spread, kbar=kbar, null_nu=nu,
                null_q95=float(np.quantile(null, 0.95)), p_ellipsym=p)


def verdict(mard, esym):
    reasons = []
    if mard["p_skew"] < 0.05:
        reasons.append("asymmetry (Mardia skew p<0.05)")
    if esym["p_ellipsym"] < 0.05:
        reasons.append("anisotropic tails (dir-kurt p<0.05)")
    return ("ELLIPTICITY REJECTED: " + " + ".join(reasons)) if reasons \
        else "elliptical plausible (no probe rejects at 5%)"


def analyze(name, X, y):
    out = {}
    groups = {"pooled": np.ones(len(y), bool), "class0": y == 0, "class1": y == 1}
    for gname, mask in groups.items():
        Xi = X[mask]
        if len(Xi) < 50:
            out[gname] = {"n": int(len(Xi)), "skip": "n<50"}
            continue
        m = mardia(Xi)
        e = elliptic_symmetry_test(Xi)
        out[gname] = {"n": int(len(Xi)), **m, **e, "verdict": verdict(m, e)}
    return out


def main():
    results = {"seed": 2026, "d": D, "R_dirs": R_DIRS, "B_boot": B_BOOT, "series": {}}
    hdr = f"{'series':<10}{'grp':<8}{'n':>6}  {'p_skew':>9}{'p_kurt':>9}{'p_ellip':>9}  verdict"
    print(hdr); print("-" * len(hdr))
    for label, f, cls, heavy in SERIES:
        r = load_returns(f"{STORE}/{f}")
        X, y = build_fx(r)
        res = analyze(label, X, y)
        results["series"][label] = {"asset_class": cls, "heavy": heavy, **res}
        for g in ("pooled", "class0", "class1"):
            d = res[g]
            if "skip" in d:
                print(f"{label:<10}{g:<8}{d['n']:>6}  {'--':>9}{'--':>9}{'--':>9}  (skip)")
                continue
            tag = "REJECT" if "REJECTED" in d["verdict"] else "ok"
            print(f"{label:<10}{g:<8}{d['n']:>6}  {d['p_skew']:>9.4f}{d['p_kurt']:>9.4f}"
                  f"{d['p_ellipsym']:>9.4f}  {tag}: {d['verdict'].split(': ',1)[-1]}")
        print()
    # summary: fraction of (series x group) rejecting ellipticity
    rej = tot = 0
    for label, sd in results["series"].items():
        for g in ("pooled", "class0", "class1"):
            if "skip" in sd[g]:
                continue
            tot += 1; rej += "REJECTED" in sd[g]["verdict"]
    results["summary"] = {"groups_tested": tot, "ellipticity_rejected": rej,
                          "reject_frac": round(rej / tot, 3) if tot else None}
    print(f"SUMMARY: ellipticity rejected in {rej}/{tot} (series x group) cells "
          f"= {results['summary']['reject_frac']:.0%}")
    json.dump(results, open("ell_ellipticity_results.json", "w"), indent=2)
    print("wrote ell_ellipticity_results.json")


if __name__ == "__main__":
    main()
