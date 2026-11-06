"""Diagnostic for case T3 (Student-t nu = 10, sieve p = (1, .5, 1.5)) after the n-sweep stayed flat.

Two questions:
  1. Is the "population" covariance V_full itself stable? Its plug-in estimate is a sample variance
     of s + G psi; s carries T^1.5, whose variance is finite (needs E T^3, nu > 6) but whose sample
     variance converges only as fast as E T^6 allows (infinite at nu = 10). We recompute beta*,
     V_naive and V_full on three independent population samples.
  2. Is the Monte Carlo spread dominated by a few outlying fits? We keep the draws at n = 16000 and
     compare the classical sd with a robust one (IQR / 1.349), and the mean with the median.

Run:  python check_thm_clt_diag.py   (seed 20261107; writes check_thm_clt_diag.json)
"""
import json
import pathlib

import numpy as np

import check_thm_clt as C

SEED = 20261107
OUT = pathlib.Path(__file__).with_suffix(".json")


def main():
    rng = np.random.default_rng(SEED)
    kind, powers, nu = C.CASES["T3_t10_p1_.5_1.5"]
    pops = [C.population(kind, powers, nu, rng) for _ in range(3)]
    f = lambda v: "[" + ", ".join(f"{x:.3g}" for x in v) + "]"
    out = {"seed": SEED, "N_pop": C.N_POP, "pops": []}
    for k, p in enumerate(pops):
        row = dict(beta=p["beta"].tolist(), V_full_diag=np.diag(p["V_full"]).tolist(),
                   V_naive_diag=np.diag(p["V_naive"]).tolist())
        out["pops"].append(row)
        print(f"pop {k}: beta*={f(row['beta'])}")
        print(f"        diag V_full={f(row['V_full_diag'])}")
    beta_star = np.mean([p["beta"] for p in pops], axis=0)

    n, R = 16000, 500
    draws = np.empty((R, len(beta_star)))
    for r in range(R):
        X, y = C.sample(rng, n, kind, nu)
        mu_h = [X[y == c].mean(0) for c in (0, 1)]
        sig_h = [np.cov(X[y == c], rowvar=False, ddof=0) for c in (0, 1)]
        Z = C.features(X, C.pack(mu_h, sig_h), powers)
        draws[r] = C.logistic_fit(Z, y, beta_star)
    dev = np.sqrt(n) * (draws - beta_star)
    sd = dev.std(0, ddof=1)
    q75, q25 = np.percentile(dev, [75, 25], axis=0)
    rsd = (q75 - q25) / 1.349
    out["mc"] = dict(n=n, R=R, sd=sd.tolist(), robust_sd=rsd.tolist(),
                     mean=dev.mean(0).tolist(), median=np.median(dev, 0).tolist(),
                     max_abs_z=(np.max(np.abs(dev - np.median(dev, 0)), 0) / rsd).tolist())
    print(f"MC n={n}: sd        ={f(sd)}")
    print(f"          robust sd ={f(rsd)}")
    print(f"          mean      ={f(dev.mean(0))}")
    print(f"          median    ={f(np.median(dev, 0))}")
    print(f"          max|z|    ={f(out['mc']['max_abs_z'])}")
    for k, p in enumerate(pops):
        print(f"  sd^2 / V_full(pop {k}) = {f(sd ** 2 / np.diag(p['V_full']))}")
    OUT.write_text(json.dumps(out, indent=1))
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
