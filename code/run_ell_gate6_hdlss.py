"""G-ELL-6  HDLSS d>>n probe.

Does the derived radial link survive when the dimension approaches/exceeds the
per-class sample size, with a ridge-regularised per-class scatter? This closes
the d>>n limitation flagged in the Discussion. In-scope tail: nu=8 (full-basis
moment scope nu>6), covariance contrast a=3 so the link matters. Heads share an
equal budget; the per-class scatter is ridge-regularised (RIDGE in class_stats).

Honest report: whatever the ordering is. We expect identity (QDA-in-radii) to
degrade relative to the derived link as d grows, because the link curvature is
exactly what the affine head cannot represent; if it does not, we say so.
"""
import json
import numpy as np
import run_ell_gate5_rates as G5
import run_ell_gate as G
import run_ell_gate3 as G3
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

NU, A, M = 8.0, 3.0, 3
NTR = 40          # per class
NTE = 1000
REPS = 12
DIMS = [20, 60, 120, 200]


def heads(Xtr, ytr, Xte, yte):
    cl = np.array([0, 1])
    mu, Si = G3.class_stats(Xtr, ytr, cl)          # ridge-regularised scatter
    Dtr = G3.global_md(Xtr, mu, Si, cl)
    Dte = G3.global_md(Xte, mu, Si, cl)
    feats = {
        "identity": (Dtr, Dte),
        "kunchenko": (
            np.hstack([G.radial_basis(Dtr[:, j], M) for j in range(2)]),
            np.hstack([G.radial_basis(Dte[:, j], M) for j in range(2)]),
        ),
        "ghosh": G3.spline(Dtr, Dte),
    }
    out = {}
    for name, (Ftr, Fte) in feats.items():
        sc = StandardScaler().fit(Ftr)
        clf = LogisticRegression(max_iter=3000, C=10).fit(sc.transform(Ftr), ytr)
        out[name] = float((clf.predict(sc.transform(Fte)) != yte).mean())
    return out


def main():
    rows = []
    for d in DIMS:
        Xte, yte = G5.sample(NTE, d, NU, A, G5.rng_of(7 + d))
        acc = {k: [] for k in ("identity", "kunchenko", "ghosh")}
        for r in range(REPS):
            Xtr, ytr = G5.sample(2 * NTR, d, NU, A, G5.rng_of(100 + r + d))
            h = heads(Xtr, ytr, Xte, yte)
            for k in acc:
                acc[k].append(1 - h[k])
        row = {"d": d, "ratio_d_over_n": round(d / NTR, 2),
               **{k: round(float(np.mean(v)), 4) for k, v in acc.items()},
               **{k + "_se": round(float(np.std(v) / np.sqrt(REPS)), 4) for k, v in acc.items()}}
        rows.append(row)
        print(f"d={d:>4} d/n={d/NTR:>4.1f}  identity={row['identity']:.3f}  "
              f"kunchenko={row['kunchenko']:.3f}  ghosh={row['ghosh']:.3f}", flush=True)
    out = {"setup": {"nu": NU, "a": A, "m": M, "ntr_per_class": NTR, "nte": NTE,
                     "reps": REPS, "dims": DIMS,
                     "scatter": "ridge-regularised per-class (G3.class_stats)"},
           "rows": rows}
    with open("ell_gate6_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote ell_gate6_results.json", flush=True)


if __name__ == "__main__":
    main()
