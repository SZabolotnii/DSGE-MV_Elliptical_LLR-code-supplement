"""n-sweep for the inconclusive heavy-tailed case T3 of check_thm_clt.py (Student-t nu = 10,
working sieve p = (1, .5, 1.5)). At n = 4000 the sqrt(n)-bias reached 0.55 sd and Var_emp was
0.33-0.82 of every asymptotic formula. If the ratio Var_emp / V_full climbs toward 1 and the
bias/sd falls as n grows, the discrepancy was finite-sample; if it stays flat, the formula is wrong.

Run:  python check_thm_clt_nsweep.py   (seed 20261106; writes check_thm_clt_nsweep.json)
"""
import json
import pathlib

import numpy as np

import check_thm_clt as C

SEED = 20261106
OUT = pathlib.Path(__file__).with_suffix(".json")
NS = (4000, 16000, 64000)
R = 500


def main():
    rng = np.random.default_rng(SEED)
    kind, powers, nu = C.CASES["T3_t10_p1_.5_1.5"]
    pop = C.population(kind, powers, nu, rng)
    res = {"seed": SEED, "N_pop": C.N_POP, "R": R, "beta_star": pop["beta"].tolist(), "by_n": {}}
    for n in NS:
        C.N, C.R = n, R
        bias, V_emp = C.monte_carlo(kind, powers, nu, pop["beta"], rng)
        d = np.diag(V_emp)
        row = dict(
            ratio_emp_over_full=(d / np.diag(pop["V_full"])).tolist(),
            ratio_emp_over_naive=(d / np.diag(pop["V_naive"])).tolist(),
            bias_over_sd=(bias / np.sqrt(d)).tolist(),
            mc_se_of_ratio=float(np.sqrt(2.0 / (R - 1))),
        )
        res["by_n"][str(n)] = row
        f = lambda v: "[" + ", ".join(f"{x:.2f}" for x in v) + "]"
        print(f"n={n:6d}  emp/full={f(row['ratio_emp_over_full'])}  bias/sd={f(row['bias_over_sd'])}", flush=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
