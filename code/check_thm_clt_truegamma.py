import sys, json, numpy as np
sys.path.insert(0, sys.argv[1])
import check_thm_clt as C
rng = np.random.default_rng(20261108)
kind, powers, nu = C.CASES["T3_t10_p1_.5_1.5"]
pops = [C.population(kind, powers, nu, rng) for _ in range(2)]
bstar = np.mean([p["beta"] for p in pops], 0)
g0 = C.pack(C.MU, C.SIG)
for n in (4000, 16000):
    R = 500; est = np.empty((R, 2, len(bstar)))
    for r in range(R):
        X, y = C.sample(rng, n, kind, nu)
        est[r, 0] = C.logistic_fit(C.features(X, g0, powers), y, bstar)
        mu_h = [X[y == c].mean(0) for c in (0, 1)]; sig_h = [np.cov(X[y == c], rowvar=False, ddof=0) for c in (0, 1)]
        est[r, 1] = C.logistic_fit(C.features(X, C.pack(mu_h, sig_h), powers), y, bstar)
    for k, lab in enumerate(("true gamma", "estimated gamma")):
        dev = np.sqrt(n) * (est[:, k] - bstar); v = dev.var(0, ddof=1)
        print(f"n={n} {lab:16s} var/V_naive=" + str([[round(x, 2) for x in v / np.diag(p['V_naive'])] for p in pops]))
        print(f"        {'':16s} bias/sd=" + str([round(x, 2) for x in dev.mean(0) / np.sqrt(v)]))
