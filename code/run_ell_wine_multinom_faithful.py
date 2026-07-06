"""Wine only: derived link vs the TRUE multinomial MD-GAM (not one-vs-rest), to
remove the OvR caveat from Table 2. Same 5x5 folds/seed as run_ell_gate3; the ghosh
head is the fixed-df multinomial GAM in mgcv_mdgam_multinom.R. Prints the paired CI.
"""
import os, subprocess, tempfile
import numpy as np
from sklearn import datasets as SKD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
import run_ell_gate as G
import run_ell_gate3 as G3
from run_ell_gate_mgcv_faithful import eval_simple, SEED

RMUL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mgcv_mdgam_multinom.R")


def multinom_acc(Dg_tr, ytr, Dg_te, yte):
    m, s = Dg_tr.mean(0), Dg_tr.std(0) + 1e-9
    Dg_tr, Dg_te = (Dg_tr - m) / s, (Dg_te - m) / s
    with tempfile.TemporaryDirectory() as td:
        cols = [f"d{j}" for j in range(Dg_tr.shape[1])]
        np.savetxt(f"{td}/tr.csv", np.column_stack([Dg_tr, ytr]), delimiter=",",
                   comments="", header=",".join(cols + ["y"]))
        np.savetxt(f"{td}/te.csv", Dg_te, delimiter=",", comments="", header=",".join(cols))
        r = subprocess.run(["Rscript", RMUL, f"{td}/tr.csv", f"{td}/te.csv", f"{td}/out.csv"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("mgcv multinom failed: " + r.stderr[-400:])
        pred = np.loadtxt(f"{td}/out.csv", delimiter=",", skiprows=1).astype(int)
    return float((pred == yte).mean())


def main():
    b = SKD.load_wine(); X = np.asarray(b.data, float); y = np.asarray(b.target)
    classes = np.unique(y)
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    accs = {h: [] for h in ("kunchenko", "qda", "ghosh_multinom")}
    for tri, tei in rkf.split(X, y):
        Xtr, Xte = X[tri], X[tei]; ytr, yte = y[tri], y[tei]
        sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        accs["kunchenko"].append(eval_simple(Xtr, ytr, Xte, yte, "kunchenko", classes))
        accs["qda"].append(eval_simple(Xtr, ytr, Xte, yte, "qda", classes))
        mu, Si = G3.class_stats(Xtr, ytr, classes)
        Dtr = np.sqrt(np.clip(G3.global_md(Xtr, mu, Si, classes), 0, None))
        Dte = np.sqrt(np.clip(G3.global_md(Xte, mu, Si, classes), 0, None))
        accs["ghosh_multinom"].append(multinom_acc(Dtr, ytr, Dte, yte))
    import json
    ci = G.boot_ci(accs["kunchenko"], accs["ghosh_multinom"])
    for h in accs:
        print(f"{h:<16}{np.mean(accs[h]):.4f}  sd {np.std(accs[h]):.4f}")
    print("kunchenko - ghosh_multinom (TRUE multinomial GAM):", ci)
    out = {"seed": SEED, "dataset": "wine", "comparator": "true multinomial GAM (mgcv, fixed-df spline)",
           "acc_mean": {h: round(float(np.mean(accs[h])), 4) for h in accs},
           "acc_sd": {h: round(float(np.std(accs[h])), 4) for h in accs},
           "ci_kunchenko_minus_ghosh_multinom": [round(float(ci[0]), 4), round(float(ci[1]), 4)]}
    json.dump(out, open("ell_wine_multinom_faithful_results.json", "w"), indent=2)
    print("wrote ell_wine_multinom_faithful_results.json")


if __name__ == "__main__":
    main()
