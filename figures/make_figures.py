#!/usr/bin/env python3
"""Generate the two manuscript figures from the gate JSON results."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
GATE = os.path.abspath(os.path.join(HERE, "..", ".."))
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
CB = {"qda": "#D55E00", "kun": "#0072B2", "ghosh": "#009E73", "acc": "#0072B2"}

# ---------- Figure 1: adaptivity (gate4 nu-sweep) ----------
g4 = json.load(open(os.path.join(GATE, "ell_gate4_results.json")))
sw = g4["nu_sweep"]
order = ["inf", "16.0", "8.0", "5.0", "4.0", "3.0", "2.5"]
xs, mids, los, his = [], [], [], []
for k in order:
    if k not in sw:
        continue
    nu = 0.0 if k == "inf" else 1.0 / float(k)   # x = 1/nu (Gaussian at 0)
    lo, hi = sw[k]["ci_kun_minus_qda"]
    xs.append(nu); mids.append((lo + hi) / 2); los.append(lo); his.append(hi)
xs = np.array(xs); mids = np.array(mids)
fig, ax = plt.subplots(figsize=(4.0, 3.2))
ax.errorbar(xs, mids, yerr=[mids - np.array(los), np.array(his) - mids],
            fmt="o-", color=CB["acc"], capsize=3, lw=1.6, ms=5)
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel(r"$1/\nu$  (Gaussian $\to$ heavy-tailed)")
ax.set_ylabel(r"accuracy advantage  (kunchenko $-$ QDA)")
ax.set_title("Adaptivity on real covariance (G-ELL-4A)")
for k, nu in zip(order, xs):
    lab = r"$\infty$" if k == "inf" else k
    ax.annotate(lab, (nu, mids[list(xs).index(nu)]), textcoords="offset points",
                xytext=(4, 6), fontsize=7)
fig.savefig(os.path.join(HERE, "adaptivity.pdf")); plt.close(fig)

# ---------- Figure 2: the two levers (gate5 C1, C2) ----------
g5 = json.load(open(os.path.join(GATE, "ell_gate5_results.json")))
c1 = g5["levers"]["cov_contrast_nu8"]; c2 = g5["levers"]["tail_sweep_a3"]
fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.2, 3.2))

# C1: covariance contrast at nu=8
a_keys = sorted(c1, key=float); av = [float(a) for a in a_keys]
axL.plot(av, [c1[a]["exc_qda"] for a in a_keys], "o-", color=CB["qda"], label="QDA", lw=1.6, ms=5)
axL.plot(av, [c1[a]["exc_kun"] for a in a_keys], "s-", color=CB["kun"], label="derived link", lw=1.6, ms=5)
axL.set_xlabel(r"covariance contrast $a=|\Sigma_1|^{1/d}/|\Sigma_0|^{1/d}$")
axL.set_ylabel("excess risk  $R-R^\\star$")
axL.set_title(r"Covariance lever ($\nu=8$, in scope)")
axL.legend(frameon=False, fontsize=8)

# C2: tail-heaviness at a=3
nu_keys = sorted(c2, key=lambda x: -float(x)); nv = [float(n) for n in nu_keys]
axR.plot(nv, [c2[n]["exc_qda"] for n in nu_keys], "o-", color=CB["qda"], label="QDA", lw=1.6, ms=5)
axR.plot(nv, [c2[n]["exc_kun"] for n in nu_keys], "s-", color=CB["kun"], label="derived link", lw=1.6, ms=5)
axR.axvspan(6, max(nv) + 1, color="0.85", alpha=0.5, lw=0)
axR.set_xlabel(r"degrees of freedom $\nu$  (heavy $\to$ light)")
axR.set_ylabel("excess risk  $R-R^\\star$")
axR.set_title(r"Tail lever ($a=3$; shaded $=$ in scope)")
axR.legend(frameon=False, fontsize=8)
axR.invert_xaxis()
fig.tight_layout()
fig.savefig(os.path.join(HERE, "levers.pdf")); plt.close(fig)

print("wrote adaptivity.pdf, levers.pdf")
