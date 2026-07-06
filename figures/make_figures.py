#!/usr/bin/env python3
"""Generate the manuscript figure(s) from the gate JSON results.

Figure 1 is a single full-width row of three equally sized panels so that every
panel renders at the same physical scale (and hence the same font size) when
included at \\textwidth. Previously the adaptivity panel and a two-in-one
"levers" panel were emitted as separate PDFs and both placed at 0.49\\textwidth,
which squeezed the two-panel file to half scale and made its labels unreadable.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
GATE = os.path.abspath(os.path.join(HERE, "..", "results"))
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
    "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 200,
    "savefig.bbox": "tight", "lines.linewidth": 1.7, "lines.markersize": 5.5,
})
CB = {"qda": "#D55E00", "kun": "#0072B2", "ghosh": "#009E73", "acc": "#0072B2"}

# One figure, three equal panels: adaptivity | covariance lever | tail lever.
fig, (axA, axL, axR) = plt.subplots(1, 3, figsize=(13.0, 3.7))

# ---------- Panel A: adaptivity (gate4 nu-sweep) ----------
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
axA.errorbar(xs, mids, yerr=[mids - np.array(los), np.array(his) - mids],
             fmt="o-", color=CB["acc"], capsize=3)
axA.axhline(0, color="k", lw=0.8, ls="--")
axA.set_xlabel(r"$1/\nu$  (Gaussian $\to$ heavy-tailed)")
axA.set_ylabel(r"accuracy advantage (kunchenko $-$ QDA)")
axA.set_title(r"(a) Adaptivity on real covariance")
for i, (k, nu) in enumerate(zip(order, xs)):
    lab = r"$\nu{=}\infty$" if k == "inf" else rf"$\nu{{=}}{k}$"
    off = (6, 8) if i % 2 == 0 else (6, -14)   # alternate to avoid label collisions
    axA.annotate(lab, (nu, mids[list(xs).index(nu)]), textcoords="offset points",
                 xytext=off, fontsize=8)

# ---------- Panels B/C: the two levers (gate5 C1, C2) ----------
g5 = json.load(open(os.path.join(GATE, "ell_gate5_results.json")))
c1 = g5["levers"]["cov_contrast_nu8"]; c2 = g5["levers"]["tail_sweep_a3"]

# Panel B -- covariance contrast at nu=8
a_keys = sorted(c1, key=float); av = [float(a) for a in a_keys]
axL.plot(av, [c1[a]["exc_qda"] for a in a_keys], "o-", color=CB["qda"], label="QDA")
axL.plot(av, [c1[a]["exc_kun"] for a in a_keys], "s-", color=CB["kun"], label="derived link")
axL.set_xlabel(r"covariance contrast $a=|\Sigma_1|^{1/d}/|\Sigma_0|^{1/d}$")
axL.set_ylabel(r"excess risk  $R-R^\star$")
axL.set_title(r"(b) Covariance lever ($\nu=8$, in scope)")
axL.legend(frameon=False)

# Panel C -- tail-heaviness at a=3
nu_keys = sorted(c2, key=lambda x: -float(x)); nv = [float(n) for n in nu_keys]
axR.plot(nv, [c2[n]["exc_qda"] for n in nu_keys], "o-", color=CB["qda"], label="QDA")
axR.plot(nv, [c2[n]["exc_kun"] for n in nu_keys], "s-", color=CB["kun"], label="derived link")
axR.axvspan(6, max(nv) + 1, color="0.85", alpha=0.5, lw=0)
axR.set_xlabel(r"degrees of freedom $\nu$  (heavy $\to$ light)")
axR.set_ylabel(r"excess risk  $R-R^\star$")
axR.set_title(r"(c) Tail lever ($a=3$; shaded $=$ in scope)")
axR.legend(frameon=False)
axR.invert_xaxis()

fig.tight_layout(w_pad=2.0)
fig.savefig(os.path.join(HERE, "figure1.pdf")); plt.close(fig)

print("wrote figure1.pdf")
