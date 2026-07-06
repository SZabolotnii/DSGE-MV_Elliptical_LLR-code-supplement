# Closed-form fractional radial links for elliptical Mahalanobis discriminant analysis

This folder is the standalone reproducibility package for the manuscript
`Closed-form fractional radial links for elliptical Mahalanobis discriminant analysis`.

It contains the experiment runners, recorded JSON outputs, the Lean 4 proof
scaffold, and the figure-generation script used by the article. The package is
prepared as a seed for a separate public repository.

## Contents

- `code/` - Python and R experiment runners.
- `results/` - JSON outputs reported by the manuscript gates.
- `figures/` - figure PDFs and the script used to regenerate them.
- `lean/` - Lean 4 development for the formal bridge checks.
- `docs/` - gate specification and verdict notes.

## Data

Most synthetic and sklearn benchmark gates run without external files. The
financial-series gates require public FRED CSV files:

- single-series CAD/USD gate: set `GELL_DEXCAUS_CSV=/path/to/DEXCAUS.csv`
- multi-series gates: set `GELL_FRED_DATA_DIR=/path/to/fred-heavytail`

The multi-series directory should contain `DEXCAUS.csv`, `DEXJPUS.csv`,
`DEXUSUK.csv`, `DCOILWTICO.csv`, `SP500.csv`, and `DEXUSEU.csv`.

All randomized experiments use seed `2026`.

## Software

Python dependencies are listed in `requirements.txt`. The faithful fitted-GAM
comparators also require R with the `mgcv` package.

Run commands are summarized in `RUN_ALL.md`.
