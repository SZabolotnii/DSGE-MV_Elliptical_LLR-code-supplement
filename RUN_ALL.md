# Run Notes

Run commands from the package root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Core gates:

```bash
cd code
python run_ell_gate.py
python run_ell_gate2.py
python run_ell_gate3.py
python run_ell_gate5_rates.py
python run_ell_gate6_hdlss.py
```

Financial-series gates need local FRED CSV files:

```bash
export GELL_DEXCAUS_CSV=/path/to/DEXCAUS.csv
export GELL_FRED_DATA_DIR=/path/to/fred-heavytail
cd code
python run_ell_gate4.py
python run_ell_gate4b_blocked.py
python run_ell_gate4c_multiseries.py
python run_ell_ellipticity_test.py
```

Faithful fitted-GAM comparators require R and `mgcv`:

```bash
Rscript -e 'install.packages("mgcv", repos="https://cloud.r-project.org")'
cd code
python run_ell_gate_mgcv_faithful.py
python run_ell_gate4c_mgcv_faithful.py
python run_ell_wine_multinom_faithful.py
```

Regenerate manuscript figures from recorded results:

```bash
cd figures
python make_figures.py
```

The scripts write JSON outputs into the current working directory. The checked
outputs used for the manuscript are stored in `results/`.
