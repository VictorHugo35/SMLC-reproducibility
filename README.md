# SMLC — Spectral Mixed-Likelihood Calibration

Reference implementation of Spectral Mixed-Likelihood Calibration (SMLC) for
probabilistic daily electric-load forecasting. The repository provides the complete
modeling, calibration, evaluation, and reserve-analysis pipeline used in the associated
study.

## Method implemented

The pipeline:

1. loads and aggregates the UCI ElectricityLoadDiagrams20112014 data;
2. constructs leakage-free training, calibration, and test blocks;
3. fits Ridge point predictors in DCT and PCA coordinates;
4. calibrates Normal, Laplace, or Student-t residual families per spectral mode;
5. evaluates DCT/PCA Gaussian and mixed variants together with QR-Direct and FTS;
6. computes probabilistic scores, coverage, significance tests, computational cost, and
   reserve-oriented evaluations; and
7. writes reproducible tables and vector figures to `figures/`.

## Repository structure

```text
.
├── src/smlc/
│   ├── config.py
│   ├── data.py
│   ├── transforms.py
│   ├── metrics.py
│   ├── evaluation.py
│   ├── reserve.py
│   ├── figures.py
│   └── models/
│       ├── smlc.py
│       ├── qr_direct.py
│       └── fts.py
├── scripts/
│   ├── run_pipeline.py
│   └── smoke_test.py
├── data/
│   └── README.md
├── figures/
│   └── README.md
├── requirements.txt
├── requirements-lock.txt
└── CITATION.cff
```

## Data

Download the public UCI ElectricityLoadDiagrams20112014 dataset from
<https://doi.org/10.24432/C58C86> and place the extracted file at:

```text
data/LD2011_2014.txt
```

Expected SHA-256:

```text
D51565F2CB5A6B768D06BA1BBD3C084C6E2F3AAB07F00C6F2DCB80E90175124B
```

The raw dataset is excluded from version control.

## Installation

Python 3.10 or newer is required. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements-lock.txt` records the exact dependency versions used for the validated
release under Python 3.14.6. Use `requirements.txt` for ordinary installation on other
supported Python versions.

## Quick validation

From the repository root:

```powershell
python .\scripts\smoke_test.py
```

## Full experiment

From the repository root:

```powershell
python -u .\scripts\run_pipeline.py
```

To save the terminal output:

```powershell
python -u .\scripts\run_pipeline.py 2>&1 | Tee-Object -FilePath .\run_pipeline.log
```

Optional arguments:

```powershell
python .\scripts\run_pipeline.py --data .\data\LD2011_2014.txt --figures-dir .\figures
python .\scripts\run_pipeline.py --cost-repeats 1
```

The complete experiment may take approximately 45–75 minutes or longer depending on the
computer. Generated tables and figures are written to `figures/` and are ignored by Git.

## Reproducibility notes

- The random seed is fixed in `src/smlc/config.py`.
- Distribution selection and model selection use only training/calibration data.
- The test block is reserved for final evaluation.
- The loader processes the raw dataset in chunks.
- Quarter-hour kW readings are converted to interval energy before hourly aggregation.
- Every probabilistic model is evaluated with the same Monte Carlo ensemble size.
