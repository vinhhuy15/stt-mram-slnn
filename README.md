# STT-MRAM: Quick Run Guide

## Prerequisites

Run the following commands from the project directory using the virtual environment:

```powershell
.\.venv\Scripts\python.exe run_all_curves.py --help
```

If the required dependencies have not been installed yet:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 1. Sweep by `sigma_mu`

By default, `P1 = 2e-4` is fixed, while `sigma_mu` is swept from `10` to `15`, generating all five curves:

```powershell
.\.venv\Scripts\python.exe run_all_curves.py
```

To run only selected sigma values:

```powershell
.\.venv\Scripts\python.exe run_all_curves.py --sigmas 9 10 11
```

Main output:

```text
results/all_curves_sigma10_15.csv
```

## 2. Sweep by `P1`

Fix `sigma_mu = 10%`:

```powershell
.\.venv\Scripts\python.exe run_all_curves.py --p1-sweep-only --p1-sigma 10
```

To speed up the simulation by skipping the `only-BCH` curve:

```powershell
.\.venv\Scripts\python.exe run_all_curves.py --p1-sweep-only --p1-sigma 10 --skip-only-bch
```

For conditions close to those used in Figure 6 with `sigma_mu = 9%`:

```powershell
.\.venv\Scripts\python.exe run_all_curves.py --p1-sweep-only --p1-sigma 9 --skip-only-bch
```

To specify custom `P1` values:

```powershell
.\.venv\Scripts\python.exe run_all_curves.py --p1-sweep-only --p1-sigma 10 --p1-values 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 --skip-only-bch
```

Output files are named according to the selected sigma value. For example:

* `results/all_curves_p1_sigma9.csv`
* `results/all_curves_p1_sigma10.csv`
* `results/run_manifest_p1_sigma9.json`
* `results/run_manifest_p1_sigma10.json`

The P1-sweep CSV files contain both BER and FER for each simulated curve.

## 3. FFNN Checkpoint

By default, the program loads the existing checkpoint:

```text
models/deep_ffnn_model.pt
```

To retrain the model from scratch:

```powershell
.\.venv\Scripts\python.exe run_all_curves.py --retrain
```

To fine-tune the existing checkpoint:

```powershell
.\.venv\Scripts\python.exe run_all_curves.py --finetune
```

`--retrain` and `--finetune` cannot be used at the same time.

## Notes

* `--skip-only-bch` currently applies only to the P1 sweep.
* All simulations use a fixed random seed of `42`.
* Results are appended to the CSV file incrementally after each simulation point.
