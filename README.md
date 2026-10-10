# SLNN-Assisted Decoding for STT-MRAM

This project explores neural-network-assisted decoding of a **7/9-rate sparse code** for STT-MRAM. I used **Python and PyTorch** to simulate channel errors, train a feed-forward neural network, and compare decoding performance using **BER** and **FER**.

## Paper

**“SLNN-Assisted Decoding of a 7/9-Rate Sparse Code for Reliable STT-MRAM”**

Accepted at **ICEBA 2026** and for publication in **Springer's Lecture Notes in Electrical Engineering (LNEE)**.

*Paper link will be added when available.*

## Quick Start

Run these commands from the project folder using Windows PowerShell.

**Install dependencies:**

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 1. Sweep `sigma_mu`

By default, `P1 = 2e-4` is fixed and `sigma_mu` is swept from `10` to `15` to generate five curves.

```powershell
.\.venv\Scripts\python.exe run_all_curves.py
```

For specific values:

```powershell
.\.venv\Scripts\python.exe run_all_curves.py --sigmas 9 10 11
```

Default output: `results/all_curves_sigma10_15.csv`

### 2. Sweep `P1`

Fix `sigma_mu = 10%`:

```powershell
.\.venv\Scripts\python.exe run_all_curves.py --p1-sweep-only --p1-sigma 10
```

For settings close to **Figure 6** in the paper (`sigma_mu = 9%`):

```powershell
.\.venv\Scripts\python.exe run_all_curves.py --p1-sweep-only --p1-sigma 9 --skip-only-bch
```

Results are saved in `results/` as CSV and JSON files, including BER and FER measurements.

## Model Checkpoint

The script uses the saved model at `models/deep_ffnn_model.pt` by default.

To train from scratch:

```powershell
.\.venv\Scripts\python.exe run_all_curves.py --retrain
```

To fine-tune the existing model:

```powershell
.\.venv\Scripts\python.exe run_all_curves.py --finetune
```

These two options cannot be used together.

## Notes

- Random seed: `42`.
- Results are saved incrementally during simulations.
- `--skip-only-bch` only works for the `P1` sweep.
- Run `.\.venv\Scripts\python.exe run_all_curves.py --help` for more options.
