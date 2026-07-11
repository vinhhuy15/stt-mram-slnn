from __future__ import annotations

"""Reproduce a five-curve STT-MRAM comparison for sigma_mu=10..15%.

Curves written to CSV (the MLNN column is intentionally omitted):

1) SLNN [1]
   7 payload bits -> Table-1 sparse encoder (7->9) -> STT-MRAM channel
   -> Deep FFNN 128-class decoder -> 7 payload bits.

   The label "SLNN" is retained to match the user's spreadsheet. The network
   itself is the requested Deep FFNN; there is no BCH outer code and no
   Euclidean/alpha decoder in this branch.

2) without coding
   7 raw payload bits -> STT-MRAM channel -> hard threshold detector.
   BER/FER are evaluated analytically, not by Monte Carlo.

3) BCH+sparse
   7 payload bits -> systematic BCH(15,7,5), t=2 -> append six zero pad bits
   -> three 7-bit blocks -> three Table-1 sparse encoders (3 x 7->9)
   -> channel -> same Deep FFNN per sparse block -> remove padding
   -> exhaustive nearest-BCH-codeword decoder -> 7 payload bits.

   IMPORTANT: this main comparison uses all 7 BCH message bits, rather than
   the earlier one-bit frozen-message experiment. This makes BER and FER
   directly comparable with the other curves.

4) only-BCH
   7 payload bits -> BCH(15,7,5), t=2 -> channel -> hard threshold
   -> exhaustive nearest-BCH-codeword decoder -> 7 payload bits.
   This branch is evaluated exactly by enumerating all 128 transmitted BCH
   codewords and all 2^15 hard detector outputs.

5) only-sparse (ML decoding)
   Original paper baseline: 7 payload bits -> Table-1 sparse encoder (7->9)
   -> channel -> attenuator alpha=2.5 -> Euclidean LUT decoder -> 7 bits.

Common channel assumptions:
- no offset
- P1=2e-4
- P0=P1/100 and Pr=P1/100
- read direction = write0
- mu0=1 kOhm, mu1=2 kOhm
- sigma0/mu0 = sigma1/mu1 = sigma_mu

Deep FFNN:
9 -> Linear(9,128) -> BN -> LeakyReLU(0.01)
  -> Linear(128,64) -> BN -> LeakyReLU(0.01)
  -> Linear(64,32) -> BN -> LeakyReLU(0.01)
  -> Linear(32,128) logits.

Training strategy used in this revision:
- Train exactly one global FFNN, not one model per evaluation sigma.
- Training sigmas: 0.08, 0.09, ..., 0.15.
- Total training samples: 1,500,000 (187,500 per sigma).
- Training uses no offset: offset_mu=0 and offset_sigma_ratio=0.
- The same global checkpoint is reused at every no-offset evaluation point.

CrossEntropyLoss receives raw logits. Softmax is used only by forward/inference;
argmax(logits) and argmax(softmax(logits)) give the same class decision.
"""

import argparse
import copy
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from torch import nn

from paper79.channel import (
    ChannelConfig,
    conditional_detector_error_probabilities,
    sample_resistance,
)
from paper79.codebook import CODEBOOK, MESSAGE_BITS, validate_codebook


# Dense CPU inference/training is faster and more reproducible with one thread.
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

GENERATOR = np.asarray([1, 1, 1, 0, 1, 0, 0, 0, 1], dtype=np.uint8)
SIGMAS_PERCENT = (10, 11, 12, 13, 14, 15)
SPARSE_TREE = cKDTree(CODEBOOK.astype(np.float64))


@dataclass(frozen=True)
class Config:
    # Channel
    p1: float = 2e-4
    p0_over_p1: float = 1e-2
    pr_over_p1: float = 1e-2
    read_direction: str = "write0"
    mu0: float = 1.0
    mu1: float = 2.0
    offset_mean: float = 0.0
    offset_sigma_over_mu1: float = 0.0

    # Paper's Euclidean sparse decoder
    alpha: float = 2.5

    # One global FFNN training set, mixed across sigma, with no offset.
    train_sigmas: tuple[float, ...] = (
        0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15
    )
    nr_train: int = 1_500_000
    training_chunk_size: int = 10_000
    train_fraction: float = 0.90
    train_batch_size: int = 256
    max_epochs: int = 30
    patience: int = 5
    min_delta: float = 1e-4
    learning_rate: float = 0.01
    step_size: int = 10
    lr_gamma: float = 0.5
    weight_decay: float = 0.0
    # Monte Carlo
    evaluation_batch_frames: int = 5000
    minimum_bit_errors: int = 500
    maximum_frames: int = 5_000_000

    # Reproducibility
    seed: int = 42


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Deterministic CPU kernels are used in the supplied experiment.
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def stream_rng(base_seed: int, sigma_percent: int, stream_id: int) -> np.random.Generator:
    """Create independent deterministic streams derived from base seed 42."""
    sequence = np.random.SeedSequence([base_seed, sigma_percent, stream_id])
    return np.random.default_rng(sequence)


def bits_to_indices(bits: np.ndarray) -> np.ndarray:
    arr = np.asarray(bits, dtype=np.uint8)
    if arr.ndim != 2 or arr.shape[1] != 7:
        raise ValueError("bits_to_indices expects shape [N,7]")
    weights = (1 << np.arange(6, -1, -1, dtype=np.int64))
    return (arr.astype(np.int64) @ weights).astype(np.int64)


def bch_encode(messages: np.ndarray) -> np.ndarray:
    """Systematic BCH(15,7,5), MSB-first, g(x)=111010001."""
    m = np.asarray(messages, dtype=np.uint8)
    one_message = False
    if m.ndim == 1:
        m = m[None, :]
        one_message = True
    if m.ndim != 2 or m.shape[1] != 7:
        raise ValueError("BCH messages must have shape [N,7]")

    work = np.concatenate([m, np.zeros((m.shape[0], 8), dtype=np.uint8)], axis=1)
    for position in range(7):
        rows = work[:, position].astype(bool)
        if np.any(rows):
            work[rows, position : position + 9] ^= GENERATOR
    remainder = work[:, 7:15]
    codeword = np.concatenate([m, remainder], axis=1)
    return codeword[0] if one_message else codeword


BCH_MESSAGES = MESSAGE_BITS.copy()
BCH_CODEBOOK = bch_encode(BCH_MESSAGES)


def validate_bch() -> dict[str, int]:
    if BCH_CODEBOOK.shape != (128, 15):
        raise AssertionError("BCH codebook shape is not 128x15")
    if not np.array_equal(BCH_CODEBOOK[:, :7], BCH_MESSAGES):
        raise AssertionError("BCH encoder is not systematic")

    d_min = 15
    for i in range(127):
        distances = np.sum(BCH_CODEBOOK[i + 1 :] != BCH_CODEBOOK[i], axis=1)
        d_min = min(d_min, int(distances.min()))
    if d_min != 5:
        raise AssertionError(f"Expected BCH d_min=5, obtained {d_min}")
    return {"n": 15, "k": 7, "d_min": 5, "t": 2}


class DeepFFNN(nn.Module):
    """16,288-parameter, 128-class sparse-block decoder."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(9, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 128)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        # Intentionally omit a=0.01 to reproduce the supplied training code.
        for layer in (self.fc1, self.fc2, self.fc3, self.fc4):
            nn.init.kaiming_normal_(
                layer.weight,
                nonlinearity="leaky_relu",
            )
            nn.init.zeros_(layer.bias)
        for batch_norm in (self.bn1, self.bn2, self.bn3):
            nn.init.ones_(batch_norm.weight)
            nn.init.zeros_(batch_norm.bias)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.activation(self.bn2(self.fc2(x)))
        x = self.activation(self.bn3(self.fc3(x)))
        return self.fc4(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(x), dim=1)

def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def normalize_ffnn(raw: np.ndarray) -> np.ndarray:
    return ((raw.astype(np.float32) - 1.5) / (0.5 + 1e-8)).astype(np.float32)


def mram_channel_for_training(
    data_batch: np.ndarray,
    sigma0_ratio: float,
    config: Config,
    offset_mu: float,
    offset_sigma_ratio: float,
) -> np.ndarray:
    """Training channel matching the supplied NumPy implementation.

    A single offset_mu/offset_sigma_ratio pair is supplied for a whole chunk,
    while the Gaussian offset realization remains independent per cell.
    """
    data = np.asarray(data_batch, dtype=np.uint8)
    if data.ndim != 2 or data.shape[1] != 9:
        raise ValueError("data_batch must have shape [batch, 9]")
    if config.read_direction != "write0":
        raise ValueError("The supplied training specification uses write0 direction")

    p0_physical = config.p1 * config.p0_over_p1
    pr = config.p1 * config.pr_over_p1
    p0 = (p0_physical / 2.0) * (1.0 - pr)
    p1 = config.p1 / 2.0 + (1.0 - config.p1 / 2.0) * pr

    sigma_0 = config.mu0 * sigma0_ratio
    sigma_1 = config.mu1 * sigma0_ratio
    sigma_ofs = offset_sigma_ratio * config.mu1

    rand_vals = np.random.rand(*data.shape)
    temp = np.where(
        data == 0,
        (rand_vals <= p0).astype(np.uint8),
        (rand_vals <= (1.0 - p1)).astype(np.uint8),
    )

    read_noise = np.random.randn(*data.shape)
    if sigma_ofs > 0.0:
        offset = offset_mu + sigma_ofs * np.random.randn(*data.shape)
    else:
        offset = float(offset_mu)

    return np.where(
        temp == 0,
        config.mu0 + sigma_0 * read_noise,
        config.mu1 + offset + sigma_1 * read_noise,
    )


def create_mixed_training_dataset(
    config: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create one 1.5M-sample dataset mixed over all training sigmas."""
    if not config.train_sigmas:
        raise ValueError("train_sigmas must not be empty")
    if config.nr_train % len(config.train_sigmas) != 0:
        raise ValueError(
            "nr_train must be divisible by the number of training sigmas "
            "to reproduce an equal number of samples per sigma"
        )

    # Reproduce the legacy global NumPy and PyTorch RNG behavior.
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    samples_per_sigma = config.nr_train // len(config.train_sigmas)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for sigma in config.train_sigmas:
        for start in range(0, samples_per_sigma, config.training_chunk_size):
            chunk_n = min(
                config.training_chunk_size,
                samples_per_sigma - start,
            )
            labels = np.random.randint(0, 128, size=chunk_n).astype(np.int64)

            received_raw = mram_channel_for_training(
                CODEBOOK[labels],
                sigma0_ratio=float(sigma),
                config=config,
                offset_mu=0.0,
                offset_sigma_ratio=0.0,
            )
            x_parts.append(received_raw)
            y_parts.append(labels)

    x_raw = np.concatenate(x_parts, axis=0)
    labels = np.concatenate(y_parts, axis=0)
    if x_raw.shape != (config.nr_train, 9):
        raise AssertionError(f"Unexpected training shape: {x_raw.shape}")

    x = normalize_ffnn(x_raw)
    permutation = np.random.permutation(len(x))
    x = x[permutation]
    labels = labels[permutation]

    split = int(config.train_fraction * len(x))
    return x[:split], labels[:split], x[split:], labels[split:]


def train_global_ffnn(
    config: Config,
    model_path: Path,
) -> tuple[DeepFFNN, list[dict[str, Any]], dict[str, Any]]:
    """Train one global network and reuse it at every evaluation sigma."""
    set_global_seed(config.seed)
    x_train, y_train, x_val, y_val = create_mixed_training_dataset(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepFFNN().to(device)
    if trainable_parameter_count(model) != 16_288:
        raise AssertionError("Deep FFNN parameter count must be 16,288")

    # Match the supplied implementation: keep the complete split on device.
    x_train_t = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y_train_t = torch.as_tensor(y_train, dtype=torch.long, device=device)
    x_val_t = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    y_val_t = torch.as_tensor(y_val, dtype=torch.long, device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.step_size,
        gamma=config.lr_gamma,
    )

    best_validation_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    start_time = time.time()

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        permutation = torch.randperm(x_train_t.shape[0], device=device)
        x_shuffled = x_train_t[permutation]
        y_shuffled = y_train_t[permutation]

        # Deliberately average batch means without weighting the final batch,
        # matching the documented implementation.
        batch_losses: list[float] = []
        train_correct = 0
        train_seen = 0

        for start in range(0, x_shuffled.shape[0], config.train_batch_size):
            xb = x_shuffled[start : start + config.train_batch_size]
            yb = y_shuffled[start : start + config.train_batch_size]

            optimizer.zero_grad(set_to_none=True)
            logits = model.logits(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            batch_losses.append(float(loss.item()))
            train_correct += int((logits.argmax(dim=1) == yb).sum().item())
            train_seen += int(yb.numel())

        average_train_loss = float(np.mean(batch_losses))
        train_accuracy = train_correct / train_seen

        model.eval()
        with torch.no_grad():
            val_logits = model.logits(x_val_t)
            validation_loss = float(criterion(val_logits, y_val_t).item())
            val_probabilities = torch.softmax(val_logits, dim=1)
            predictions = torch.argmax(val_probabilities, dim=1)
            validation_accuracy = float(
                (predictions == y_val_t).float().mean().item()
            )

        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "learning_rate": current_lr,
                "train_loss": average_train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )

        if validation_loss < best_validation_loss - config.min_delta:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        scheduler.step()
        print(
            f"[global train] epoch={epoch:02d} lr={current_lr:.5g} "
            f"train_loss={average_train_loss:.6g} "
            f"val_loss={validation_loss:.6g} "
            f"val_acc={validation_accuracy:.6g} "
            f"patience={epochs_without_improvement}/{config.patience}",
            flush=True,
        )
        if epochs_without_improvement >= config.patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state)

    # Recompute the metrics of the restored best checkpoint.
    model.eval()
    with torch.no_grad():
        restored_logits = model.logits(x_val_t)
        restored_loss = float(criterion(restored_logits, y_val_t).item())
        restored_accuracy = float(
            (restored_logits.argmax(dim=1) == y_val_t).float().mean().item()
        )

    model = model.cpu().eval()
    checkpoint = {
        "format": "torch_ffnn_v5_global_mixed_sigma_no_offset",
        "input_size": 9,
        "h1": 128,
        "h2": 64,
        "h3": 32,
        "output_size": 128,
        "lr": config.learning_rate,
        "output_mode": "softmax",
        "history": history,
        "state_dict": model.state_dict(),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "restored_validation_loss": restored_loss,
        "restored_validation_accuracy": restored_accuracy,
        "trainable_parameters": trainable_parameter_count(model),
        "normalization": "(x_raw - 1.5)/(0.5 + 1e-8)",
        "training_sigmas": list(config.train_sigmas),
        "nr_train": config.nr_train,
        "training_chunk_size": config.training_chunk_size,
        "training_offset_mu": 0.0,
        "training_offset_sigma_ratio": 0.0,
        "config": asdict(config),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, model_path)

    summary = {
        "checkpoint_source": "trained_now",
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "validation_loss": restored_loss,
        "validation_accuracy": restored_accuracy,
        "training_seconds": time.time() - start_time,
        "n_training_samples": int(y_train.size),
        "n_validation_samples": int(y_val.size),
        "training_sigmas": ";".join(f"{s:.2f}" for s in config.train_sigmas),
        "training_offset_mu": 0.0,
        "training_offset_sigma_ratio": 0.0,
        "device": str(device),
    }
    return model, history, summary


def load_or_train_global_ffnn(
    config: Config,
    model_path: Path,
    retrain: bool,
) -> tuple[DeepFFNN, list[dict[str, Any]], dict[str, Any]]:
    if model_path.exists() and not retrain:
        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location="cpu")

        expected_format = "torch_ffnn_v5_global_mixed_sigma_no_offset"
        if checkpoint.get("format") != expected_format:
            raise ValueError(
                "Existing checkpoint is not the global mixed-sigma model. "
                "Run with --retrain or use a clean model directory."
            )
        if tuple(float(x) for x in checkpoint.get("training_sigmas", [])) != tuple(
            config.train_sigmas
        ):
            raise ValueError("Checkpoint training-sigma list does not match Config")
        if int(checkpoint.get("nr_train", -1)) != config.nr_train:
            raise ValueError("Checkpoint nr_train does not match Config")

        model = DeepFFNN()
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        if trainable_parameter_count(model) != 16_288:
            raise AssertionError("Loaded FFNN does not have 16,288 parameters")

        summary = {
            "checkpoint_source": "loaded_existing",
            "best_epoch": checkpoint.get("best_epoch", np.nan),
            "epochs_run": len(checkpoint.get("history", [])),
            "validation_loss": checkpoint.get("restored_validation_loss", np.nan),
            "validation_accuracy": checkpoint.get(
                "restored_validation_accuracy", np.nan
            ),
            "training_seconds": 0.0,
            "n_training_samples": int(config.nr_train * config.train_fraction),
            "n_validation_samples": config.nr_train
            - int(config.nr_train * config.train_fraction),
            "training_sigmas": ";".join(f"{s:.2f}" for s in config.train_sigmas),
            "training_offset_mu": checkpoint.get("training_offset_mu", 0.0),
            "training_offset_sigma_ratio": checkpoint.get(
                "training_offset_sigma_ratio", 0.0
            ),
            "device": "loaded_on_cpu",
        }
        return model, list(checkpoint.get("history", [])), summary

    return train_global_ffnn(config, model_path)


@torch.inference_mode()
def ffnn_decode_indices(
    model: DeepFFNN,
    raw: np.ndarray,
    inference_batch_size: int = 65_536,
) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    x = torch.from_numpy(normalize_ffnn(raw.reshape(-1, 9)))
    predictions: list[np.ndarray] = []
    for start in range(0, x.shape[0], inference_batch_size):
        xb = x[start : start + inference_batch_size].to(device)
        logits = model.logits(xb)
        predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions).astype(np.int64)


def sparse_ml_decode_indices(raw: np.ndarray, alpha: float, mu0: float) -> np.ndarray:
    decoder_input = raw.reshape(-1, 9) / (alpha * mu0)
    _, decoded = SPARSE_TREE.query(decoder_input, k=1, workers=-1)
    return np.asarray(decoded, dtype=np.int64)


def build_all_hard_vectors() -> np.ndarray:
    values = np.arange(1 << 15, dtype=np.uint16)
    shifts = np.arange(14, -1, -1, dtype=np.uint16)
    return ((values[:, None] >> shifts[None, :]) & 1).astype(np.uint8)


ALL_HARD_15 = build_all_hard_vectors()


def build_bch_nearest_lookup(chunk_size: int = 1024) -> np.ndarray:
    """Map every possible hard 15-bit vector to nearest BCH codeword index.

    np.argmin provides a deterministic lowest-index tie break.
    """
    output = np.empty(1 << 15, dtype=np.uint8)
    for start in range(0, 1 << 15, chunk_size):
        received = ALL_HARD_15[start : start + chunk_size]
        distances = np.sum(
            received[:, None, :] != BCH_CODEBOOK[None, :, :],
            axis=2,
            dtype=np.int16,
        )
        output[start : start + received.shape[0]] = np.argmin(distances, axis=1).astype(np.uint8)
    return output


BCH_NEAREST_LOOKUP = build_bch_nearest_lookup()
BINARY_15_WEIGHTS = (1 << np.arange(14, -1, -1, dtype=np.int64))


def hard_vectors_to_integer(received: np.ndarray) -> np.ndarray:
    arr = np.asarray(received, dtype=np.uint8)
    if arr.ndim != 2 or arr.shape[1] != 15:
        raise ValueError("Expected received hard BCH vectors with shape [N,15]")
    return (arr.astype(np.int64) @ BINARY_15_WEIGHTS).astype(np.int64)


def nearest_bch_indices(received: np.ndarray) -> np.ndarray:
    return BCH_NEAREST_LOOKUP[hard_vectors_to_integer(received)].astype(np.int64)


def wilson_interval(errors: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        return math.nan, math.nan
    p = errors / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / trials + z*z/(4.0*trials*trials)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def metrics_record(
    method: str,
    bit_errors: int,
    frame_errors: int,
    frames: int,
    payload_bits_per_frame: int = 7,
    exact: bool = False,
) -> dict[str, Any]:
    bit_trials = frames * payload_bits_per_frame
    ber = bit_errors / bit_trials
    fer = frame_errors / frames
    if exact:
        ber_low = ber_high = ber
        fer_low = fer_high = fer
    else:
        ber_low, ber_high = wilson_interval(bit_errors, bit_trials)
        fer_low, fer_high = wilson_interval(frame_errors, frames)
    return {
        "method": method,
        "BER": ber,
        "FER": fer,
        "bit_errors": bit_errors,
        "frame_errors": frame_errors,
        "frames": frames,
        "payload_bit_trials": bit_trials,
        "BER_ci95_low": ber_low,
        "BER_ci95_high": ber_high,
        "FER_ci95_low": fer_low,
        "FER_ci95_high": fer_high,
        "evaluation": "exact" if exact else "Monte Carlo",
    }


def exact_without_coding(channel: ChannelConfig) -> dict[str, Any]:
    """Exact raw 7-bit BER and FER for equiprobable independent payload bits."""
    e0, e1 = conditional_detector_error_probabilities(channel)
    ber = 0.5 * (e0 + e1)
    fer = 1.0 - (1.0 - ber) ** 7
    return {
        "method": "without coding",
        "BER": ber,
        "FER": fer,
        "bit_errors": np.nan,
        "frame_errors": np.nan,
        "frames": np.nan,
        "payload_bit_trials": np.nan,
        "BER_ci95_low": ber,
        "BER_ci95_high": ber,
        "FER_ci95_low": fer,
        "FER_ci95_high": fer,
        "evaluation": "exact analytical",
        "detector_error_given_0": e0,
        "detector_error_given_1": e1,
    }


def exact_only_bch(channel: ChannelConfig) -> dict[str, Any]:
    """Exact only-BCH BER/FER by exhaustive hard-output enumeration."""
    e0, e1 = conditional_detector_error_probabilities(channel)
    decoded_messages = BCH_MESSAGES[BCH_NEAREST_LOOKUP]

    ber_sum = 0.0
    fer_sum = 0.0
    probability_mass_error_max = 0.0

    # Average uniformly over all 128 possible 7-bit messages.
    for tx_index in range(128):
        transmitted = BCH_CODEBOOK[tx_index]
        # Matrix [32768,15]: Pr(received hard bit | transmitted hard bit)
        probabilities_by_bit = np.empty_like(ALL_HARD_15, dtype=np.float64)

        tx0 = transmitted == 0
        tx1 = ~tx0
        probabilities_by_bit[:, tx0] = np.where(
            ALL_HARD_15[:, tx0] == 0,
            1.0 - e0,
            e0,
        )
        probabilities_by_bit[:, tx1] = np.where(
            ALL_HARD_15[:, tx1] == 1,
            1.0 - e1,
            e1,
        )
        probabilities = np.prod(probabilities_by_bit, axis=1)
        probability_mass_error_max = max(
            probability_mass_error_max,
            abs(float(probabilities.sum()) - 1.0),
        )

        payload_errors = decoded_messages != BCH_MESSAGES[tx_index]
        hamming = np.sum(payload_errors, axis=1)
        # Elementwise sum is intentionally used instead of np.dot here.
        # On some BLAS builds, dot(float, bool) is unexpectedly very slow.
        ber_sum += float(np.sum(probabilities * (hamming / 7.0)))
        fer_sum += float(np.sum(probabilities * (hamming > 0)))

    ber = ber_sum / 128.0
    fer = fer_sum / 128.0
    return {
        "method": "only-BCH",
        "BER": ber,
        "FER": fer,
        "bit_errors": np.nan,
        "frame_errors": np.nan,
        "frames": 128 * (1 << 15),
        "payload_bit_trials": np.nan,
        "BER_ci95_low": ber,
        "BER_ci95_high": ber,
        "FER_ci95_low": fer,
        "FER_ci95_high": fer,
        "evaluation": "exact exhaustive enumeration",
        "probability_mass_error_max": probability_mass_error_max,
        "detector_error_given_0": e0,
        "detector_error_given_1": e1,
    }


def simulate_slnn_and_only_sparse_ml(
    model: nn.Module,
    channel: ChannelConfig,
    config: Config,
    sigma_percent: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Paired Monte Carlo: FFNN and paper ML decode the same received blocks."""
    rng = stream_rng(config.seed, sigma_percent, stream_id=1)
    frames = 0
    ffnn_bit_errors = 0
    ffnn_frame_errors = 0
    ml_bit_errors = 0
    ml_frame_errors = 0
    start_time = time.time()

    while frames < config.maximum_frames:
        n = min(config.evaluation_batch_frames, config.maximum_frames - frames)
        source_indices = rng.integers(0, 128, size=n, dtype=np.int64)
        transmitted_message = MESSAGE_BITS[source_indices]
        raw = sample_resistance(CODEBOOK[source_indices], channel, rng)

        ffnn_indices = ffnn_decode_indices(model, raw)
        ml_indices = sparse_ml_decode_indices(raw, config.alpha, channel.mu0)

        ffnn_errors = MESSAGE_BITS[ffnn_indices] != transmitted_message
        ml_errors = MESSAGE_BITS[ml_indices] != transmitted_message

        ffnn_bit_errors += int(ffnn_errors.sum())
        ffnn_frame_errors += int(np.any(ffnn_errors, axis=1).sum())
        ml_bit_errors += int(ml_errors.sum())
        ml_frame_errors += int(np.any(ml_errors, axis=1).sum())
        frames += n

        if frames % 100_000 == 0 or frames == config.maximum_frames:
            print(
                f"[eval sparse] sigma={sigma_percent}% frames={frames:,} "
                f"SLNN_bit_errors={ffnn_bit_errors} ML_bit_errors={ml_bit_errors}",
                flush=True,
            )

        if (
            ffnn_bit_errors >= config.minimum_bit_errors
            and ml_bit_errors >= config.minimum_bit_errors
        ):
            break

    ffnn_result = metrics_record(
        "SLNN [1]",
        ffnn_bit_errors,
        ffnn_frame_errors,
        frames,
    )
    ml_result = metrics_record(
        "only-sparse (ML decoding)",
        ml_bit_errors,
        ml_frame_errors,
        frames,
    )
    elapsed = time.time() - start_time
    ffnn_result["evaluation_seconds"] = elapsed
    ml_result["evaluation_seconds"] = elapsed
    ffnn_result["stopped_by_bit_error_target"] = ffnn_bit_errors >= config.minimum_bit_errors
    ml_result["stopped_by_bit_error_target"] = ml_bit_errors >= config.minimum_bit_errors
    return ffnn_result, ml_result


def simulate_bch_sparse_ffnn(
    model: nn.Module,
    channel: ChannelConfig,
    config: Config,
    sigma_percent: int,
) -> dict[str, Any]:
    """Full-7-bit BCH+sparse+FFNN simulation (not the frozen one-bit variant)."""
    rng = stream_rng(config.seed, sigma_percent, stream_id=2)
    frames = 0
    bit_errors = 0
    frame_errors = 0
    bch_codeword_errors = 0
    sparse_block_errors = 0
    sparse_block_bit_errors = 0
    start_time = time.time()

    while frames < config.maximum_frames:
        n = min(config.evaluation_batch_frames, config.maximum_frames - frames)
        source_indices = rng.integers(0, 128, size=n, dtype=np.int64)
        transmitted_messages = BCH_MESSAGES[source_indices]
        transmitted_bch = BCH_CODEBOOK[source_indices]

        bits21 = np.concatenate(
            [transmitted_bch, np.zeros((n, 6), dtype=np.uint8)],
            axis=1,
        )
        blocks7 = bits21.reshape(n * 3, 7)
        sparse_indices = bits_to_indices(blocks7)
        raw = sample_resistance(CODEBOOK[sparse_indices], channel, rng)
        decoded_sparse_indices = ffnn_decode_indices(model, raw)
        decoded_blocks7 = MESSAGE_BITS[decoded_sparse_indices]

        sparse_block_errors += int(np.sum(decoded_sparse_indices != sparse_indices))
        sparse_block_bit_errors += int(np.sum(decoded_blocks7 != blocks7))

        received21 = decoded_blocks7.reshape(n, 21)
        received15 = received21[:, :15]
        decoded_bch_indices = nearest_bch_indices(received15)
        decoded_messages = BCH_MESSAGES[decoded_bch_indices]

        payload_errors = decoded_messages != transmitted_messages
        bit_errors += int(payload_errors.sum())
        frame_errors += int(np.any(payload_errors, axis=1).sum())
        bch_codeword_errors += int(np.sum(decoded_bch_indices != source_indices))
        frames += n

        if frames % 100_000 == 0 or frames == config.maximum_frames:
            print(
                f"[eval BCH+sparse] sigma={sigma_percent}% frames={frames:,} "
                f"bit_errors={bit_errors} frame_errors={frame_errors}",
                flush=True,
            )
        if bit_errors >= config.minimum_bit_errors:
            break

    result = metrics_record("BCH+sparse", bit_errors, frame_errors, frames)
    result.update(
        {
            "BCH_codeword_error_rate": bch_codeword_errors / frames,
            "BCH_codeword_errors": bch_codeword_errors,
            "sparse_block_class_error_rate": sparse_block_errors / (frames * 3),
            "sparse_block_message_BER": sparse_block_bit_errors / (frames * 3 * 7),
            "evaluation_seconds": time.time() - start_time,
            "stopped_by_bit_error_target": bit_errors >= config.minimum_bit_errors,
        }
    )
    return result


def channel_for_sigma(config: Config, sigma_percent: int) -> ChannelConfig:
    return ChannelConfig(
        mu0=config.mu0,
        mu1=config.mu1,
        sigma_ratio=sigma_percent / 100.0,
        p1=config.p1,
        p0_over_p1=config.p0_over_p1,
        pr_over_p1=config.pr_over_p1,
        read_direction=config.read_direction,
        offset_mean=config.offset_mean,
        offset_sigma_over_mu1=config.offset_sigma_over_mu1,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    config = Config(
        train_sigmas=tuple(args.train_sigmas),
        nr_train=args.nr_train,
        training_chunk_size=args.training_chunk_size,
        train_batch_size=args.train_batch_size,
        max_epochs=args.max_epochs,
        minimum_bit_errors=args.min_bit_errors,
        maximum_frames=args.max_frames,
        evaluation_batch_frames=args.batch_frames,
    )
    validate_codebook()
    bch_description = validate_bch()
    if trainable_parameter_count(DeepFFNN()) != 16_288:
        raise AssertionError("FFNN parameter-count validation failed")

    compact_rows: list[dict[str, Any]] = []
    detailed_rows: list[dict[str, Any]] = []
    run_start = time.time()

    # Train/load exactly one mixed-sigma, no-offset FFNN.
    model_path = model_dir / "deep_ffnn_model.pt"
    model, training_history_rows, training_summary = load_or_train_global_ffnn(
        config,
        model_path,
        retrain=args.retrain,
    )
    training_rows: list[dict[str, Any]] = [training_summary]

    for sigma_percent in args.sigmas:
        # Evaluation remains no-offset at each requested sigma.
        channel = channel_for_sigma(config, sigma_percent)

        without_coding = exact_without_coding(channel)
        only_bch = exact_only_bch(channel)
        slnn, only_sparse_ml = simulate_slnn_and_only_sparse_ml(
            model,
            channel,
            config,
            sigma_percent,
        )
        bch_sparse = simulate_bch_sparse_ffnn(
            model,
            channel,
            config,
            sigma_percent,
        )

        method_results = [slnn, without_coding, bch_sparse, only_bch, only_sparse_ml]
        for method_result in method_results:
            row = {
                "P1": config.p1,
                "sigma_mu_percent": sigma_percent,
                "sigma_mu_ratio": sigma_percent / 100.0,
                "offset_mu_kohm": config.offset_mean,
                "offset_sigma_ratio": config.offset_sigma_over_mu1,
                **method_result,
            }
            detailed_rows.append(row)

        compact_rows.append(
            {
                "P1": config.p1,
                "sigma_mu": sigma_percent,
                "SLNN [1] BER": slnn["BER"],
                "SLNN [1] FER": slnn["FER"],
                "without coding BER": without_coding["BER"],
                "without coding FER": without_coding["FER"],
                "BCH+sparse BER": bch_sparse["BER"],
                "BCH+sparse FER": bch_sparse["FER"],
                "only-BCH BER": only_bch["BER"],
                "only-BCH FER": only_bch["FER"],
                "only-sparse (ML decoding) BER": only_sparse_ml["BER"],
                "only-sparse (ML decoding) FER": only_sparse_ml["FER"],
            }
        )

        # Save incrementally so a long run is not lost if interrupted.
        pd.DataFrame(compact_rows).to_csv(
            output_dir / "all_curves_sigma10_15.csv",
            index=False,
            float_format="%.12e",
        )
        pd.DataFrame(detailed_rows).to_csv(
            output_dir / "all_curves_sigma10_15_detailed.csv",
            index=False,
            float_format="%.12e",
        )
        pd.DataFrame(training_rows).to_csv(
            output_dir / "ffnn_checkpoint_summary.csv",
            index=False,
            float_format="%.12e",
        )
        if training_history_rows:
            pd.DataFrame(training_history_rows).to_csv(
                output_dir / "ffnn_training_history.csv",
                index=False,
                float_format="%.12e",
            )

        print("\nCurrent compact table:")
        print(pd.DataFrame(compact_rows).to_string(index=False), flush=True)

    script_path = Path(__file__).resolve()
    manifest = {
        "description": "Five-curve sigma_mu sweep; MLNN omitted",
        "created_by_script": str(script_path),
        "script_sha256": file_sha256(script_path),
        "config": asdict(config),
        "sigmas_percent": list(args.sigmas),
        "bch": bch_description,
        "sparse_codebook": {
            "source": "Table 1 of Nguyen, IEEE Access 2021",
            "shape": list(CODEBOOK.shape),
            "weight_2_codewords": int(np.sum(CODEBOOK.sum(axis=1) == 2)),
            "weight_4_codewords": int(np.sum(CODEBOOK.sum(axis=1) == 4)),
        },
        "method_definitions": {
            "SLNN [1]": "sparse 7/9 encoder + Deep FFNN decoder; no BCH; no alpha",
            "without coding": "7 raw bits + threshold detector; exact analytical BER/FER",
            "BCH+sparse": "full 7-bit BCH(15,7,t=2) + pad6 + 3 sparse blocks + FFNN + standard nearest-BCH decoder",
            "only-BCH": "BCH(15,7,t=2) + threshold + nearest BCH; exact exhaustive hard-output enumeration",
            "only-sparse (ML decoding)": "paper baseline sparse 7/9 + alpha=2.5 Euclidean decoder",
        },
        "payload_bits_per_frame": 7,
        "physical_bits_per_frame": {
            "SLNN [1]": 9,
            "without coding": 7,
            "BCH+sparse": 27,
            "only-BCH": 15,
            "only-sparse (ML decoding)": 9,
        },
        "ffnn_training_strategy": (
            "One global no-offset model trained on sigma=0.08..0.15 with "
            "1.5M samples; reused for all evaluation sigmas"
        ),
        "checkpoint_file": {
            "path": str(model_path),
            "sha256": file_sha256(model_path),
        },
        "elapsed_seconds": time.time() - run_start,
        "software": {
            "python": __import__("sys").version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
        },
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run SLNN, raw, BCH+sparse, only-BCH and only-sparse ML curves."
    )
    parser.add_argument(
        "--sigmas",
        type=int,
        nargs="+",
        default=list(SIGMAS_PERCENT),
        help="sigma_mu percentages; default: 10 11 12 13 14 15",
    )
    parser.add_argument(
        "--train-sigmas",
        type=float,
        nargs="+",
        default=[0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15],
        help="Sigma ratios mixed into the single FFNN training set.",
    )
    parser.add_argument("--nr-train", type=int, default=1_500_000)
    parser.add_argument("--training-chunk-size", type=int, default=10_000)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--batch-frames", type=int, default=5000)
    parser.add_argument("--min-bit-errors", type=int, default=500)
    parser.add_argument("--max-frames", type=int, default=5_000_000)
    parser.add_argument(
        "--model-dir",
        default=str(project_dir / "models"),
        help="Directory containing/receiving the single global FFNN checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_dir / "results"),
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Ignore an existing checkpoint and retrain the single global FFNN.",
    )
    parsed = parser.parse_args()
    invalid = [sigma for sigma in parsed.sigmas if not (1 <= sigma <= 100)]
    if invalid:
        parser.error(f"Invalid sigma percentages: {invalid}")
    invalid_train = [sigma for sigma in parsed.train_sigmas if not (0.0 < sigma < 1.0)]
    if invalid_train:
        parser.error(f"Invalid training sigma ratios: {invalid_train}")
    if parsed.nr_train <= 0 or parsed.training_chunk_size <= 0:
        parser.error("nr-train and training-chunk-size must be positive")
    if parsed.nr_train % len(parsed.train_sigmas) != 0:
        parser.error("nr-train must be divisible by the number of train-sigmas")
    return parsed


if __name__ == "__main__":
    run(parse_args())
