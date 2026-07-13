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
   16 payload bits -> systematic BCH(31,16,7), t=3 -> append four shaping bits
   -> five 7-bit blocks -> five Table-1 sparse encoders (5 x 7->9)
   -> channel -> same Deep FFNN per sparse block -> remove padding
   -> discard shaping bits -> syndrome bounded-distance BCH decoder -> 16 bits.
   For each 3-bit BCH suffix, the encoder selects one of 16 padding patterns
   whose final sparse class has the best calibrated FFNN accuracy.

4) only-BCH
   16 payload bits -> BCH(31,16,7), t=3 -> channel -> hard threshold
   -> syndrome bounded-distance decoder -> 16 payload bits (Monte Carlo).

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
import itertools
import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass, replace
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
from paper79.distance_decoders import (
    euclidean_decode_indices,
    mahalanobis_decode_indices,
    pooled_within_class_covariance,
)
from paper79.joint_bch_sparse import (
    BCH15_PHYSICAL_CODEBOOK,
    encode_bch15_sparse,
    joint_ml_decode_indices,
    sequential_bch15_decode_from_sparse_indices,
    validate_joint_bch15_sparse,
)


# Dense CPU inference/training is faster and more reproducible with one thread.
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

# Primitive narrow-sense binary BCH(31,16,7), t=3.  Coefficients are
# MSB-first: x^15+x^11+x^10+x^9+x^8+x^7+x^5+x^3+x^2+x+1.
GENERATOR = np.asarray(
    [1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1], dtype=np.uint8
)
SIGMAS_PERCENT = (10, 11, 12, 13, 14, 15)
P1_SWEEP = (2e-8, 2e-7, 2e-6, 2e-5, 2e-4, 2e-3, 2e-2)
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
    shaping_calibration_samples: int = 2_000

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
    """Systematic BCH(31,16,7), MSB-first."""
    m = np.asarray(messages, dtype=np.uint8)
    one_message = False
    if m.ndim == 1:
        m = m[None, :]
        one_message = True
    if m.ndim != 2 or m.shape[1] != 16:
        raise ValueError("BCH messages must have shape [N,16]")

    work = np.concatenate([m, np.zeros((m.shape[0], 15), dtype=np.uint8)], axis=1)
    for position in range(16):
        rows = work[:, position].astype(bool)
        if np.any(rows):
            work[rows, position : position + 16] ^= GENERATOR
    remainder = work[:, 16:31]
    codeword = np.concatenate([m, remainder], axis=1)
    return codeword[0] if one_message else codeword


BCH_MESSAGE_VALUES = np.arange(1 << 16, dtype=np.uint32)
BCH_MESSAGES = (
    (BCH_MESSAGE_VALUES[:, None] >> np.arange(15, -1, -1, dtype=np.uint32)) & 1
).astype(np.uint8)


def validate_bch() -> dict[str, int]:
    codebook = bch_encode(BCH_MESSAGES)
    if codebook.shape != (65536, 31):
        raise AssertionError("BCH codebook shape is not 65536x31")
    if not np.array_equal(codebook[:, :16], BCH_MESSAGES):
        raise AssertionError("BCH encoder is not systematic")
    # The code is linear, so its minimum distance is the minimum nonzero
    # codeword weight; no O(M^2) pairwise comparison is needed.
    d_min = int(np.sum(codebook[1:], axis=1).min())
    if d_min != 7:
        raise AssertionError(f"Expected BCH d_min=7, obtained {d_min}")
    return {"n": 31, "k": 16, "d_min": 7, "t": 3, "padding_bits": 4}


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
    initial_state: dict[str, torch.Tensor] | None = None,
    checkpoint_source: str = "trained_now",
) -> tuple[DeepFFNN, list[dict[str, Any]], dict[str, Any]]:
    """Train one global network and reuse it at every evaluation sigma."""
    set_global_seed(config.seed)
    x_train, y_train, x_val, y_val = create_mixed_training_dataset(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepFFNN().to(device)
    if initial_state is not None:
        model.load_state_dict(initial_state)
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
        "checkpoint_source": checkpoint_source,
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
    finetune: bool,
) -> tuple[DeepFFNN, list[dict[str, Any]], dict[str, Any]]:
    if retrain and finetune:
        raise ValueError("--retrain and --finetune cannot be used together")

    if finetune:
        if not model_path.exists():
            raise FileNotFoundError("--finetune requires an existing FFNN checkpoint")
        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location="cpu")
        if "state_dict" not in checkpoint:
            raise ValueError("Existing checkpoint has no state_dict to fine-tune")
        backup_path = model_path.with_name(model_path.stem + "_pre_finetune.pt")
        shutil.copy2(model_path, backup_path)
        print(f"[fine-tune] backed up original checkpoint to {backup_path}", flush=True)
        return train_global_ffnn(
            config,
            model_path,
            initial_state=checkpoint["state_dict"],
            checkpoint_source="fine_tuned_existing",
        )

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
            "n_training_samples": int(
                checkpoint.get("nr_train", config.nr_train) * config.train_fraction
            ),
            "n_validation_samples": int(
                checkpoint.get("nr_train", config.nr_train) * (1.0 - config.train_fraction)
            ),
            "training_sigmas": ";".join(
                f"{float(s):.2f}" for s in checkpoint.get("training_sigmas", [])
            ),
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


def calibrate_shaping_padding(
    model: DeepFFNN,
    channel: ChannelConfig,
    rng: np.random.Generator,
    samples_per_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose the most reliable last-block class for each 3-bit BCH suffix.

    The high three bits of each candidate are fixed by BCH.  Its low four
    bits are free shaping bits, giving 16 candidate sparse codewords.
    """
    class_indices = np.repeat(np.arange(128, dtype=np.int64), samples_per_class)
    raw = sample_resistance(CODEBOOK[class_indices], channel, rng)
    decoded = ffnn_decode_indices(model, raw)
    accuracy = np.bincount(
        class_indices, weights=(decoded == class_indices), minlength=128
    ) / samples_per_class

    chosen_indices = np.empty(8, dtype=np.int64)
    for prefix in range(8):
        start = prefix << 4
        # np.argmax gives a deterministic lowest-padding tie break.
        chosen_indices[prefix] = start + int(np.argmax(accuracy[start : start + 16]))
    padding = MESSAGE_BITS[chosen_indices, 3:].copy()
    return padding, accuracy


def sparse_ml_decode_indices(raw: np.ndarray, alpha: float, mu0: float) -> np.ndarray:
    decoder_input = raw.reshape(-1, 9) / (alpha * mu0)
    _, decoded = SPARSE_TREE.query(decoder_input, k=1, workers=-1)
    return np.asarray(decoded, dtype=np.int64)


SYNDROME_WEIGHTS = (1 << np.arange(14, -1, -1, dtype=np.uint32))
BCH_BIT_WEIGHTS = (1 << np.arange(30, -1, -1, dtype=np.uint32))


def bch_syndromes(vectors: np.ndarray) -> np.ndarray:
    """Return polynomial remainders (15-bit syndromes) for 31-bit vectors."""
    work = np.asarray(vectors, dtype=np.uint8).copy()
    if work.ndim != 2 or work.shape[1] != 31:
        raise ValueError("Expected BCH vectors with shape [N,31]")
    for position in range(16):
        rows = work[:, position].astype(bool)
        if np.any(rows):
            work[rows, position : position + 16] ^= GENERATOR
    return (work[:, 16:].astype(np.uint32) @ SYNDROME_WEIGHTS).astype(np.uint16)


def build_bch_error_lookup() -> tuple[np.ndarray, np.ndarray]:
    """Coset leaders for every error pattern of weight <= 3."""
    masks = np.zeros(1 << 15, dtype=np.uint32)
    valid = np.zeros(1 << 15, dtype=bool)
    valid[0] = True
    for weight in (1, 2, 3):
        combinations = list(itertools.combinations(range(31), weight))
        errors = np.zeros((len(combinations), 31), dtype=np.uint8)
        for row, positions in enumerate(combinations):
            errors[row, list(positions)] = 1
        syndromes = bch_syndromes(errors)
        integer_masks = errors.astype(np.uint32) @ BCH_BIT_WEIGHTS
        if np.any(valid[syndromes]):
            raise AssertionError("BCH syndromes collide for error weight <= 3")
        masks[syndromes] = integer_masks
        valid[syndromes] = True
    return masks, valid


BCH_ERROR_MASKS, BCH_CORRECTABLE_SYNDROMES = build_bch_error_lookup()


def bch_decode(received: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bounded-distance decode; uncorrectable words are returned unchanged."""
    arr = np.asarray(received, dtype=np.uint8)
    syndromes = bch_syndromes(arr)
    values = arr.astype(np.uint32) @ BCH_BIT_WEIGHTS
    corrected_values = values ^ BCH_ERROR_MASKS[syndromes]
    corrected = (
        (corrected_values[:, None] >> np.arange(30, -1, -1, dtype=np.uint32)) & 1
    ).astype(np.uint8)
    return corrected[:, :16], BCH_CORRECTABLE_SYNDROMES[syndromes]


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


def simulate_only_bch(
    channel: ChannelConfig, config: Config, sigma_percent: int
) -> dict[str, Any]:
    """Monte Carlo BCH(31,16,7) with hard detection and t=3 decoding."""
    e0, e1 = conditional_detector_error_probabilities(channel)
    rng = stream_rng(config.seed, sigma_percent, stream_id=3)
    frames = bit_errors = frame_errors = decoder_failures = 0
    start_time = time.time()
    while frames < config.maximum_frames:
        n = min(config.evaluation_batch_frames, config.maximum_frames - frames)
        messages = rng.integers(0, 2, size=(n, 16), dtype=np.uint8)
        transmitted = bch_encode(messages)
        flip_probability = np.where(transmitted == 0, e0, e1)
        received = transmitted ^ (rng.random(transmitted.shape) < flip_probability)
        decoded, correctable = bch_decode(received)
        errors = decoded != messages
        bit_errors += int(errors.sum())
        frame_errors += int(np.any(errors, axis=1).sum())
        decoder_failures += int((~correctable).sum())
        frames += n
        if bit_errors >= config.minimum_bit_errors:
            break
    result = metrics_record(
        "only-BCH", bit_errors, frame_errors, frames, payload_bits_per_frame=16
    )
    result.update({
        "evaluation_seconds": time.time() - start_time,
        "stopped_by_bit_error_target": bit_errors >= config.minimum_bit_errors,
        "decoder_failure_rate": decoder_failures / frames,
        "detector_error_given_0": e0,
        "detector_error_given_1": e1,
    })
    return result


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


def simulate_classical_distance_decoders(
    channel: ChannelConfig,
    config: Config,
    sigma_percent: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Paired comparison of Euclidean and Mahalanobis, with no AI model."""
    rng = stream_rng(config.seed, sigma_percent, stream_id=4)
    counters = {
        "Euclidean": {"bits": 0, "frames": 0},
        "Mahalanobis": {"bits": 0, "frames": 0},
    }
    frames = 0
    start_time = time.time()
    while frames < config.maximum_frames:
        n = min(config.evaluation_batch_frames, config.maximum_frames - frames)
        source_indices = rng.integers(0, 128, size=n, dtype=np.int64)
        transmitted_messages = MESSAGE_BITS[source_indices]
        raw = sample_resistance(CODEBOOK[source_indices], channel, rng)

        decoded = {
            "Euclidean": euclidean_decode_indices(raw, channel),
            "Mahalanobis": mahalanobis_decode_indices(raw, channel),
        }
        for name, decoded_indices in decoded.items():
            errors = MESSAGE_BITS[decoded_indices] != transmitted_messages
            counters[name]["bits"] += int(errors.sum())
            counters[name]["frames"] += int(np.any(errors, axis=1).sum())
        frames += n

        if frames % 100_000 == 0 or frames == config.maximum_frames:
            print(
                f"[classical distance] sigma={sigma_percent}% frames={frames:,} "
                f"Euclidean_errors={counters['Euclidean']['bits']} "
                f"Mahalanobis_errors={counters['Mahalanobis']['bits']}",
                flush=True,
            )
        if all(
            values["bits"] >= config.minimum_bit_errors
            for values in counters.values()
        ):
            break

    elapsed = time.time() - start_time
    results = []
    for name in ("Euclidean", "Mahalanobis"):
        result = metrics_record(
            name,
            counters[name]["bits"],
            counters[name]["frames"],
            frames,
        )
        result.update(
            {
                "evaluation_seconds": elapsed,
                "stopped_by_bit_error_target": (
                    counters[name]["bits"] >= config.minimum_bit_errors
                ),
            }
        )
        results.append(result)
    return results[0], results[1]


def simulate_bch_sparse_ffnn(
    model: nn.Module,
    channel: ChannelConfig,
    config: Config,
    sigma_percent: int,
) -> dict[str, Any]:
    """BCH(31,16,7) + four zero pad bits + five sparse/FFNN blocks."""
    rng = stream_rng(config.seed, sigma_percent, stream_id=2)
    frames = 0
    bit_errors = 0
    frame_errors = 0
    bch_codeword_errors = 0
    sparse_block_errors = 0
    sparse_block_bit_errors = 0
    start_time = time.time()
    shaping_rng = stream_rng(config.seed, sigma_percent, stream_id=22)
    shaping_padding, shaping_class_accuracy = calibrate_shaping_padding(
        model,
        channel,
        shaping_rng,
        config.shaping_calibration_samples,
    )
    shaping_indices = (
        np.arange(8, dtype=np.int64) * 16
        + (shaping_padding.astype(np.int64) @ np.asarray([8, 4, 2, 1]))
    )
    fixed_zero_indices = np.arange(8, dtype=np.int64) * 16

    while frames < config.maximum_frames:
        n = min(config.evaluation_batch_frames, config.maximum_frames - frames)
        transmitted_messages = rng.integers(0, 2, size=(n, 16), dtype=np.uint8)
        transmitted_bch = bch_encode(transmitted_messages)

        suffix_values = bits_to_indices(
            np.concatenate(
                [np.zeros((n, 4), dtype=np.uint8), transmitted_bch[:, 28:31]], axis=1
            )
        )
        selected_padding = shaping_padding[suffix_values]
        bits35 = np.concatenate([transmitted_bch, selected_padding], axis=1)
        blocks7 = bits35.reshape(n * 5, 7)
        sparse_indices = bits_to_indices(blocks7)
        raw = sample_resistance(CODEBOOK[sparse_indices], channel, rng)
        decoded_sparse_indices = ffnn_decode_indices(model, raw)
        decoded_blocks7 = MESSAGE_BITS[decoded_sparse_indices]

        sparse_block_errors += int(np.sum(decoded_sparse_indices != sparse_indices))
        sparse_block_bit_errors += int(np.sum(decoded_blocks7 != blocks7))

        received35 = decoded_blocks7.reshape(n, 35)
        decoded_messages, correctable = bch_decode(received35[:, :31])

        payload_errors = decoded_messages != transmitted_messages
        bit_errors += int(payload_errors.sum())
        frame_errors += int(np.any(payload_errors, axis=1).sum())
        bch_codeword_errors += int(np.any(payload_errors, axis=1).sum())
        frames += n

        if frames % 100_000 == 0 or frames == config.maximum_frames:
            print(
                f"[eval BCH+sparse] sigma={sigma_percent}% frames={frames:,} "
                f"bit_errors={bit_errors} frame_errors={frame_errors}",
                flush=True,
            )
        if bit_errors >= config.minimum_bit_errors:
            break

    result = metrics_record(
        "BCH+sparse", bit_errors, frame_errors, frames, payload_bits_per_frame=16
    )
    result.update(
        {
            "BCH_codeword_error_rate": bch_codeword_errors / frames,
            "BCH_codeword_errors": bch_codeword_errors,
            "sparse_block_class_error_rate": sparse_block_errors / (frames * 5),
            "sparse_block_message_BER": sparse_block_bit_errors / (frames * 5 * 7),
            "shaping_padding_by_3bit_suffix": ";".join(
                "".join(map(str, row.tolist())) for row in shaping_padding
            ),
            "shaping_selected_accuracy_mean": float(
                shaping_class_accuracy[shaping_indices].mean()
            ),
            "shaping_fixed_zero_accuracy_mean": float(
                shaping_class_accuracy[fixed_zero_indices].mean()
            ),
            "evaluation_seconds": time.time() - start_time,
            "stopped_by_bit_error_target": bit_errors >= config.minimum_bit_errors,
        }
    )
    return result


def simulate_bch15_sparse_decoders(
    model: nn.Module,
    channel: ChannelConfig,
    config: Config,
    sigma_percent: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compare cascaded FFNN+BCH against exact 27-resistance joint ML.

    Both decoders see exactly the same BCH(15,7) messages and channel samples:
    15 BCH bits + six fixed zeros -> three sparse 7/9 blocks -> 27 cells.
    """
    rng = stream_rng(config.seed, sigma_percent, stream_id=15)
    frames = 0
    sequential_bit_errors = 0
    sequential_frame_errors = 0
    sequential_decoder_failures = 0
    joint_bit_errors = 0
    joint_frame_errors = 0
    sequential_seconds = 0.0
    joint_seconds = 0.0
    started = time.time()

    while frames < config.maximum_frames:
        n = min(config.evaluation_batch_frames, config.maximum_frames - frames)
        message_indices = rng.integers(0, 128, size=n, dtype=np.int64)
        transmitted_messages = MESSAGE_BITS[message_indices]
        _, physical_bits = encode_bch15_sparse(transmitted_messages)
        raw27 = sample_resistance(physical_bits, channel, rng)

        decoder_started = time.perf_counter()
        hard_sparse_indices = ffnn_decode_indices(
            model, raw27.reshape(n * 3, 9)
        ).reshape(n, 3)
        sequential_messages, correctable = sequential_bch15_decode_from_sparse_indices(
            hard_sparse_indices
        )
        sequential_seconds += time.perf_counter() - decoder_started

        decoder_started = time.perf_counter()
        joint_indices = joint_ml_decode_indices(raw27, channel)
        joint_messages = MESSAGE_BITS[joint_indices]
        joint_seconds += time.perf_counter() - decoder_started

        sequential_errors = sequential_messages != transmitted_messages
        joint_errors = joint_messages != transmitted_messages
        sequential_bit_errors += int(sequential_errors.sum())
        sequential_frame_errors += int(np.any(sequential_errors, axis=1).sum())
        sequential_decoder_failures += int((~correctable).sum())
        joint_bit_errors += int(joint_errors.sum())
        joint_frame_errors += int(np.any(joint_errors, axis=1).sum())
        frames += n

        if frames % 100_000 == 0 or frames == config.maximum_frames:
            print(
                f"[eval BCH15+sparse] sigma={sigma_percent}% frames={frames:,} "
                f"sequential_errors={sequential_bit_errors} "
                f"joint_errors={joint_bit_errors}",
                flush=True,
            )
        if (
            sequential_bit_errors >= config.minimum_bit_errors
            and joint_bit_errors >= config.minimum_bit_errors
        ):
            break

    sequential = metrics_record(
        "BCH(15,7)+sparse sequential FFNN",
        sequential_bit_errors,
        sequential_frame_errors,
        frames,
        payload_bits_per_frame=7,
    )
    joint = metrics_record(
        "BCH(15,7)+sparse joint ML",
        joint_bit_errors,
        joint_frame_errors,
        frames,
        payload_bits_per_frame=7,
    )
    common = {
        "physical_bits_per_frame": 27,
        "BCH_n": 15,
        "BCH_k": 7,
        "BCH_t": 2,
        "padding_bits": 6,
        "sparse_blocks": 3,
        "same_received_samples": True,
        "evaluation_seconds_total": time.time() - started,
    }
    sequential.update(
        {
            **common,
            "decoder_seconds": sequential_seconds,
            "decoder_failure_rate": sequential_decoder_failures / frames,
            "stopped_by_bit_error_target": (
                sequential_bit_errors >= config.minimum_bit_errors
            ),
        }
    )
    joint.update(
        {
            **common,
            "decoder_seconds": joint_seconds,
            "joint_candidates": int(BCH15_PHYSICAL_CODEBOOK.shape[0]),
            "stopped_by_bit_error_target": joint_bit_errors >= config.minimum_bit_errors,
        }
    )
    return sequential, joint


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


def evaluate_five_methods(
    model: nn.Module,
    config: Config,
    sigma_percent: int,
    include_only_bch: bool = True,
) -> list[dict[str, Any]]:
    """Evaluate the established curves for one controlled channel point."""
    channel = channel_for_sigma(config, sigma_percent)
    without_coding = exact_without_coding(channel)
    slnn, only_sparse_ml = simulate_slnn_and_only_sparse_ml(
        model, channel, config, sigma_percent
    )
    bch_sparse = simulate_bch_sparse_ffnn(model, channel, config, sigma_percent)
    results = [slnn, without_coding, bch_sparse]
    if include_only_bch:
        results.append(simulate_only_bch(channel, config, sigma_percent))
    results.append(only_sparse_ml)
    return results


def result_by_method(
    method_results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {str(result["method"]): result for result in method_results}


def run_p1_sweep(
    args: argparse.Namespace,
    model: nn.Module,
    base_config: Config,
    output_dir: Path,
    model_path: Path,
) -> None:
    """Write the BER-vs-P1 table at fixed sigma_mu=10%."""
    sigma_percent = args.p1_sigma
    rows: list[dict[str, Any]] = []
    started = time.time()
    output_path = output_dir / f"all_curves_p1_sigma{sigma_percent}.csv"

    for p1 in args.p1_values:
        point_config = replace(base_config, p1=float(p1))
        method_results = evaluate_five_methods(
            model,
            point_config,
            sigma_percent,
            include_only_bch=not args.skip_only_bch,
        )
        by_method = result_by_method(method_results)
        row = {
            "sigma_mu": sigma_percent,
            "P1": float(p1),
            "SLNN [1] BER": by_method["SLNN [1]"]["BER"],
            "SLNN [1] FER": by_method["SLNN [1]"]["FER"],
            "without coding BER": by_method["without coding"]["BER"],
            "without coding FER": by_method["without coding"]["FER"],
            "BCH+sparse BER": by_method["BCH+sparse"]["BER"],
            "BCH+sparse FER": by_method["BCH+sparse"]["FER"],
            "only-sparse (ML decoding) BER": by_method[
                "only-sparse (ML decoding)"
            ]["BER"],
            "only-sparse (ML decoding) FER": by_method[
                "only-sparse (ML decoding)"
            ]["FER"],
        }
        if not args.skip_only_bch:
            row["only-BCH BER"] = by_method["only-BCH"]["BER"]
            row["only-BCH FER"] = by_method["only-BCH"]["FER"]
        rows.append(row)
        pd.DataFrame(rows).to_csv(output_path, index=False, float_format="%.12e")
        print("\nCurrent P1-sweep BER table:")
        print(pd.DataFrame(rows).to_string(index=False), flush=True)

    script_path = Path(__file__).resolve()
    manifest = {
        "description": "Five established BER curves versus P1 at fixed sigma_mu",
        "created_by_script": str(script_path),
        "script_sha256": file_sha256(script_path),
        "controlled_variable": "P1",
        "fixed_sigma_mu_percent": sigma_percent,
        "p1_values": [float(value) for value in args.p1_values],
        "curve_names": [str(result["method"]) for result in method_results],
        "only_bch_skipped": bool(args.skip_only_bch),
        "note": (
            "Figure 6 supplies only the sweep layout; every BER is recomputed "
            "from this repository's five existing method implementations."
        ),
        "common_random_numbers": (
            "The same deterministic streams are reused across P1 points to reduce "
            "Monte Carlo comparison variance."
        ),
        "config_except_p1": asdict(base_config),
        "checkpoint_file": {
            "path": str(model_path),
            "sha256": file_sha256(model_path),
        },
        "output_file": str(output_path),
        "elapsed_seconds": time.time() - started,
    }
    with (output_dir / f"run_manifest_p1_sigma{sigma_percent}.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(manifest, file, indent=2)


def run_joint_bch15_experiment(
    args: argparse.Namespace,
    model: nn.Module,
    config: Config,
    output_dir: Path,
    model_path: Path,
) -> None:
    """Run the controlled cascaded-versus-joint BCH(15,7)+sparse experiment."""
    rows: list[dict[str, Any]] = []
    started = time.time()
    output_path = output_dir / "joint_bch15_sparse27.csv"
    bch_description = validate_joint_bch15_sparse()

    for sigma_percent in args.sigmas:
        channel = channel_for_sigma(config, sigma_percent)
        sequential, joint = simulate_bch15_sparse_decoders(
            model, channel, config, sigma_percent
        )
        for result in (sequential, joint):
            rows.append(
                {
                    "P1": config.p1,
                    "sigma_mu_percent": sigma_percent,
                    **result,
                }
            )
        pd.DataFrame(rows).to_csv(output_path, index=False, float_format="%.12e")
        print("\nCurrent BCH(15,7)+sparse joint-decoder table:")
        print(
            pd.DataFrame(rows)[
                ["sigma_mu_percent", "method", "BER", "FER", "frames"]
            ].to_string(index=False),
            flush=True,
        )

    script_path = Path(__file__).resolve()
    manifest = {
        "description": (
            "Controlled BCH(15,7)+6 zero padding+3 sparse blocks experiment: "
            "cascaded FFNN/hard/BCH versus exact joint ML over 27 resistances"
        ),
        "created_by_script": str(script_path),
        "script_sha256": file_sha256(script_path),
        "config": asdict(config),
        "sigmas_percent": list(args.sigmas),
        "bch_sparse": bch_description,
        "hypothesis": "Joint 27-cell ML reduces payload BER/FER.",
        "controlled_comparison": (
            "Both decoders consume identical transmitted messages and resistance samples."
        ),
        "baseline": "global FFNN per 9-cell block -> hard 21 bits -> BCH(15,7)",
        "proposed": "argmax over 128 exact channel likelihoods using all 27 cells",
        "checkpoint_file": {
            "path": str(model_path),
            "sha256": file_sha256(model_path),
        },
        "output_file": str(output_path),
        "elapsed_seconds": time.time() - started,
    }
    with (output_dir / "run_manifest_joint_bch15_sparse27.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(manifest, file, indent=2)


def run_classical_distance_experiment(
    args: argparse.Namespace,
    config: Config,
    output_dir: Path,
) -> None:
    """Compare two non-learning sparse decoders on paired channel samples."""
    rows: list[dict[str, Any]] = []
    started = time.time()
    output_path = output_dir / "classical_distance_comparison.csv"
    covariance_diagonals: dict[str, list[float]] = {}

    for sigma_percent in args.sigmas:
        channel = channel_for_sigma(config, sigma_percent)
        euclidean, mahalanobis = simulate_classical_distance_decoders(
            channel, config, sigma_percent
        )
        covariance_diagonals[str(sigma_percent)] = np.diag(
            pooled_within_class_covariance(channel)
        ).tolist()
        for result in (euclidean, mahalanobis):
            rows.append(
                {
                    "P1": config.p1,
                    "sigma_mu_percent": sigma_percent,
                    **result,
                }
            )
        pd.DataFrame(rows).to_csv(output_path, index=False, float_format="%.12e")
        print("\nCurrent non-AI distance comparison:")
        print(
            pd.DataFrame(rows)[
                ["sigma_mu_percent", "method", "BER", "FER", "frames"]
            ].to_string(index=False),
            flush=True,
        )

    script_path = Path(__file__).resolve()
    manifest = {
        "description": "Paired non-AI Euclidean versus Mahalanobis sparse decoding",
        "created_by_script": str(script_path),
        "script_sha256": file_sha256(script_path),
        "hypothesis": "Channel-aware pooled Mahalanobis distance reduces BER/FER.",
        "controlled_comparison": (
            "Both decoders use the same analytical channel centroids, transmitted "
            "messages, resistance samples, stopping rules, and seed."
        ),
        "euclidean": "argmin_c (r-c)^T(r-c)",
        "mahalanobis": "argmin_c (r-c)^T Sigma^-1 (r-c)",
        "covariance": (
            "Analytical pooled within-class covariance. It is diagonal because "
            "the implemented channel samples cells conditionally independently."
        ),
        "covariance_diagonal_by_sigma": covariance_diagonals,
        "config": asdict(config),
        "sigmas_percent": list(args.sigmas),
        "output_file": str(output_path),
        "elapsed_seconds": time.time() - started,
        "uses_ai": False,
    }
    with (output_dir / "run_manifest_classical_distance.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(manifest, file, indent=2)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    training_sigmas = (
        tuple(args.finetune_train_sigmas) if args.finetune else tuple(args.train_sigmas)
    )
    config = Config(
        train_sigmas=training_sigmas,
        nr_train=args.finetune_nr_train if args.finetune else args.nr_train,
        training_chunk_size=args.training_chunk_size,
        train_batch_size=args.train_batch_size,
        max_epochs=args.finetune_epochs if args.finetune else args.max_epochs,
        patience=args.finetune_patience if args.finetune else 5,
        learning_rate=args.finetune_lr if args.finetune else 0.01,
        step_size=15 if args.finetune else 10,
        minimum_bit_errors=args.min_bit_errors,
        maximum_frames=args.max_frames,
        evaluation_batch_frames=args.batch_frames,
        shaping_calibration_samples=args.shaping_calibration_samples,
    )
    validate_codebook()
    bch_description = validate_bch()
    if args.classical_distance_only:
        run_classical_distance_experiment(args, config, output_dir)
        return
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
        finetune=args.finetune,
    )
    training_rows: list[dict[str, Any]] = [training_summary]

    if args.p1_sweep_only:
        run_p1_sweep(args, model, config, output_dir, model_path)
        return
    if args.joint_bch15_only:
        run_joint_bch15_experiment(args, model, config, output_dir, model_path)
        return

    for sigma_percent in args.sigmas:
        # Evaluation remains no-offset at each requested sigma.
        channel = channel_for_sigma(config, sigma_percent)

        without_coding = exact_without_coding(channel)
        only_bch = simulate_only_bch(channel, config, sigma_percent)
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
            "BCH+sparse": "BCH(31,16,t=3) + 4 calibrated shaping bits + 5 sparse blocks + FFNN + syndrome decoder",
            "only-BCH": "BCH(31,16,t=3) + threshold + syndrome decoder; Monte Carlo",
            "only-sparse (ML decoding)": "paper baseline sparse 7/9 + alpha=2.5 Euclidean decoder",
        },
        "payload_bits_per_frame": {"BCH+sparse": 16, "only-BCH": 16, "other_methods": 7},
        "physical_bits_per_frame": {
            "SLNN [1]": 9,
            "without coding": 7,
            "BCH+sparse": 45,
            "only-BCH": 31,
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
        "--p1-sweep-only",
        action="store_true",
        help=(
            "Generate all_curves_p1_sigma10.csv from the five existing curves "
            "without rerunning the sigma sweep."
        ),
    )
    parser.add_argument(
        "--joint-bch15-only",
        action="store_true",
        help=(
            "Compare BCH(15,7)+6 zero padding+3 sparse blocks using cascaded "
            "FFNN/hard/BCH and exact joint ML over all 27 resistances."
        ),
    )
    parser.add_argument(
        "--classical-distance-only",
        action="store_true",
        help=(
            "Compare pure Euclidean and pooled-covariance Mahalanobis sparse "
            "decoders on identical samples; neither decoder uses AI."
        ),
    )
    parser.add_argument(
        "--p1-values",
        type=float,
        nargs="+",
        default=list(P1_SWEEP),
        help="P1 values for --p1-sweep-only.",
    )
    parser.add_argument(
        "--p1-sigma",
        type=int,
        default=10,
        help="Fixed sigma_mu percentage for --p1-sweep-only; default: 10.",
    )
    parser.add_argument(
        "--skip-only-bch",
        action="store_true",
        help="Skip the only-BCH curve in a P1 sweep to reduce runtime.",
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
        "--shaping-calibration-samples",
        type=int,
        default=2_000,
        help="Noisy FFNN samples per sparse class used to choose shaping bits.",
    )
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
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Continue from the existing checkpoint using the harder fine-tune schedule.",
    )
    parser.add_argument("--finetune-nr-train", type=int, default=1_800_000)
    parser.add_argument("--finetune-epochs", type=int, default=60)
    parser.add_argument("--finetune-patience", type=int, default=10)
    parser.add_argument("--finetune-lr", type=float, default=1e-3)
    parser.add_argument(
        "--finetune-train-sigmas",
        type=float,
        nargs="+",
        default=[0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18],
        help="Harder sigma mixture used only with --finetune.",
    )
    parsed = parser.parse_args()
    if parsed.retrain and parsed.finetune:
        parser.error("--retrain and --finetune are mutually exclusive")
    exclusive_modes = sum(
        [
            parsed.p1_sweep_only,
            parsed.joint_bch15_only,
            parsed.classical_distance_only,
        ]
    )
    if exclusive_modes > 1:
        parser.error(
            "--p1-sweep-only, --joint-bch15-only, and "
            "--classical-distance-only are mutually exclusive"
        )
    invalid = [sigma for sigma in parsed.sigmas if not (1 <= sigma <= 100)]
    if invalid:
        parser.error(f"Invalid sigma percentages: {invalid}")
    if not (1 <= parsed.p1_sigma <= 100):
        parser.error("p1-sigma must be a percentage from 1 to 100")
    invalid_p1 = [p1 for p1 in parsed.p1_values if not (0.0 <= p1 <= 1.0)]
    if invalid_p1:
        parser.error(f"Invalid P1 probabilities: {invalid_p1}")
    invalid_train = [sigma for sigma in parsed.train_sigmas if not (0.0 < sigma < 1.0)]
    if invalid_train:
        parser.error(f"Invalid training sigma ratios: {invalid_train}")
    if parsed.nr_train <= 0 or parsed.training_chunk_size <= 0:
        parser.error("nr-train and training-chunk-size must be positive")
    if parsed.shaping_calibration_samples <= 0:
        parser.error("shaping-calibration-samples must be positive")
    if parsed.nr_train % len(parsed.train_sigmas) != 0:
        parser.error("nr-train must be divisible by the number of train-sigmas")
    invalid_finetune = [
        sigma for sigma in parsed.finetune_train_sigmas if not (0.0 < sigma < 1.0)
    ]
    if invalid_finetune:
        parser.error(f"Invalid fine-tune sigma ratios: {invalid_finetune}")
    if parsed.finetune_nr_train <= 0 or parsed.finetune_epochs <= 0:
        parser.error("fine-tune sample and epoch counts must be positive")
    if parsed.finetune_patience <= 0 or parsed.finetune_lr <= 0.0:
        parser.error("fine-tune patience and learning rate must be positive")
    if parsed.finetune_nr_train % len(parsed.finetune_train_sigmas) != 0:
        parser.error(
            "finetune-nr-train must be divisible by the number of fine-tune sigmas"
        )
    return parsed


if __name__ == "__main__":
    run(parse_args())
