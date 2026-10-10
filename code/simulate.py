"""Exact detector metrics and Monte-Carlo Euclidean-decoder metrics."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .channel import ChannelConfig, conditional_detector_error_probabilities, sample_resistance
from .codebook import CODEBOOK, MESSAGE_BITS


_TREE = cKDTree(CODEBOOK.astype(np.float64))


def exact_detector_metrics(cfg: ChannelConfig) -> dict[str, float]:
    """Compute exact BER/FER for raw and coded hard-threshold outputs."""
    e0, e1 = conditional_detector_error_probabilities(cfg)

    bit_error_by_symbol = np.where(CODEBOOK == 0, e0, e1)
    coded_ber = float(bit_error_by_symbol.mean())
    coded_fer = float(np.mean(1.0 - np.prod(1.0 - bit_error_by_symbol, axis=1)))

    raw_ber = 0.5 * (e0 + e1)
    raw_fer = 1.0 - (1.0 - raw_ber) ** 7

    return {
        "e0": e0,
        "e1": e1,
        "detector_ber": coded_ber,
        "detector_fer": coded_fer,
        "raw_ber": raw_ber,
        "raw_fer": raw_fer,
    }


def _ci95_from_batch_means(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    if len(arr) < 2:
        return mean, mean
    half = 1.96 * float(arr.std(ddof=1)) / np.sqrt(len(arr))
    return max(0.0, mean - half), min(1.0, mean + half)


def simulate_decoder_metrics(
    cfg: ChannelConfig,
    *,
    n_frames: int,
    seed: int,
    alpha: float = 2.5,
    chunk_size: int = 100_000,
    workers: int = -1,
) -> dict[str, Any]:
    """Simulate the paper's Euclidean LUT decoder.

    The paper compares binary codewords to the attenuated continuous output.
    In kOhm units with mu0=1, this is y/alpha. For unit invariance we use
    y/(alpha*mu0), which reduces exactly to the paper setting.
    """
    if n_frames <= 0 or chunk_size <= 0:
        raise ValueError("n_frames and chunk_size must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    rng = np.random.default_rng(seed)
    total_bit_errors = 0
    total_frame_errors = 0
    ber_batches: list[float] = []
    fer_batches: list[float] = []

    done = 0
    while done < n_frames:
        m = min(chunk_size, n_frames - done)
        source_index = rng.integers(0, 128, size=m, dtype=np.int16)
        tx_codeword = CODEBOOK[source_index]
        y = sample_resistance(tx_codeword, cfg, rng)

        decoder_input = y / (alpha * cfg.mu0)
        _, decoded_index = _TREE.query(decoder_input, k=1, workers=workers)
        decoded_index = np.asarray(decoded_index, dtype=np.int16)

        bit_errors = MESSAGE_BITS[decoded_index] != MESSAGE_BITS[source_index]
        n_bit_errors = int(bit_errors.sum())
        n_frame_errors = int(np.any(bit_errors, axis=1).sum())
        total_bit_errors += n_bit_errors
        total_frame_errors += n_frame_errors
        ber_batches.append(n_bit_errors / (m * 7.0))
        fer_batches.append(n_frame_errors / m)
        done += m

    decoder_ber = total_bit_errors / (n_frames * 7.0)
    decoder_fer = total_frame_errors / n_frames
    ber_lo, ber_hi = _ci95_from_batch_means(ber_batches)
    fer_lo, fer_hi = _ci95_from_batch_means(fer_batches)

    return {
        "decoder_ber": decoder_ber,
        "decoder_fer": decoder_fer,
        "decoder_bit_errors": total_bit_errors,
        "decoder_frame_errors": total_frame_errors,
        "decoder_ber_ci95_low": ber_lo,
        "decoder_ber_ci95_high": ber_hi,
        "decoder_fer_ci95_low": fer_lo,
        "decoder_fer_ci95_high": fer_hi,
        "n_frames": n_frames,
        "seed": seed,
        "alpha": alpha,
        "channel": asdict(cfg),
    }


def evaluate_point(
    cfg: ChannelConfig,
    *,
    n_frames: int,
    seed: int,
    alpha: float = 2.5,
    chunk_size: int = 100_000,
    workers: int = -1,
) -> dict[str, Any]:
    result = exact_detector_metrics(cfg)
    result.update(
        simulate_decoder_metrics(
            cfg,
            n_frames=n_frames,
            seed=seed,
            alpha=alpha,
            chunk_size=chunk_size,
            workers=workers,
        )
    )
    return result
