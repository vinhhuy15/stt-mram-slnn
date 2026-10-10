"""Cascaded BAC + read-disturb Z-channel + Gaussian resistance model."""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Literal

import numpy as np
from scipy.stats import norm

ReadDirection = Literal["write0", "write1"]


@dataclass(frozen=True)
class ChannelConfig:
    """Physical/channel parameters, expressed in kOhm units."""

    mu0: float = 1.0
    mu1: float = 2.0
    sigma_ratio: float = 0.09
    p1: float = 2e-4
    p0_over_p1: float = 1e-2
    pr_over_p1: float = 1e-2
    read_direction: ReadDirection = "write0"
    offset_mean: float = 0.0
    offset_sigma_over_mu1: float = 0.0

    @property
    def p0(self) -> float:
        return self.p1 * self.p0_over_p1

    @property
    def pr(self) -> float:
        return self.p1 * self.pr_over_p1

    @property
    def sigma0(self) -> float:
        return self.mu0 * self.sigma_ratio

    @property
    def sigma1(self) -> float:
        return self.mu1 * self.sigma_ratio

    @property
    def offset_sigma(self) -> float:
        return self.mu1 * self.offset_sigma_over_mu1

    @property
    def effective_mu1(self) -> float:
        # The offset is additive and occurs only for the HRS (logical 1).
        return self.mu1 + self.offset_mean

    @property
    def effective_sigma1(self) -> float:
        # Sum of independent Gaussian read variation and Gaussian offset.
        return sqrt(self.sigma1**2 + self.offset_sigma**2)

    @property
    def threshold(self) -> float:
        return 0.5 * (self.mu0 + self.mu1)


def combined_transition_probabilities(cfg: ChannelConfig) -> tuple[float, float, float, float]:
    """Return (p0, q0, p1, q1) for the combined binary channel.

    Definitions follow the arrows in Fig. 3:
      p0 = Pr(final state=1 | target bit=0), q0 = 1-p0
      p1 = Pr(final state=0 | target bit=1), q1 = 1-p1

    The factors P0/2 and P1/2 account for an equiprobable previous cell state.
    """
    P0, P1, Pr = cfg.p0, cfg.p1, cfg.pr
    if cfg.read_direction == "write0":
        p0 = (P0 / 2.0) * (1.0 - Pr)
        q0 = (1.0 - P0 / 2.0) + (P0 / 2.0) * Pr
        p1 = (P1 / 2.0) + (1.0 - P1 / 2.0) * Pr
        q1 = (1.0 - P1 / 2.0) * (1.0 - Pr)
    elif cfg.read_direction == "write1":
        p0 = (P0 / 2.0) + (1.0 - P0 / 2.0) * Pr
        q0 = (1.0 - P0 / 2.0) * (1.0 - Pr)
        p1 = (P1 / 2.0) * (1.0 - Pr)
        q1 = (1.0 - P1 / 2.0) + (P1 / 2.0) * Pr
    else:
        raise ValueError(f"Unsupported read direction: {cfg.read_direction}")

    # Numerical and transcription sanity checks.
    if not (0.0 <= p0 <= 1.0 and 0.0 <= p1 <= 1.0):
        raise ValueError("Invalid crossover probability")
    if abs((p0 + q0) - 1.0) > 1e-12 or abs((p1 + q1) - 1.0) > 1e-12:
        raise ValueError("Combined transition probabilities do not sum to one")
    return p0, q0, p1, q1


def sample_resistance(
    target_bits: np.ndarray,
    cfg: ChannelConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample the continuous channel output for a target-bit array.

    This samples the *combined* BAC/Z transition first and then the GMC.
    The result is mathematically equivalent to simulating the two binary
    subchannels sequentially.
    """
    x = np.asarray(target_bits, dtype=np.uint8)
    if x.ndim != 2:
        raise ValueError("target_bits must be a 2-D array")

    p0, _, p1, _ = combined_transition_probabilities(cfg)
    u = rng.random(x.shape)
    final_state = x.copy()
    final_state[(x == 0) & (u < p0)] = 1
    final_state[(x == 1) & (u < p1)] = 0

    mean = np.where(final_state == 0, cfg.mu0, cfg.effective_mu1)
    std = np.where(final_state == 0, cfg.sigma0, cfg.effective_sigma1)
    return mean + rng.standard_normal(x.shape) * std


def conditional_detector_error_probabilities(cfg: ChannelConfig) -> tuple[float, float]:
    """Exact hard-threshold error probabilities (e0, e1).

    e0 = Pr(detector says 1 | target bit 0)
    e1 = Pr(detector says 0 | target bit 1)
    """
    p0, q0, p1, q1 = combined_transition_probabilities(cfg)
    t = cfg.threshold

    # Detector event probabilities conditioned on the final resistance state.
    state0_to_1 = float(norm.sf((t - cfg.mu0) / cfg.sigma0))
    state0_to_0 = 1.0 - state0_to_1
    state1_to_0 = float(norm.cdf((t - cfg.effective_mu1) / cfg.effective_sigma1))
    state1_to_1 = 1.0 - state1_to_0

    e0 = q0 * state0_to_1 + p0 * state1_to_1
    e1 = p1 * state0_to_0 + q1 * state1_to_0
    return e0, e1
