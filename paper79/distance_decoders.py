"""Classical nearest-centroid decoders for the 7/9 sparse code."""
from __future__ import annotations

import numpy as np

from .channel import ChannelConfig, combined_transition_probabilities
from .codebook import CODEBOOK


def channel_state_moments(cfg: ChannelConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return conditional resistance means and variances for target bits 0/1."""
    p0, q0, p1, q1 = combined_transition_probabilities(cfg)
    means = np.asarray(
        [
            q0 * cfg.mu0 + p0 * cfg.effective_mu1,
            p1 * cfg.mu0 + q1 * cfg.effective_mu1,
        ],
        dtype=np.float64,
    )
    variances = np.asarray(
        [
            q0 * (cfg.sigma0**2 + (cfg.mu0 - means[0]) ** 2)
            + p0
            * (cfg.effective_sigma1**2 + (cfg.effective_mu1 - means[0]) ** 2),
            p1 * (cfg.sigma0**2 + (cfg.mu0 - means[1]) ** 2)
            + q1
            * (cfg.effective_sigma1**2 + (cfg.effective_mu1 - means[1]) ** 2),
        ],
        dtype=np.float64,
    )
    return means, variances


def sparse_centroids(cfg: ChannelConfig) -> np.ndarray:
    """Map each binary sparse codeword to its channel-conditional centroid."""
    means, _ = channel_state_moments(cfg)
    return means[CODEBOOK]


def pooled_within_class_covariance(cfg: ChannelConfig) -> np.ndarray:
    """Analytical pooled covariance of residuals around the 128 centroids.

    Cell outputs are conditionally independent in the repository channel model,
    so off-diagonal entries are zero. Diagonal entries can differ because the
    Table-1 codebook has a different fraction of ones at each cell position.
    """
    _, variances = channel_state_moments(cfg)
    ones_fraction = CODEBOOK.mean(axis=0, dtype=np.float64)
    diagonal = (1.0 - ones_fraction) * variances[0] + ones_fraction * variances[1]
    if np.any(diagonal <= 0.0):
        raise ValueError("Mahalanobis covariance must be positive definite")
    return np.diag(diagonal)


def _squared_distances(
    observations: np.ndarray,
    centroids: np.ndarray,
    inverse_diagonal: np.ndarray,
) -> np.ndarray:
    weighted_observations = observations * inverse_diagonal
    return (
        np.sum(weighted_observations * observations, axis=1, keepdims=True)
        - 2.0 * weighted_observations @ centroids.T
        + np.sum(centroids * centroids * inverse_diagonal, axis=1)[None, :]
    )


def euclidean_decode_indices(raw: np.ndarray, cfg: ChannelConfig) -> np.ndarray:
    """Decode with pure squared Euclidean distance to physical centroids."""
    observations = np.asarray(raw, dtype=np.float64).reshape(-1, 9)
    centroids = sparse_centroids(cfg)
    distances = _squared_distances(observations, centroids, np.ones(9))
    return np.argmin(distances, axis=1).astype(np.int64)


def mahalanobis_decode_indices(raw: np.ndarray, cfg: ChannelConfig) -> np.ndarray:
    """Decode with d^2=(r-c)^T Sigma^-1 (r-c), without any learned model."""
    observations = np.asarray(raw, dtype=np.float64).reshape(-1, 9)
    centroids = sparse_centroids(cfg)
    covariance = pooled_within_class_covariance(cfg)
    inverse_diagonal = 1.0 / np.diag(covariance)
    distances = _squared_distances(observations, centroids, inverse_diagonal)
    return np.argmin(distances, axis=1).astype(np.int64)
