import numpy as np

from paper79.channel import ChannelConfig
from paper79.codebook import CODEBOOK
from paper79.distance_decoders import (
    euclidean_decode_indices,
    mahalanobis_decode_indices,
    pooled_within_class_covariance,
    sparse_centroids,
)


def test_covariance_is_positive_diagonal_and_not_scalar_identity() -> None:
    covariance = pooled_within_class_covariance(ChannelConfig(sigma_ratio=0.10))
    assert covariance.shape == (9, 9)
    assert np.allclose(covariance, np.diag(np.diag(covariance)))
    assert np.all(np.diag(covariance) > 0.0)
    assert np.ptp(np.diag(covariance)) > 0.0


def test_both_classical_decoders_recover_all_noiseless_centroids() -> None:
    cfg = ChannelConfig(sigma_ratio=0.10)
    centroids = sparse_centroids(cfg)
    expected = np.arange(len(CODEBOOK), dtype=np.int64)
    assert np.array_equal(euclidean_decode_indices(centroids, cfg), expected)
    assert np.array_equal(mahalanobis_decode_indices(centroids, cfg), expected)


def test_distance_decoders_accept_multiple_batches_as_flat_rows() -> None:
    cfg = ChannelConfig(sigma_ratio=0.12)
    raw = sparse_centroids(cfg)[:6].reshape(2, 3, 9)
    assert euclidean_decode_indices(raw, cfg).shape == (6,)
    assert mahalanobis_decode_indices(raw, cfg).shape == (6,)
