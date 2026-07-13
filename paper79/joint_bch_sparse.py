"""BCH(15,7,5) + three sparse blocks with exact 27-cell joint ML decoding."""
from __future__ import annotations

import itertools
import math

import numpy as np

from .channel import ChannelConfig, combined_transition_probabilities
from .codebook import CODEBOOK, MESSAGE_BITS


# Primitive narrow-sense binary BCH(15,7,5), MSB-first.
# g(x) = x^8 + x^7 + x^6 + x^4 + 1.
BCH15_GENERATOR = np.asarray([1, 1, 1, 0, 1, 0, 0, 0, 1], dtype=np.uint8)
BCH15_BIT_WEIGHTS = 1 << np.arange(14, -1, -1, dtype=np.uint16)
BCH15_SYNDROME_WEIGHTS = 1 << np.arange(7, -1, -1, dtype=np.uint16)


def bits7_to_indices(bits: np.ndarray) -> np.ndarray:
    """Convert MSB-first arrays shaped ``[..., 7]`` to values 0..127."""
    arr = np.asarray(bits, dtype=np.uint8)
    if arr.ndim < 1 or arr.shape[-1] != 7:
        raise ValueError("Expected final dimension of length 7")
    weights = 1 << np.arange(6, -1, -1, dtype=np.uint8)
    return (arr.astype(np.uint16) @ weights.astype(np.uint16)).astype(np.int64)


def bch15_encode(messages: np.ndarray) -> np.ndarray:
    """Systematically encode one or more 7-bit BCH messages into 15 bits."""
    source = np.asarray(messages, dtype=np.uint8)
    one_message = source.ndim == 1
    if one_message:
        source = source[None, :]
    if source.ndim != 2 or source.shape[1] != 7:
        raise ValueError("BCH(15,7) messages must have shape [N,7]")

    work = np.concatenate(
        [source, np.zeros((source.shape[0], 8), dtype=np.uint8)], axis=1
    )
    for position in range(7):
        rows = work[:, position].astype(bool)
        if np.any(rows):
            work[rows, position : position + 9] ^= BCH15_GENERATOR
    encoded = np.concatenate([source, work[:, 7:]], axis=1)
    return encoded[0] if one_message else encoded


def bch15_syndromes(vectors: np.ndarray) -> np.ndarray:
    """Return the 8-bit polynomial remainder for each length-15 vector."""
    work = np.asarray(vectors, dtype=np.uint8).copy()
    if work.ndim != 2 or work.shape[1] != 15:
        raise ValueError("BCH(15,7) vectors must have shape [N,15]")
    for position in range(7):
        rows = work[:, position].astype(bool)
        if np.any(rows):
            work[rows, position : position + 9] ^= BCH15_GENERATOR
    return (work[:, 7:].astype(np.uint16) @ BCH15_SYNDROME_WEIGHTS).astype(np.uint8)


def _build_bch15_error_lookup() -> tuple[np.ndarray, np.ndarray]:
    masks = np.zeros(1 << 8, dtype=np.uint16)
    correctable = np.zeros(1 << 8, dtype=bool)
    correctable[0] = True
    for weight in (1, 2):
        positions_at_weight = list(itertools.combinations(range(15), weight))
        errors = np.zeros((len(positions_at_weight), 15), dtype=np.uint8)
        for row, positions in enumerate(positions_at_weight):
            errors[row, list(positions)] = 1
        syndromes = bch15_syndromes(errors)
        if np.any(correctable[syndromes]):
            raise AssertionError("BCH(15,7) syndromes collide at weight <= 2")
        masks[syndromes] = errors.astype(np.uint16) @ BCH15_BIT_WEIGHTS
        correctable[syndromes] = True
    return masks, correctable


BCH15_ERROR_MASKS, BCH15_CORRECTABLE_SYNDROMES = _build_bch15_error_lookup()
BCH15_MESSAGES = MESSAGE_BITS.copy()
BCH15_CODEBOOK = bch15_encode(BCH15_MESSAGES)
BCH15_PADDED_BITS = np.concatenate(
    [BCH15_CODEBOOK, np.zeros((128, 6), dtype=np.uint8)], axis=1
)
BCH15_SPARSE_INDICES = bits7_to_indices(BCH15_PADDED_BITS.reshape(128, 3, 7))
BCH15_SPARSE_CODEWORDS = CODEBOOK[BCH15_SPARSE_INDICES]
BCH15_PHYSICAL_CODEBOOK = BCH15_SPARSE_CODEWORDS.reshape(128, 27)


def bch15_decode(received: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bounded-distance decode; uncorrectable words are returned unchanged."""
    arr = np.asarray(received, dtype=np.uint8)
    if arr.ndim != 2 or arr.shape[1] != 15:
        raise ValueError("BCH(15,7) received words must have shape [N,15]")
    syndromes = bch15_syndromes(arr)
    values = arr.astype(np.uint16) @ BCH15_BIT_WEIGHTS
    corrected_values = values ^ BCH15_ERROR_MASKS[syndromes]
    corrected = (
        (corrected_values[:, None] >> np.arange(14, -1, -1, dtype=np.uint16)) & 1
    ).astype(np.uint8)
    return corrected[:, :7], BCH15_CORRECTABLE_SYNDROMES[syndromes]


def encode_bch15_sparse(messages: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return padded logical blocks ``[N,3,7]`` and physical bits ``[N,27]``."""
    encoded = bch15_encode(messages)
    if encoded.ndim == 1:
        encoded = encoded[None, :]
    padded = np.concatenate(
        [encoded, np.zeros((encoded.shape[0], 6), dtype=np.uint8)], axis=1
    )
    blocks = padded.reshape(-1, 3, 7)
    sparse_indices = bits7_to_indices(blocks)
    return blocks, CODEBOOK[sparse_indices].reshape(-1, 27)


def _normal_logpdf(y: np.ndarray, mean: float, std: float) -> np.ndarray:
    if std <= 0.0:
        raise ValueError("Resistance standard deviations must be positive")
    return -0.5 * ((y - mean) / std) ** 2 - math.log(std * math.sqrt(2.0 * math.pi))


def cell_log_likelihoods(
    resistance: np.ndarray, channel: ChannelConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-cell ``log p(y|x=0)`` and ``log p(y|x=1)`` exactly."""
    y = np.asarray(resistance, dtype=np.float64)
    if y.ndim != 2 or y.shape[1] != 27:
        raise ValueError("Joint decoder expects resistance with shape [N,27]")
    p0, q0, p1, q1 = combined_transition_probabilities(channel)
    state0 = _normal_logpdf(y, channel.mu0, channel.sigma0)
    state1 = _normal_logpdf(
        y, channel.effective_mu1, channel.effective_sigma1
    )

    def weighted_log(probability: float, log_density: np.ndarray) -> np.ndarray:
        if probability == 0.0:
            return np.full_like(log_density, -np.inf)
        return math.log(probability) + log_density

    log_p_y_given_0 = np.logaddexp(
        weighted_log(q0, state0), weighted_log(p0, state1)
    )
    log_p_y_given_1 = np.logaddexp(
        weighted_log(p1, state0), weighted_log(q1, state1)
    )
    return log_p_y_given_0, log_p_y_given_1


def joint_ml_decode_indices(
    resistance: np.ndarray, channel: ChannelConfig
) -> np.ndarray:
    """Select one of 128 BCH+sparse messages from all 27 observations."""
    log0, log1 = cell_log_likelihoods(resistance, channel)
    # score(m) = sum_i log p(y_i|0) + C_mi * (log p(y_i|1)-log p(y_i|0)).
    # The first term is common to every candidate and can be dropped.
    scores = (log1 - log0) @ BCH15_PHYSICAL_CODEBOOK.T.astype(np.float64)
    return scores.argmax(axis=1).astype(np.int64)


def sequential_bch15_decode_from_sparse_indices(
    decoded_sparse_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Baseline: three hard sparse decisions followed by BCH decoding."""
    indices = np.asarray(decoded_sparse_indices, dtype=np.int64)
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("Sparse decisions must have shape [N,3]")
    hard21 = MESSAGE_BITS[indices].reshape(-1, 21)
    return bch15_decode(hard21[:, :15])


def validate_joint_bch15_sparse() -> dict[str, int]:
    """Validate the code parameters and the 27-cell candidate construction."""
    if BCH15_CODEBOOK.shape != (128, 15):
        raise AssertionError("BCH(15,7) codebook shape mismatch")
    if not np.array_equal(BCH15_CODEBOOK[:, :7], BCH15_MESSAGES):
        raise AssertionError("BCH(15,7) encoder is not systematic")
    d_min = int(BCH15_CODEBOOK[1:].sum(axis=1).min())
    if d_min != 5:
        raise AssertionError(f"Expected BCH(15,7) d_min=5, obtained {d_min}")
    if np.any(BCH15_PADDED_BITS[:, 15:] != 0):
        raise AssertionError("The six shaping bits must be fixed zeros")
    if np.unique(BCH15_PHYSICAL_CODEBOOK, axis=0).shape[0] != 128:
        raise AssertionError("Physical BCH+sparse candidates must be unique")
    return {
        "n": 15,
        "k": 7,
        "d_min": d_min,
        "t": 2,
        "padding_bits": 6,
        "sparse_blocks": 3,
        "resistance_observations": 27,
        "joint_candidates": 128,
    }


validate_joint_bch15_sparse()
