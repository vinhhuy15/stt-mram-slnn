import itertools

import numpy as np

from paper79.channel import ChannelConfig
from paper79.codebook import MESSAGE_BITS
from paper79.joint_bch_sparse import (
    BCH15_CODEBOOK,
    BCH15_PADDED_BITS,
    BCH15_PHYSICAL_CODEBOOK,
    bch15_decode,
    encode_bch15_sparse,
    joint_ml_decode_indices,
    validate_joint_bch15_sparse,
)


def test_code_and_sparse_shapes() -> None:
    description = validate_joint_bch15_sparse()
    assert description == {
        "n": 15,
        "k": 7,
        "d_min": 5,
        "t": 2,
        "padding_bits": 6,
        "sparse_blocks": 3,
        "resistance_observations": 27,
        "joint_candidates": 128,
    }
    assert BCH15_PADDED_BITS.shape == (128, 21)
    assert BCH15_PHYSICAL_CODEBOOK.shape == (128, 27)


def test_bch_corrects_every_error_pattern_up_to_weight_two() -> None:
    messages = MESSAGE_BITS[[0, 1, 42, 127]]
    transmitted = BCH15_CODEBOOK[[0, 1, 42, 127]]
    patterns = [()] + list(itertools.combinations(range(15), 1))
    patterns += list(itertools.combinations(range(15), 2))
    received = np.repeat(transmitted, len(patterns), axis=0)
    expected = np.repeat(messages, len(patterns), axis=0)
    for message_row in range(len(messages)):
        offset = message_row * len(patterns)
        for pattern_row, positions in enumerate(patterns):
            received[offset + pattern_row, list(positions)] ^= 1
    decoded, correctable = bch15_decode(received)
    assert np.all(correctable)
    np.testing.assert_array_equal(decoded, expected)


def test_encoder_appends_six_zeros_before_three_sparse_blocks() -> None:
    blocks, physical = encode_bch15_sparse(MESSAGE_BITS)
    assert blocks.shape == (128, 3, 7)
    assert physical.shape == (128, 27)
    np.testing.assert_array_equal(blocks.reshape(128, 21)[:, 15:], 0)
    np.testing.assert_array_equal(physical, BCH15_PHYSICAL_CODEBOOK)


def test_joint_ml_decodes_all_candidates_at_state_means() -> None:
    channel = ChannelConfig(sigma_ratio=0.01, p1=0.0)
    resistance = np.where(
        BCH15_PHYSICAL_CODEBOOK == 0, channel.mu0, channel.effective_mu1
    )
    decoded = joint_ml_decode_indices(resistance, channel)
    np.testing.assert_array_equal(decoded, np.arange(128))
