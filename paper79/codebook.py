"""Exact 7/9 sparse-code LUT from Table 1 of Nguyen (IEEE Access, 2021).

The paper states that codewords c0..c127 are read top-to-bottom, then
left-to-right across the four table columns. That is the order below.
"""
from __future__ import annotations

import numpy as np

CODEWORD_STRINGS: tuple[str, ...] = (
    # Table 1, column 1: c0..c31
    "111100000", "000000011", "111010000", "000000101",
    "111001000", "000000110", "111000100", "000001001",
    "111000010", "000001010", "111000001", "000001100",
    "110110000", "000001111", "110101000", "000010001",
    "110100100", "000010010", "110100010", "000010100",
    "110100001", "000010111", "110011000", "000011000",
    "110010100", "000011011", "110010010", "000011101",
    "110010001", "000011110", "110001100", "000100001",
    # Table 1, column 2: c32..c63
    "110001010", "000100010", "110001001", "000100100",
    "110000110", "000100111", "110000101", "000101000",
    "110000011", "000101011", "110000000", "000101101",
    "101110000", "000101110", "101101000", "000110000",
    "101100100", "000110011", "101100010", "000110101",
    "101100001", "000110110", "101011000", "000111001",
    "101010100", "000111010", "101010010", "000111100",
    "101010001", "001000001", "101001100", "001000010",
    # Table 1, column 3: c64..c95
    "101001010", "001000100", "101001001", "001000111",
    "101000110", "001001000", "101000101", "001001011",
    "101000011", "001001101", "101000000", "001010000",
    "100110100", "001010011", "100110010", "001010101",
    "100110001", "001010110", "100101100", "001011001",
    "100101010", "001011010", "100101001", "001011100",
    "100100110", "001100000", "100100101", "001100011",
    "100100011", "001100101", "100100000", "001100110",
    # Table 1, column 4: c96..c127
    "100011100", "001101001", "100011010", "001110001",
    "100010101", "001110010", "100010011", "001110100",
    "100010000", "010000001", "100001101", "010000010",
    "100001011", "010000100", "100001000", "010000111",
    "100000111", "010001000", "100000100", "010001011",
    "100000010", "010001101", "100000001", "010010000",
    "011101000", "010010011", "011100100", "010010101",
    "010011001", "011011000", "010100000", "011000000",
)


def _bits(s: str) -> list[int]:
    return [int(ch) for ch in s]


CODEBOOK: np.ndarray = np.asarray([_bits(s) for s in CODEWORD_STRINGS], dtype=np.uint8)
MESSAGE_BITS: np.ndarray = np.asarray(
    [_bits(f"{i:07b}") for i in range(128)], dtype=np.uint8
)
CODEWORD_WEIGHTS: np.ndarray = CODEBOOK.sum(axis=1)


def validate_codebook() -> None:
    """Fail loudly if the LUT was transcribed or reordered incorrectly."""
    assert CODEBOOK.shape == (128, 9)
    assert MESSAGE_BITS.shape == (128, 7)
    assert len(set(CODEWORD_STRINGS)) == 128
    assert set(CODEWORD_WEIGHTS.tolist()) == {2, 4}
    assert int(np.sum(CODEWORD_WEIGHTS == 2)) == 36
    assert int(np.sum(CODEWORD_WEIGHTS == 4)) == 92

    # The paper says all C(9,2)=36 weight-2 words are included.
    all_weight2 = {
        f"{x:09b}" for x in range(1 << 9) if f"{x:09b}".count("1") == 2
    }
    selected_weight2 = {s for s in CODEWORD_STRINGS if s.count("1") == 2}
    assert selected_weight2 == all_weight2


validate_codebook()
