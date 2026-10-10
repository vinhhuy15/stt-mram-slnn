"""Plot styles matching Figures 6, 7, and 8 in the paper."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_figure6(df: pd.DataFrame, output_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot(df["p1"], df["decoder_ber"], "k-d", linewidth=1.6, markersize=5,
            markerfacecolor="white", label="proposed code, decoder output")
    ax.plot(df["p1"], df["detector_ber"], "b-o", linewidth=1.6, markersize=5,
            markerfacecolor="white", label="proposed code, detector output")
    ax.plot(df["p1"], df["raw_ber"], "r-^", linewidth=1.6, markersize=5,
            label="raw data w/o coding, detector output")
    ax.set_xscale("log")
    ax.set_xlim(1e-8, 1e-3)
    ax.set_ylim(0.0, 2.0e-3)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3), useMathText=True)
    ax.set_xlabel(r"$P_1$")
    ax.set_ylabel("BER")
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    ax.grid(False)
    fig.tight_layout()
    _save(fig, output_base)


def plot_figure7(df: pd.DataFrame, output_base: Path) -> None:
    x = 100.0 * df["sigma_ratio"]
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.semilogy(x, df["decoder_ber"], "k-^", linewidth=1.6, markersize=6,
                markerfacecolor="white", label="BER - decoder output")
    ax.semilogy(x, df["detector_ber"], "b-o", linewidth=1.6, markersize=6,
                markerfacecolor="white", label="BER - detector output")
    ax.semilogy(x, df["raw_ber"], "r-*", linewidth=1.6, markersize=8,
                label="BER - raw data w/o coding, detector output")
    ax.semilogy(x, df["decoder_fer"], "k--^", linewidth=1.6, markersize=6,
                markerfacecolor="white", label="FER - decoder output")
    ax.semilogy(x, df["detector_fer"], "b--o", linewidth=1.6, markersize=6,
                markerfacecolor="white", label="FER - detector output")
    ax.semilogy(x, df["raw_fer"], "r--*", linewidth=1.6, markersize=8,
                label="FER - raw data w/o coding, detector output")
    ax.set_xlim(8, 15)
    ax.set_ylim(1e-4, 1.1)
    ax.set_xticks(range(8, 16))
    ax.set_xlabel(r"$\sigma_0/\mu_0$ (%)")
    ax.set_ylabel("BER & FER")
    ax.legend(loc="upper left", fontsize=7.5, frameon=True)
    ax.grid(False)
    fig.tight_layout()
    _save(fig, output_base)


def plot_figure8(df: pd.DataFrame, output_base: Path) -> None:
    x = 100.0 * df["sigma_ratio"]
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.semilogy(x, df["raw_ber"], "k-^", linewidth=1.6, markersize=6,
                markerfacecolor="white", label="BER - raw data w/o coding, detector output")
    ax.semilogy(x, df["detector_ber"], "b-o", linewidth=1.6, markersize=6,
                markerfacecolor="white", label="BER - proposed code, detector output")
    ax.semilogy(x, df["decoder_ber"], "g-s", linewidth=1.6, markersize=5,
                markerfacecolor="white", label="BER - proposed code, decoder output")
    ax.semilogy(x, df["raw_fer"], "k--^", linewidth=1.6, markersize=6,
                markerfacecolor="white", label="FER - raw data w/o coding, detector output")
    ax.semilogy(x, df["detector_fer"], "b--o", linewidth=1.6, markersize=6,
                markerfacecolor="white", label="FER - proposed code, detector output")
    ax.semilogy(x, df["decoder_fer"], "g--s", linewidth=1.6, markersize=5,
                markerfacecolor="white", label="FER - proposed code, decoder output")
    ax.set_xlim(2, 10)
    ax.set_ylim(1e-4, 1.1)
    ax.set_xticks(range(2, 11))
    ax.set_xlabel(r"$\sigma_0/\mu_0$ (%)")
    ax.set_ylabel("BER & FER")
    ax.legend(loc="upper left", fontsize=7.0, frameon=True)
    ax.grid(False)
    fig.tight_layout()
    _save(fig, output_base)
