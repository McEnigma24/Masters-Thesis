"""Wykresy używane w lekcjach."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_constellation(
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    title: str = "Konstelacja QAM",
) -> None:
    """Rysuje punkty konstelacji (nadane i opcjonalnie odebrane)."""
    x_np = x.detach().cpu().numpy().reshape(-1)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x_np.real, x_np.imag, alpha=0.4, label="TX", s=8)
    if y is not None:
        y_np = y.detach().cpu().numpy().reshape(-1)
        ax.scatter(y_np.real, y_np.imag, alpha=0.4, label="RX", s=8)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()


def plot_ber_curve(
    snr_db: np.ndarray,
    ber_values: np.ndarray,
    label: str = "BER",
) -> None:
    """Krzywa BER vs SNR (skala logarytmiczna)."""
    plt.figure(figsize=(7, 5))
    plt.semilogy(snr_db, ber_values, "o-", label=label)
    plt.xlabel("Eb/N0 [dB]")
    plt.ylabel("BER")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
