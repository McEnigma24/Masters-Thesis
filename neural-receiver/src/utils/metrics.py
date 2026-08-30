"""Metryki link-level: BER i BLER."""

from __future__ import annotations

import torch


def ber(bits: torch.Tensor, bits_hat: torch.Tensor) -> float:
    """Bit Error Rate między transmisją a odbiorem."""
    return (bits != bits_hat).float().mean().item()


def bler(bits: torch.Tensor, bits_hat: torch.Tensor) -> float:
    """Block Error Rate — błąd jeśli choć jeden bit w bloku jest zły."""
    block_errors = (bits != bits_hat).any(dim=-1).float()
    return block_errors.mean().item()
