"""Prosty CNN Neural Receiver — zastępuje estymację kanału + equalizację + demapping."""

from __future__ import annotations

import torch
import torch.nn as nn


class NeuralReceiverCNN(nn.Module):
    """
    Wejście:  odebrany resource grid [B, 2, T, F]  (real/imag)
    Wyjście: LLR [B, num_bits]

    Zastępuje łańcuch:
        Channel Estimation → Equalization → Demapper
    """

    def __init__(self, num_bits: int, hidden_channels: int = 32):
        super().__init__()
        self.num_bits = num_bits
        self.conv = nn.Sequential(
            nn.Conv2d(2, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_channels, num_bits)

    def forward(self, y_grid: torch.Tensor) -> torch.Tensor:
        # y_grid: [B, 2, T, F]
        features = self.conv(y_grid)  # [B, C, T, F]
        pooled = features.mean(dim=(2, 3))  # global average pooling
        return self.head(pooled)

    @staticmethod
    def grid_to_tensor(y: torch.Tensor) -> torch.Tensor:
        """Konwertuje zespolony resource grid na tensor [B, 2, T, F]."""
        real = y.real
        imag = y.imag
        # y shape: [B, num_tx, num_streams, T, F] — bierzemy pierwszy tx/stream
        if y.ndim == 5:
            real = real[:, 0, 0]
            imag = imag[:, 0, 0]
        stacked = torch.stack([real, imag], dim=1)
        return stacked.float()
