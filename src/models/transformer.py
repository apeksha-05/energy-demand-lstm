"""
Compact encoder-only Transformer for multivariate, multi-step demand
forecasting.

Unlike RNN/LSTM/GRU, which process timesteps sequentially, this model
uses self-attention to let every timestep attend to every other timestep
in parallel — sidestepping vanishing gradients for long-range
dependencies at the cost of quadratic (O(n^2)) attention computation
versus recurrence's linear (O(n)) cost.

Positional encoding is added explicitly since self-attention has no
inherent notion of sequence order (unlike recurrence, which processes
timesteps in order by construction).
"""

import math

import torch
import torch.nn as nn

from src.config import (
    TRANSFORMER_D_MODEL,
    TRANSFORMER_NUM_HEADS,
    TRANSFORMER_NUM_LAYERS,
    TRANSFORMER_DIM_FEEDFORWARD,
    FORECAST_HORIZON,
)


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (Vaswani et al., 2017).
    Injects position information since self-attention has no built-in
    notion of sequence order.
    """

    def __init__(self, d_model: int, max_len: int = 500) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        return x + self.pe[:, : x.size(1), :]


class TransformerForecaster(nn.Module):
    """
    Linear projection -> positional encoding -> Transformer encoder
    layer(s) -> mean pooling across timesteps -> linear output layer.
    """

    def __init__(
        self,
        num_features: int,
        d_model: int = TRANSFORMER_D_MODEL,
        num_heads: int = TRANSFORMER_NUM_HEADS,
        num_layers: int = TRANSFORMER_NUM_LAYERS,
        dim_feedforward: int = TRANSFORMER_DIM_FEEDFORWARD,
        horizon: int = FORECAST_HORIZON,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(num_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, lookback, num_features)

        Returns
        -------
        torch.Tensor, shape (batch, horizon)
        """
        x = self.input_projection(x)          # (batch, seq_len, d_model)
        x = self.positional_encoding(x)        # add position info
        x = self.transformer_encoder(x)        # (batch, seq_len, d_model) — self-attention applied

        pooled = x.mean(dim=1)                 # (batch, d_model) — mean pooling across all timesteps
        return self.output_layer(pooled)