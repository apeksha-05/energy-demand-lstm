"""
Attention-augmented LSTM for multivariate, multi-step demand forecasting.

Architecture:
    Input (batch, lookback, num_features)
          -> LSTM (returns hidden state at EVERY timestep, not just the last)
          -> Additive (Bahdanau-style) attention over all timesteps
          -> Context vector (weighted sum of all hidden states)
          -> Dense (hidden_size -> horizon)
          -> Prediction

Unlike the single-layer LSTM (Phase 8), which only uses the FINAL
timestep's hidden state, this model lets every timestep contribute to
the final prediction, weighted by a learned relevance score. This
removes the "single fixed-size vector must represent all 72 timesteps"
bottleneck inherent to using only h_n.
"""

import torch
import torch.nn as nn

from src.config import RNN_HIDDEN_SIZE, FORECAST_HORIZON


class AdditiveAttention(nn.Module):
    """
    Bahdanau-style additive attention over an LSTM's full output sequence.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        lstm_out : torch.Tensor, shape (batch, seq_len, hidden_size)
            Hidden states at EVERY timestep.

        Returns
        -------
        context : torch.Tensor, shape (batch, hidden_size)
            Weighted sum of all hidden states.
        attention_weights : torch.Tensor, shape (batch, seq_len)
            The learned α_t weights, summing to 1 across seq_len for
            each sample. Returned separately so we can visualize them.
        """
        # scores: (batch, seq_len, 1) -> squeeze to (batch, seq_len)
        scores = self.v(torch.tanh(self.W(lstm_out))).squeeze(-1)
        attention_weights = torch.softmax(scores, dim=1)  # normalize across timesteps

        # Weighted sum: (batch, seq_len, 1) * (batch, seq_len, hidden_size), summed over seq_len
        context = torch.sum(attention_weights.unsqueeze(-1) * lstm_out, dim=1)
        return context, attention_weights


class AttentionLSTMForecaster(nn.Module):
    """
    Single-layer LSTM followed by additive attention over its full output
    sequence, then a linear output layer.
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int = RNN_HIDDEN_SIZE,
        horizon: int = FORECAST_HORIZON,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_size, batch_first=True)
        self.attention = AdditiveAttention(hidden_size)
        self.output_layer = nn.Linear(hidden_size, horizon)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, lookback, num_features)
        return_attention : bool
            If True, also return the attention weights (for visualization).

        Returns
        -------
        torch.Tensor, shape (batch, horizon), or
        tuple[torch.Tensor, torch.Tensor] if return_attention=True
        """
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size) — EVERY timestep
        context, attention_weights = self.attention(lstm_out)
        predictions = self.output_layer(context)

        if return_attention:
            return predictions, attention_weights
        return predictions