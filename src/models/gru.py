"""
GRU (Gated Recurrent Unit) model for multivariate, multi-step demand
forecasting.

Structurally identical in shape to LSTMForecaster (Phase 8) — the only
difference is nn.GRU's simpler 2-gate mechanism (update, reset) versus
nn.LSTM's 3-gate mechanism (forget, input, output) plus separate cell
state. This makes the LSTM-vs-GRU comparison a controlled experiment.
"""

import torch
import torch.nn as nn

from src.config import RNN_HIDDEN_SIZE, FORECAST_HORIZON


class GRUForecaster(nn.Module):
    """
    A single-layer GRU followed by a linear output layer mapping the
    final hidden state to all `horizon` forecast steps.
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int = RNN_HIDDEN_SIZE,
        horizon: int = FORECAST_HORIZON,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, lookback, num_features)

        Returns
        -------
        torch.Tensor, shape (batch, horizon)
        """
        # GRU returns only (output, h_n) — no cell state, unlike LSTM
        gru_out, h_n = self.gru(x)
        final_hidden = h_n.squeeze(0)
        return self.output_layer(final_hidden)