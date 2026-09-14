"""
Bidirectional LSTM — a CONTROLLED EXPERIMENT, not a production candidate.

IMPORTANT: This model processes the lookback window (already-observed
historical data) in both directions, which is safe because the entire
window is genuinely historical at prediction time. It is deliberately
NOT used as this project's final forecasting model:
  1. Real-time deployment only ever has past data available — there is
     no legitimate "future" beyond the input window to look backward
     from, so the backward pass adds compute cost without adding new
     real-world information a forward-only model couldn't eventually
     learn from the same window.
  2. Building forecasting pipelines around bidirectional processing
     normalizes a pattern that becomes genuinely dangerous (temporal
     leakage) the moment window boundaries are handled less carefully
     than they are in this project's sequence_builder.py.

We build and measure it anyway, as instructed, to demonstrate empirically
whether/how much it helps on the (safe) historical window, and to make
its practical limitations for forecasting concrete rather than abstract.
"""

import torch
import torch.nn as nn

from src.config import RNN_HIDDEN_SIZE, FORECAST_HORIZON


class BiLSTMForecaster(nn.Module):
    """
    A single-layer Bidirectional LSTM followed by a linear output layer.
    The forward and backward final hidden states are concatenated before
    being passed to the output layer.
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int = RNN_HIDDEN_SIZE,
        horizon: int = FORECAST_HORIZON,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        # hidden_size * 2 because forward and backward hidden states are concatenated
        self.output_layer = nn.Linear(hidden_size * 2, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, lookback, num_features)

        Returns
        -------
        torch.Tensor, shape (batch, horizon)
        """
        # h_n shape: (num_directions, batch, hidden_size) = (2, batch, hidden_size)
        # h_n[0] = forward direction's final hidden state (at the LAST timestep)
        # h_n[1] = backward direction's final hidden state (at the FIRST timestep,
        #          since it processed the sequence in reverse)
        lstm_out, (h_n, c_n) = self.lstm(x)

        forward_final = h_n[0]   # (batch, hidden_size)
        backward_final = h_n[1]  # (batch, hidden_size)
        combined = torch.cat([forward_final, backward_final], dim=1)  # (batch, hidden_size*2)

        return self.output_layer(combined)