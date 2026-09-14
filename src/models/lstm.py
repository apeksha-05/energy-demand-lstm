"""
Single-layer LSTM model for multivariate, multi-step demand forecasting.

Architecture:
    Input (batch, lookback, num_features)
          -> nn.LSTM(hidden_size)
          -> take final timestep's hidden state
          -> nn.Linear(hidden_size -> horizon)
          -> Prediction (batch, horizon)

Structurally identical in shape to SimpleRNNForecaster (Phase 7) —
the ONLY difference is nn.LSTM's internal gating mechanism (forget,
input, output gates + cell state) versus nn.RNN's single tanh update.
This makes the RNN-vs-LSTM comparison a controlled experiment: same
architecture shape, same training loop, same data.
"""

import torch
import torch.nn as nn

from src.config import RNN_HIDDEN_SIZE, FORECAST_HORIZON


class LSTMForecaster(nn.Module):
    """
    A single-layer LSTM followed by a linear output layer that maps the
    final hidden state to all `horizon` forecast steps (direct
    multi-output forecasting).
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
        # lstm_out: (batch, lookback, hidden_size) — hidden state at every timestep
        # h_n: (1, batch, hidden_size) — final hidden state
        # c_n: (1, batch, hidden_size) — final CELL state (LSTM's extra memory,
        #      not used directly for output here, but this is where the
        #      long-term memory actually lives internally)
        lstm_out, (h_n, c_n) = self.lstm(x)

        final_hidden = h_n.squeeze(0)
        return self.output_layer(final_hidden)