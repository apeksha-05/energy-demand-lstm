"""
Simple (vanilla) RNN model for multivariate, multi-step demand forecasting.

Architecture:
    Input (batch, lookback, num_features)
          -> nn.RNN(hidden_size)
          -> take final timestep's hidden state
          -> nn.Linear(hidden_size -> horizon)
          -> Prediction (batch, horizon)

This is a deliberately simple baseline neural sequence model. Its purpose
in this project is pedagogical as much as predictive: to demonstrate, by
direct comparison with LSTM in Phase 8, why vanilla RNNs struggle with
long-term dependencies (vanishing gradients over a 168-step sequence).
"""

import torch
import torch.nn as nn

from src.config import RNN_HIDDEN_SIZE, FORECAST_HORIZON


class SimpleRNNForecaster(nn.Module):
    """
    A single-layer vanilla RNN followed by a linear output layer that
    maps the final hidden state directly to all `horizon` forecast steps
    (direct multi-output forecasting, as decided in Phase 5).
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int = RNN_HIDDEN_SIZE,
        horizon: int = FORECAST_HORIZON,
    ) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=num_features,
            hidden_size=hidden_size,
            batch_first=True,  # input shape: (batch, seq_len, features)
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
        # rnn_out: (batch, lookback, hidden_size) — hidden state at EVERY timestep
        # h_n: (1, batch, hidden_size) — hidden state at the FINAL timestep only
        rnn_out, h_n = self.rnn(x)

        # h_n's first dimension is num_layers * num_directions (=1 here);
        # squeeze it to get (batch, hidden_size)
        final_hidden = h_n.squeeze(0)

        return self.output_layer(final_hidden)