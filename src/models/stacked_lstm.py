"""
Stacked (multi-layer) LSTM model for multivariate, multi-step demand
forecasting.

Uses PyTorch's native num_layers support, which correctly handles
return_sequences internally: intermediate layers automatically receive
the FULL hidden-state sequence from the layer below them, while only
the final layer's last-timestep hidden state is exposed as h_n.

Dropout is applied BETWEEN stacked layers (not within a single layer's
recurrent connections) to reduce overfitting from the added capacity.
"""

import torch
import torch.nn as nn

from src.config import RNN_HIDDEN_SIZE, FORECAST_HORIZON, LSTM_NUM_LAYERS, LSTM_DROPOUT


class StackedLSTMForecaster(nn.Module):
    """
    A multi-layer LSTM (default 2 layers) with inter-layer dropout,
    followed by a linear output layer mapping the final layer's final
    hidden state to all `horizon` forecast steps.
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int = RNN_HIDDEN_SIZE,
        num_layers: int = LSTM_NUM_LAYERS,
        dropout: float = LSTM_DROPOUT,
        horizon: int = FORECAST_HORIZON,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,  # PyTorch warns if dropout>0 with 1 layer
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
        # h_n shape: (num_layers, batch, hidden_size) — one final hidden
        # state PER LAYER. We want only the LAST layer's final hidden state.
        lstm_out, (h_n, c_n) = self.lstm(x)

        final_layer_hidden = h_n[-1]  # (batch, hidden_size) — last layer, final timestep
        return self.output_layer(final_layer_hidden)