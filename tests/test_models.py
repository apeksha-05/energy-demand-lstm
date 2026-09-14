"""
Unit tests for model construction and prediction shapes across all
architectures — confirms every model accepts the same input shape and
produces the expected output shape, without requiring any real training.
"""

import torch
import pytest
import numpy as np
from src.models.baseline import LinearRegressionForecaster
from src.models.rnn import SimpleRNNForecaster
from src.models.lstm import LSTMForecaster
from src.models.stacked_lstm import StackedLSTMForecaster
from src.models.bilstm import BiLSTMForecaster
from src.models.gru import GRUForecaster
from src.models.attention_lstm import AttentionLSTMForecaster
from src.models.transformer import TransformerForecaster

BATCH_SIZE = 4
LOOKBACK = 24
NUM_FEATURES = 10
HORIZON = 6


@pytest.mark.parametrize(
    "model_class",
    [SimpleRNNForecaster, LSTMForecaster, StackedLSTMForecaster, BiLSTMForecaster, GRUForecaster, AttentionLSTMForecaster, TransformerForecaster],
)
def test_neural_model_output_shape(model_class):
    model = model_class(num_features=NUM_FEATURES, horizon=HORIZON) if model_class != TransformerForecaster else model_class(num_features=NUM_FEATURES, horizon=HORIZON)
    x = torch.randn(BATCH_SIZE, LOOKBACK, NUM_FEATURES)

    output = model(x)

    assert output.shape == (BATCH_SIZE, HORIZON)


def test_attention_lstm_returns_valid_attention_weights():
    model = AttentionLSTMForecaster(num_features=NUM_FEATURES, horizon=HORIZON)
    x = torch.randn(BATCH_SIZE, LOOKBACK, NUM_FEATURES)

    predictions, attention_weights = model(x, return_attention=True)

    assert predictions.shape == (BATCH_SIZE, HORIZON)
    assert attention_weights.shape == (BATCH_SIZE, LOOKBACK)
    # Attention weights must sum to ~1 across the lookback dimension for each sample
    sums = attention_weights.sum(dim=1)
    assert torch.allclose(sums, torch.ones(BATCH_SIZE), atol=1e-5)


def test_linear_regression_forecaster_output_shape():
    import numpy as np

    model = LinearRegressionForecaster()
    X_train = np.random.randn(50, LOOKBACK, NUM_FEATURES).astype(np.float32)
    y_train = np.random.randn(50, HORIZON).astype(np.float32)

    model.fit(X_train, y_train)
    predictions = model.predict(X_train[:BATCH_SIZE])

    assert predictions.shape == (BATCH_SIZE, HORIZON)


def test_linear_regression_forecaster_raises_before_fit():
    model = LinearRegressionForecaster()
    X = np.zeros((1, LOOKBACK, NUM_FEATURES), dtype="float32")
    with pytest.raises(RuntimeError):
        model.predict(X)