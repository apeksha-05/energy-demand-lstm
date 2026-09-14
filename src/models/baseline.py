"""
Baseline forecasting strategies: naive persistence, moving average, and
linear regression. All operate at the same lookback/horizon granularity
as the sequence-based deep learning models, so comparisons are fair.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.config import DEFAULT_LOOKBACK, FORECAST_HORIZON, TARGET_COL
from src.sequence_builder import get_last_known_target
from src.utils import get_logger

logger = get_logger(__name__)


def naive_forecast(
    df: pd.DataFrame,
    target_column: str = TARGET_COL,
    lookback: int = DEFAULT_LOOKBACK,
    horizon: int = FORECAST_HORIZON,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Persistence baseline: predict that all `horizon` future hours equal
    the most recently observed actual value.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        y_true shape (num_sequences, horizon), y_pred shape (num_sequences, horizon).
    """
    n = len(df)
    num_sequences = n - lookback - horizon + 1
    target_values = df[target_column].values

    last_known = get_last_known_target(df, target_column, lookback, horizon)
    y_pred = np.repeat(last_known.reshape(-1, 1), horizon, axis=1)

    y_true = np.array(
        [target_values[i + lookback : i + lookback + horizon] for i in range(num_sequences)],
        dtype=np.float32,
    )
    return y_true, y_pred


def moving_average_forecast(
    df: pd.DataFrame,
    target_column: str = TARGET_COL,
    lookback: int = DEFAULT_LOOKBACK,
    horizon: int = FORECAST_HORIZON,
    window: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Moving-average baseline: predict that all `horizon` future hours
    equal the mean of the last `window` observed hours before the
    forecast start.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        y_true shape (num_sequences, horizon), y_pred shape (num_sequences, horizon).
    """
    n = len(df)
    num_sequences = n - lookback - horizon + 1
    target_values = df[target_column].values

    y_pred_list = []
    for i in range(num_sequences):
        window_end = i + lookback  # exclusive, matches "last known" position + 1
        window_start = window_end - window
        avg = target_values[window_start:window_end].mean()
        y_pred_list.append(avg)

    y_pred = np.repeat(np.array(y_pred_list, dtype=np.float32).reshape(-1, 1), horizon, axis=1)

    y_true = np.array(
        [target_values[i + lookback : i + lookback + horizon] for i in range(num_sequences)],
        dtype=np.float32,
    )
    return y_true, y_pred


class LinearRegressionForecaster:
    """
    A real learned baseline: uses only the FINAL timestep's feature vector
    from each input window (not the full flattened sequence) as input to
    scikit-learn's LinearRegression, which natively supports multi-output
    regression (predicting all `horizon` steps at once from one model).

    Using only the last timestep is a deliberate design choice: our
    engineered features already include lag_1/24/48/168 and rolling
    statistics, so the most recent row already encodes relevant history.
    Flattening the full lookback window into a linear model is both
    computationally impractical (SVD cost scales cubically with feature
    count) and not how a linear baseline would realistically be deployed.
    """

    def __init__(self) -> None:
        self.model = LinearRegression()
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionForecaster":
        """
        Parameters
        ----------
        X : np.ndarray, shape (num_samples, lookback, num_features)
        y : np.ndarray, shape (num_samples, horizon)
        """
        X_last_step = X[:, -1, :]  # only the most recent timestep's features
        self.model.fit(X_last_step, y)
        self._is_fitted = True
        logger.info(
            "LinearRegressionForecaster fit on %d samples, %d features (last timestep only)",
            X_last_step.shape[0], X_last_step.shape[1],
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : np.ndarray, shape (num_samples, lookback, num_features)

        Returns
        -------
        np.ndarray, shape (num_samples, horizon)
        """
        if not self._is_fitted:
            raise RuntimeError("Call .fit() before .predict().")
        X_last_step = X[:, -1, :]
        return self.model.predict(X_last_step)