"""
Evaluation metrics for time-series forecasting.

All functions operate on original-scale (inverse-transformed) values,
never on scaled [0,1] values — MAE/RMSE in scaled units are not
interpretable or comparable to real-world MW.
"""

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error, in the same units as the input (MW)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error. Penalizes large errors more than MAE."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-3) -> float:
    """
    Mean Absolute Percentage Error, as a percentage.
    epsilon avoids division by zero; irrelevant here since demand is
    always well above zero, but included defensively.
    """
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-3) -> float:
    """
    Symmetric Mean Absolute Percentage Error, as a percentage.
    Bounded in [0, 200], more robust than MAPE to small actual values.
    """
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon
    return float(np.mean(numerator / denominator) * 100)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of determination. Reported for context only — see module
    docstring in this project's docs about why R² alone is misleading
    for seasonal time series.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute every metric at once. Accepts y_true/y_pred of any matching
    shape (flattened internally), e.g. (n_samples, horizon).

    Returns
    -------
    dict[str, float]
        Keys: "MAE", "RMSE", "MAPE", "sMAPE", "R2".
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    return {
        "MAE": mae(y_true_flat, y_pred_flat),
        "RMSE": rmse(y_true_flat, y_pred_flat),
        "MAPE": mape(y_true_flat, y_pred_flat),
        "sMAPE": smape(y_true_flat, y_pred_flat),
        "R2": r_squared(y_true_flat, y_pred_flat),
    }