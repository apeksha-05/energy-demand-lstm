"""
Residual-based anomaly detection using a dynamic (rolling) z-score
threshold.

Rationale: electricity demand has time-varying volatility (e.g., rapid
morning ramp-up vs. stable overnight trough), so a single fixed
threshold across the whole series would either miss anomalies during
calm periods or over-flag normal variation during volatile periods.
A rolling mean/std of the RESIDUALS (not raw demand) adapts to local
error volatility.

LEAKAGE RULE: the rolling window at time t uses only residuals from
BEFORE t (shifted by 1), matching the same discipline used for
rolling demand features in feature_engineering.py.
"""

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

DEFAULT_ROLLING_WINDOW = 168  # one week of hourly residuals
DEFAULT_Z_THRESHOLD = 3.0


class AnomalyDetectionError(Exception):
    """Raised when anomaly detection receives invalid input."""


def compute_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute forecasting residuals: actual - predicted.

    Positive residual = model under-predicted (actual demand was higher
    than expected). Negative residual = model over-predicted.
    """
    return y_true - y_pred


def detect_anomalies(
    residuals: np.ndarray,
    window: int = DEFAULT_ROLLING_WINDOW,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
) -> pd.DataFrame:
    """
    Flag anomalies using a rolling z-score of the residuals.

    Parameters
    ----------
    residuals : np.ndarray, shape (n,)
        Forecasting residuals (actual - predicted), in chronological order.
    window : int
        Rolling window size (in observations) used to compute local
        mean/std of residuals.
    z_threshold : float
        Number of rolling standard deviations beyond which a residual
        is flagged as anomalous.

    Returns
    -------
    pd.DataFrame
        Columns: residual, rolling_mean, rolling_std, z_score, is_anomaly.
        The first `window` rows will have NaN rolling stats (insufficient
        history) and are never flagged (is_anomaly=False) rather than
        producing unreliable early z-scores.
    """
    if len(residuals) < window + 1:
        raise AnomalyDetectionError(
            f"Need at least {window + 1} residuals, got {len(residuals)}"
        )

    series = pd.Series(residuals)
    shifted = series.shift(1)  # exclude current point from its own rolling stats

    rolling_mean = shifted.rolling(window=window).mean()
    rolling_std = shifted.rolling(window=window).std()

    z_score = (series - rolling_mean) / rolling_std
    is_anomaly = (z_score.abs() > z_threshold).fillna(False)

    result = pd.DataFrame({
        "residual": series,
        "rolling_mean": rolling_mean,
        "rolling_std": rolling_std,
        "z_score": z_score,
        "is_anomaly": is_anomaly,
    })

    n_anomalies = int(result["is_anomaly"].sum())
    logger.info(
        "Anomaly detection: %d/%d points flagged (%.2f%%) at |z| > %.1f, window=%d",
        n_anomalies, len(result), 100 * n_anomalies / len(result), z_threshold, window,
    )
    return result


def severity_label(z_score: float) -> str:
    """
    Convert a z-score magnitude into a human-readable severity label,
    for dashboard display.
    """
    abs_z = abs(z_score)
    if abs_z > 5.0:
        return "Severe"
    elif abs_z > 4.0:
        return "High"
    elif abs_z > 3.0:
        return "Moderate"
    return "Normal"