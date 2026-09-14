"""
Unit tests for src.anomaly.detector: correct flagging behavior and
leakage-safe rolling computation.
"""

import numpy as np
import pytest

from src.anomaly.detector import compute_residuals, detect_anomalies, severity_label, AnomalyDetectionError


def test_compute_residuals_basic():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([90.0, 210.0, 280.0])
    residuals = compute_residuals(y_true, y_pred)
    np.testing.assert_allclose(residuals, [10.0, -10.0, 20.0])


def test_detect_anomalies_flags_obvious_outlier():
    rng = np.random.RandomState(0)
    normal_residuals = rng.normal(0, 5, 300)
    residuals = normal_residuals.copy()
    residuals[250] = 500.0  # inject an obvious, large anomaly

    result = detect_anomalies(residuals, window=168, z_threshold=3.0)

    assert result.loc[250, "is_anomaly"] == True
    # Most points should NOT be flagged (a sanity check against over-flagging)
    assert result["is_anomaly"].sum() < 10


def test_detect_anomalies_raises_on_insufficient_data():
    residuals = np.random.randn(50)  # fewer than window+1
    with pytest.raises(AnomalyDetectionError):
        detect_anomalies(residuals, window=168)


def test_detect_anomalies_first_window_never_flagged():
    """
    The first `window` rows lack sufficient rolling history and must
    never be flagged (NaN rolling stats -> is_anomaly=False, not an error).
    """
    residuals = np.random.RandomState(1).normal(0, 1, 200)
    result = detect_anomalies(residuals, window=168, z_threshold=3.0)

    assert not result.iloc[:168]["is_anomaly"].any()


@pytest.mark.parametrize(
    "z, expected",
    [(1.0, "Normal"), (3.5, "Moderate"), (4.5, "High"), (6.0, "Severe"), (-6.0, "Severe")],
)
def test_severity_label(z, expected):
    assert severity_label(z) == expected