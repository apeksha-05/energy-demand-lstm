"""
Sequence generation for recurrent/attention/transformer time-series models.

Converts a flat, chronologically-ordered feature dataframe into 3D arrays
suitable for sequence models:
    X shape: (num_samples, lookback, num_features)
    y shape: (num_samples, horizon)          [horizon=1 for single-step]

LEAKAGE RULE: sequences must be built SEPARATELY per split (train/val/test),
never on the concatenated full dataset before splitting. A sequence whose
lookback window would reach across a split boundary is dropped entirely,
never truncated or filled — this guarantees no sequence's input or target
crosses from one split into another.
"""

import numpy as np
import pandas as pd

from src.config import DEFAULT_LOOKBACK, FORECAST_HORIZON, TARGET_COL
from src.utils import get_logger

logger = get_logger(__name__)


class SequenceBuilderError(Exception):
    """Raised when sequence generation receives invalid input or parameters."""


def create_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = TARGET_COL,
    lookback: int = DEFAULT_LOOKBACK,
    horizon: int = FORECAST_HORIZON,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window of length `lookback` across `df` to build (X, y) pairs
    for direct multi-output forecasting.

    For a window ending at row index i (inclusive):
        X[i] = df[feature_columns].values[i - lookback + 1 : i + 1]   # past `lookback` hours
        y[i] = df[target_column].values[i + 1 : i + 1 + horizon]      # next `horizon` hours

    Parameters
    ----------
    df : pd.DataFrame
        A SINGLE split (train, val, or test) — never the full unsplit
        dataset. Must already be chronologically sorted with no gaps.
    feature_columns : list[str]
        Columns to use as model input at each timestep.
    target_column : str
        Column to forecast.
    lookback : int
        Number of past hours used as input context.
    horizon : int
        Number of future hours to predict (1 = single-step).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        X of shape (num_samples, lookback, num_features),
        y of shape (num_samples, horizon).
    """
    if lookback < 1 or horizon < 1:
        raise SequenceBuilderError("lookback and horizon must both be >= 1")

    missing = set(feature_columns + [target_column]) - set(df.columns)
    if missing:
        raise SequenceBuilderError(f"Missing required columns: {missing}")

    n = len(df)
    min_required = lookback + horizon
    if n < min_required:
        raise SequenceBuilderError(
            f"Split has only {n} rows, but lookback={lookback} + horizon={horizon} "
            f"= {min_required} rows are required to build even one sequence."
        )

    feature_values = df[feature_columns].values
    target_values = df[target_column].values

    num_sequences = n - lookback - horizon + 1
    num_features = len(feature_columns)

    X = np.zeros((num_sequences, lookback, num_features), dtype=np.float32)
    y = np.zeros((num_sequences, horizon), dtype=np.float32)

    for i in range(num_sequences):
        X[i] = feature_values[i : i + lookback]
        y[i] = target_values[i + lookback : i + lookback + horizon]

    logger.info(
        "Built %d sequences: X=%s, y=%s (lookback=%d, horizon=%d, dropped %d boundary rows)",
        num_sequences, X.shape, y.shape, lookback, horizon,
        n - num_sequences - lookback - horizon + 1 + (lookback + horizon - 1),
    )
    return X, y


def create_sequences_for_all_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = TARGET_COL,
    lookback: int = DEFAULT_LOOKBACK,
    horizon: int = FORECAST_HORIZON,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Convenience wrapper: builds (X, y) sequences independently for each
    of train/val/test, guaranteeing no sequence spans a split boundary.

    Returns
    -------
    dict[str, tuple[np.ndarray, np.ndarray]]
        Keys "train", "val", "test", each mapping to an (X, y) tuple.
    """
    results = {}
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        X, y = create_sequences(
            split_df, feature_columns, target_column, lookback, horizon
        )
        results[name] = (X, y)
        logger.info("Split '%s': X=%s, y=%s", name, X.shape, y.shape)
    return results

def get_last_known_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COL,
    lookback: int = DEFAULT_LOOKBACK,
    horizon: int = FORECAST_HORIZON,
) -> np.ndarray:
    """
    For each sequence window that create_sequences() would generate from
    this split, return the target value at the LAST input timestep
    (i.e., the most recent actual observation before the forecast starts).

    Used by naive/moving-average baselines, which need "the current
    actual value" rather than a learned feature representation.

    Returns
    -------
    np.ndarray
        Shape (num_sequences,) — aligned index-for-index with the X, y
        arrays create_sequences() would produce for the same df/lookback/horizon.
    """
    n = len(df)
    num_sequences = n - lookback - horizon + 1
    target_values = df[target_column].values

    # last input timestep of window i is at row index (i + lookback - 1)
    last_known = np.array(
        [target_values[i + lookback - 1] for i in range(num_sequences)],
        dtype=np.float32,
    )
    return last_known