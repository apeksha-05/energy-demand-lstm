"""
Data cleaning and chronological splitting for the Panama electricity
load dataset.

This module is intentionally light on "fixing" the data, because Phase 2
EDA confirmed the raw dataset has no duplicate timestamps, no missing
timestamps, and no missing values. Its job is instead to:
  1. Parse and index the datetime column.
  2. Flag (not remove) known extreme demand events for later reference.
  3. Split the data chronologically into train/validation/test.
  4. Fit scalers on the training split only, and apply them everywhere.

CRITICAL: scalers must be fit ONLY on the training split. Fitting on the
full dataset before splitting leaks information about the future
(validation/test statistics) into training — this is one of the most
common time-series mistakes and is explicitly forbidden here.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import DATETIME_COL, TARGET_COL
from src.utils import get_logger

logger = get_logger(__name__)


class PreprocessingError(Exception):
    """Raised when cleaning or splitting encounters an invalid state."""


@dataclass
class SplitDatasets:
    """Container for chronologically split dataframes."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def clean_raw_data(
    df: pd.DataFrame,
    anomaly_z_threshold: float = 4.0,
) -> pd.DataFrame:
    """
    Parse timestamps, set the datetime index, and add a diagnostic
    `is_extreme_event` flag for values far outside the normal range.

    The flag is for later reference (error analysis, anomaly-detection
    validation) ONLY. It is never used as a model input feature, and
    flagged rows are never removed — Phase 2 confirmed these are
    genuine grid events, not corrupted data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe as returned by load_raw_data().
    anomaly_z_threshold : float
        Number of standard deviations from the mean beyond which a
        demand value is flagged as an extreme event (diagnostic only).

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe, datetime-indexed and sorted chronologically,
        with an added `is_extreme_event` boolean column.
    """
    if DATETIME_COL not in df.columns:
        raise PreprocessingError(f"Expected datetime column '{DATETIME_COL}' not found.")

    df = df.copy()
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])
    df = df.sort_values(DATETIME_COL).reset_index(drop=True)
    df = df.set_index(DATETIME_COL)

    n_duplicates = df.index.duplicated().sum()
    if n_duplicates > 0:
        raise PreprocessingError(
            f"Found {n_duplicates} duplicate timestamps. "
            f"This dataset was expected to have none — investigate before proceeding."
        )

    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h")
    missing_ts = full_range.difference(df.index)
    if len(missing_ts) > 0:
        raise PreprocessingError(
            f"Found {len(missing_ts)} missing timestamps. "
            f"This dataset was expected to be fully continuous — investigate before proceeding."
        )

    mean_demand = df[TARGET_COL].mean()
    std_demand = df[TARGET_COL].std()
    z_scores = (df[TARGET_COL] - mean_demand) / std_demand
    df["is_extreme_event"] = z_scores.abs() > anomaly_z_threshold

    n_flagged = df["is_extreme_event"].sum()
    logger.info(
        "Cleaning complete: %d rows, %d flagged as diagnostic extreme events (|z| > %.1f)",
        len(df), n_flagged, anomaly_z_threshold,
    )
    return df


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> SplitDatasets:
    """
    Split a time-indexed dataframe chronologically into train/val/test.

    NEVER shuffles. Train is the earliest portion, validation the middle
    portion, and test the latest portion — this ensures the model is
    always evaluated on data that is chronologically AFTER what it was
    trained on, which mirrors real-world deployment (forecasting the
    future from the past) and prevents the model from ever training on
    information that would not have been available at prediction time.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned, datetime-indexed dataframe, already sorted chronologically.
    train_frac : float
        Fraction of rows assigned to training (from the start).
    val_frac : float
        Fraction of rows assigned to validation (immediately after train).
        The remainder is assigned to test.

    Returns
    -------
    SplitDatasets
        Named container with .train, .val, .test dataframes.
    """
    if not df.index.is_monotonic_increasing:
        raise PreprocessingError(
            "Dataframe is not chronologically sorted. Refusing to split — "
            "this would silently produce an invalid time-series split."
        )
    if train_frac + val_frac >= 1.0:
        raise PreprocessingError("train_frac + val_frac must be < 1.0")

    n = len(df)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    logger.info(
        "Chronological split: train=%d (%s to %s), val=%d (%s to %s), test=%d (%s to %s)",
        len(train_df), train_df.index.min(), train_df.index.max(),
        len(val_df), val_df.index.min(), val_df.index.max(),
        len(test_df), test_df.index.min(), test_df.index.max(),
    )
    return SplitDatasets(train=train_df, val=val_df, test=test_df)


def fit_scaler(train_df: pd.DataFrame, columns: list[str]) -> MinMaxScaler:
    """
    Fit a MinMaxScaler on the TRAINING split only.

    Parameters
    ----------
    train_df : pd.DataFrame
        The training split. Must never be validation or test data.
    columns : list[str]
        Numeric columns to scale.

    Returns
    -------
    MinMaxScaler
        Fitted scaler, ready to .transform() train/val/test/inference data.
    """
    scaler = MinMaxScaler()
    scaler.fit(train_df[columns].values)
    logger.info("Scaler fit on training data: %d columns, %d rows", len(columns), len(train_df))
    return scaler


def apply_scaler(
    df: pd.DataFrame, scaler: MinMaxScaler, columns: list[str]
) -> pd.DataFrame:
    """
    Apply an already-fitted scaler to a dataframe. Never re-fits.

    Parameters
    ----------
    df : pd.DataFrame
        Any split (train, val, test) or new inference data.
    scaler : MinMaxScaler
        A scaler previously fit via fit_scaler() on training data.
    columns : list[str]
        Same columns the scaler was fit on, in the same order.

    Returns
    -------
    pd.DataFrame
        Copy of df with the specified columns replaced by scaled values.
    """
    df = df.copy()
    df[columns] = scaler.transform(df[columns].values)
    return df

def inverse_transform_column(
    scaled_values: np.ndarray,
    scaler: MinMaxScaler,
    columns: list[str],
    column_name: str,
) -> np.ndarray:
    """
    Inverse-transform a single column's scaled values back to original units.

    Necessary because our scaler was fit jointly across multiple columns
    (see fit_scaler), so a single column's values can't be inverse-transformed
    directly without reconstructing the full column layout the scaler expects.
    This works correctly because MinMaxScaler scales each column
    independently and linearly, so placing zeros in the other columns does
    not affect the target column's inverse transform.

    Parameters
    ----------
    scaled_values : np.ndarray
        Scaled values for ONE column, any shape (will be flattened and
        reshaped back).
    scaler : MinMaxScaler
        The scaler originally fit via fit_scaler() on `columns`.
    columns : list[str]
        The exact column list (and order) the scaler was fit on.
    column_name : str
        Which column `scaled_values` corresponds to.

    Returns
    -------
    np.ndarray
        Original-units values, same shape as the input.
    """
    if column_name not in columns:
        raise PreprocessingError(f"'{column_name}' not found in scaler's fitted columns.")

    original_shape = scaled_values.shape
    flat = scaled_values.flatten()
    col_idx = columns.index(column_name)

    dummy = np.zeros((len(flat), len(columns)))
    dummy[:, col_idx] = flat
    inverted = scaler.inverse_transform(dummy)[:, col_idx]

    return inverted.reshape(original_shape)