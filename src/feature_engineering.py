"""
Feature engineering for the Panama electricity load dataset.

Builds four families of features on top of the cleaned dataframe:
  1. National weather averages (collapsing 3 weather stations into 1).
  2. Cyclical calendar encodings (hour, day-of-week, month).
  3. Lag features of the target (nat_demand).
  4. Rolling statistics of the target.

CRITICAL LEAKAGE RULE: every lag and rolling feature at row t must depend
ONLY on nat_demand values strictly before t. This is enforced by always
shifting by at least 1 before applying `.rolling(...)`.

These features are computed on the full chronologically-sorted series
BEFORE train/val/test splitting. This is intentional and is NOT leakage:
lag/rolling features are deterministic, causal lookups of real historical
values (which would genuinely be available at prediction time), unlike
scalers or model weights, which must never be fit on non-training data.
"""

import numpy as np
import pandas as pd

from src.config import (
    TARGET_COL,
    TEMPERATURE_COLS,
    HUMIDITY_COLS,
    PRECIP_PROXY_COLS,
    WIND_COLS,
)
from src.utils import get_logger

logger = get_logger(__name__)

LAG_HOURS: list[int] = [1, 24, 48, 168]
ROLLING_WINDOWS: list[int] = [24, 24, 168]  # mean_24, std_24, mean_168 (paired below)


class FeatureEngineeringError(Exception):
    """Raised when feature engineering encounters an invalid input."""


def add_weather_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the 3-station weather columns into 4 national average
    features: temp_avg, humidity_avg, precip_avg, wind_avg.

    Rationale: nat_demand is a national aggregate, so a national-average
    weather signal is a more honest match than any single city, and
    averaging avoids the multicollinearity of 3 highly-correlated
    per-city columns for each variable.
    """
    df = df.copy()
    df["temp_avg"] = df[TEMPERATURE_COLS].mean(axis=1)
    df["humidity_avg"] = df[HUMIDITY_COLS].mean(axis=1)
    df["precip_avg"] = df[PRECIP_PROXY_COLS].mean(axis=1)
    df["wind_avg"] = df[WIND_COLS].mean(axis=1)
    logger.info("Added national weather averages: temp_avg, humidity_avg, precip_avg, wind_avg")
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add raw calendar features and their cyclical (sin/cos) encodings.

    Raw integer calendar features are kept alongside the cyclical
    encodings for interpretability during EDA/error analysis, but models
    should be trained on the cyclical versions (hour_sin/hour_cos etc.)
    rather than raw hour/day_of_week/month, to correctly represent their
    circular nature (e.g. hour 23 and hour 0 are 1 hour apart, not far apart).
    """
    df = df.copy()
    idx = df.index

    df["hour"] = idx.hour
    df["day_of_week"] = idx.dayofweek  # 0=Monday, 6=Sunday
    df["day_of_month"] = idx.day
    df["week_of_year"] = idx.isocalendar().week.values
    df["month"] = idx.month
    df["quarter"] = idx.quarter
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    logger.info("Added calendar features and cyclical encodings")
    return df


def add_lag_features(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """
    Add lag features of the target: lag_1, lag_24, lag_48, lag_168.

    Each lag_N at row t equals the target value at row t-N — i.e. what
    demand was N hours before this timestamp. Uses pandas' native
    .shift(N), which is inherently leakage-safe (it can only look backward).
    """
    df = df.copy()
    for lag in LAG_HOURS:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    logger.info("Added lag features: %s", [f"lag_{lag}" for lag in LAG_HOURS])
    return df


def add_rolling_features(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """
    Add rolling statistics of the target: rolling_mean_24, rolling_std_24,
    rolling_mean_168.

    LEAKAGE-SAFE BY CONSTRUCTION: the target is shifted by 1 BEFORE the
    rolling window is applied, so the window at row t covers
    [t-window, ..., t-1] — it never includes the current row's own value.
    """
    df = df.copy()
    shifted = df[target_col].shift(1)

    df["rolling_mean_24"] = shifted.rolling(window=24).mean()
    df["rolling_std_24"] = shifted.rolling(window=24).std()
    df["rolling_mean_168"] = shifted.rolling(window=168).mean()

    logger.info("Added rolling features: rolling_mean_24, rolling_std_24, rolling_mean_168")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline in order, then drop rows
    with NaN values introduced by lag/rolling features (only the first
    168 rows of the entire series will be affected).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned, datetime-indexed dataframe from preprocessing.clean_raw_data().

    Returns
    -------
    pd.DataFrame
        Fully feature-engineered dataframe, ready for chronological
        splitting and sequence generation.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise FeatureEngineeringError(
            "Expected a DatetimeIndex. Run preprocessing.clean_raw_data() first."
        )

    n_before = len(df)
    df = add_weather_averages(df)
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    df = df.dropna()
    n_after = len(df)
    logger.info(
        "Feature engineering complete: %d rows before, %d rows after dropping "
        "NaN warm-up rows (%d dropped, expected ~168)",
        n_before, n_after, n_before - n_after,
    )
    return df