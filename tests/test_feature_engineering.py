"""
Unit tests for src.feature_engineering: leakage safety and correct
feature construction.
"""

import numpy as np

from src.feature_engineering import build_features
from src.preprocessing import clean_raw_data


def test_build_features_no_nan_after_warmup_drop(synthetic_raw_df):
    df_clean = clean_raw_data(synthetic_raw_df)
    df_features = build_features(df_clean)
    assert df_features.isnull().sum().sum() == 0


def test_lag_1_matches_shifted_target(synthetic_raw_df):
    df_clean = clean_raw_data(synthetic_raw_df)
    df_features = build_features(df_clean)

    # lag_1 at row t should equal nat_demand at row t-1 (before dropping NaNs,
    # so we re-derive from the clean df directly for a ground-truth comparison)
    expected_lag1 = df_clean["nat_demand"].shift(1)
    aligned_expected = expected_lag1.loc[df_features.index]

    np.testing.assert_allclose(df_features["lag_1"].values, aligned_expected.values, rtol=1e-5)


def test_rolling_mean_excludes_current_row(synthetic_raw_df):
    """
    LEAKAGE CHECK: rolling_mean_24 at row t must be computed from rows
    BEFORE t, never including t's own nat_demand value.
    """
    df_clean = clean_raw_data(synthetic_raw_df)
    df_features = build_features(df_clean)

    # Manually recompute what rolling_mean_24 SHOULD be for a sample row,
    # using only data strictly before it, and confirm it matches.
    sample_ts = df_features.index[50]  # well within range (132 rows available) and far enough in to have full rolling history
    preceding_24 = df_clean.loc[:sample_ts, "nat_demand"].iloc[-25:-1]  # 24 rows before sample_ts, excluding it
    expected_mean = preceding_24.mean()

    actual = df_features.loc[sample_ts, "rolling_mean_24"]
    np.testing.assert_allclose(actual, expected_mean, rtol=1e-5)


def test_cyclical_encoding_bounded(synthetic_raw_df):
    df_clean = clean_raw_data(synthetic_raw_df)
    df_features = build_features(df_clean)

    for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]:
        assert df_features[col].min() >= -1.0001
        assert df_features[col].max() <= 1.0001


def test_hour_0_and_23_are_close_in_cyclical_space(synthetic_raw_df):
    """
    Confirms the core motivation for cyclical encoding: hour 23 and hour 0
    should be numerically CLOSE in (sin, cos) space, unlike raw integers.
    """
    df_clean = clean_raw_data(synthetic_raw_df)
    df_features = build_features(df_clean)

    row_hour_23 = df_features[df_features["hour"] == 23].iloc[0]
    row_hour_0 = df_features[df_features["hour"] == 0].iloc[0]

    dist = np.sqrt(
        (row_hour_23["hour_sin"] - row_hour_0["hour_sin"]) ** 2
        + (row_hour_23["hour_cos"] - row_hour_0["hour_cos"]) ** 2
    )
    assert dist < 0.5  # much less than the max possible distance of 2.0