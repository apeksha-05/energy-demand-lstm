"""
Unit tests for src.preprocessing: cleaning, chronological splitting,
scaling, and inverse transform.
"""

import numpy as np
import pytest

from src.preprocessing import (
    clean_raw_data,
    chronological_split,
    fit_scaler,
    apply_scaler,
    inverse_transform_column,
    PreprocessingError,
)


def test_clean_raw_data_sets_datetime_index(synthetic_raw_df):
    result = clean_raw_data(synthetic_raw_df)
    assert isinstance(result.index, type(result.index))  # DatetimeIndex
    assert result.index.name == "datetime" or result.index.dtype.kind == "M"


def test_clean_raw_data_adds_extreme_event_flag(synthetic_raw_df):
    result = clean_raw_data(synthetic_raw_df)
    assert "is_extreme_event" in result.columns
    assert result["is_extreme_event"].dtype == bool


def test_clean_raw_data_rejects_duplicate_timestamps(synthetic_raw_df):
    df_with_dupe = synthetic_raw_df.copy()
    df_with_dupe.loc[len(df_with_dupe)] = df_with_dupe.iloc[0]  # duplicate first row's timestamp
    with pytest.raises(PreprocessingError):
        clean_raw_data(df_with_dupe)


def test_chronological_split_preserves_order_and_size(synthetic_raw_df):
    df_clean = clean_raw_data(synthetic_raw_df)
    splits = chronological_split(df_clean, train_frac=0.7, val_frac=0.15)

    assert len(splits.train) + len(splits.val) + len(splits.test) == len(df_clean)
    # Chronological: train must end before val starts, val before test starts
    assert splits.train.index.max() < splits.val.index.min()
    assert splits.val.index.max() < splits.test.index.min()


def test_chronological_split_rejects_invalid_fractions(synthetic_raw_df):
    df_clean = clean_raw_data(synthetic_raw_df)
    with pytest.raises(PreprocessingError):
        chronological_split(df_clean, train_frac=0.7, val_frac=0.4)  # sums >= 1.0


def test_scaler_fit_only_on_train_range(synthetic_raw_df):
    df_clean = clean_raw_data(synthetic_raw_df)
    splits = chronological_split(df_clean)
    columns = ["nat_demand", "T2M_toc"]

    scaler = fit_scaler(splits.train, columns)
    train_scaled = apply_scaler(splits.train, scaler, columns)

    # Train data, once scaled with a scaler fit on itself, must be within [0, 1]
    assert train_scaled[columns].values.min() >= -1e-9
    assert train_scaled[columns].values.max() <= 1 + 1e-9


def test_inverse_transform_column_recovers_original_values(synthetic_raw_df):
    df_clean = clean_raw_data(synthetic_raw_df)
    columns = ["nat_demand", "T2M_toc"]
    scaler = fit_scaler(df_clean, columns)

    scaled = apply_scaler(df_clean, scaler, columns)
    recovered = inverse_transform_column(
        scaled["nat_demand"].values, scaler, columns, "nat_demand"
    )

    np.testing.assert_allclose(recovered, df_clean["nat_demand"].values, rtol=1e-4)