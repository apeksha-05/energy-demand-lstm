"""
Unit tests for src.sequence_builder: shape correctness and no
train/val/test boundary leakage.
"""

import numpy as np

from src.sequence_builder import create_sequences, get_last_known_target


def test_sequence_shapes_correct(synthetic_features_df):
    feature_cols = ["T2M_toc", "nat_demand"]
    X, y = create_sequences(synthetic_features_df, feature_cols, lookback=24, horizon=6)

    assert X.shape[1] == 24
    assert X.shape[2] == len(feature_cols)
    assert y.shape[1] == 6
    assert X.shape[0] == y.shape[0]


def test_sequence_x_precedes_y_no_overlap(synthetic_features_df):
    """
    Confirms sequence i's target values come from AFTER its input window
    — no overlap between X's timesteps and y's target indices.
    """
    feature_cols = ["nat_demand"]
    lookback, horizon = 24, 6
    X, y = create_sequences(synthetic_features_df, feature_cols, lookback=lookback, horizon=horizon)

    target_values = synthetic_features_df["nat_demand"].values
    expected_y0 = target_values[lookback : lookback + horizon]

    np.testing.assert_allclose(y[0], expected_y0, rtol=1e-5)


def test_get_last_known_target_aligns_with_sequences(synthetic_features_df):
    lookback, horizon = 24, 6
    last_known = get_last_known_target(synthetic_features_df, lookback=lookback, horizon=horizon)

    target_values = synthetic_features_df["nat_demand"].values
    expected_first = target_values[lookback - 1]  # last input timestep of window 0

    assert np.isclose(last_known[0], expected_first)


def test_sequence_count_matches_formula(synthetic_features_df):
    lookback, horizon = 24, 6
    feature_cols = ["nat_demand"]
    X, y = create_sequences(synthetic_features_df, feature_cols, lookback=lookback, horizon=horizon)

    expected_count = len(synthetic_features_df) - lookback - horizon + 1
    assert X.shape[0] == expected_count