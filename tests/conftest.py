"""
Shared pytest fixtures: small synthetic datasets used across multiple
test files, so we don't depend on the real (large) dataset being
present to run tests.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_raw_df() -> pd.DataFrame:
    """
    A small, synthetic dataframe matching the real dataset's schema
    (same column names), spanning 300 continuous hourly timestamps —
    enough to exercise feature engineering (needs 168+ warm-up rows)
    and sequence building (needs lookback + horizon rows per split).
    """
    n = 300
    rng = np.random.RandomState(42)
    timestamps = pd.date_range("2020-01-01", periods=n, freq="h")

    df = pd.DataFrame({
        "datetime": timestamps.astype(str),
        "nat_demand": 1000 + 200 * np.sin(np.arange(n) * 2 * np.pi / 24) + rng.normal(0, 20, n),
        "T2M_toc": 25 + rng.normal(0, 2, n),
        "QV2M_toc": 0.015 + rng.normal(0, 0.001, n),
        "TQL_toc": rng.uniform(0, 0.1, n),
        "W2M_toc": rng.uniform(1, 5, n),
        "T2M_san": 26 + rng.normal(0, 2, n),
        "QV2M_san": 0.016 + rng.normal(0, 0.001, n),
        "TQL_san": rng.uniform(0, 0.1, n),
        "W2M_san": rng.uniform(1, 5, n),
        "T2M_dav": 24 + rng.normal(0, 2, n),
        "QV2M_dav": 0.014 + rng.normal(0, 0.001, n),
        "TQL_dav": rng.uniform(0, 0.1, n),
        "W2M_dav": rng.uniform(1, 5, n),
        "Holiday_ID": 0,
        "holiday": 0,
        "school": 1,
    })
    return df


@pytest.fixture
def synthetic_features_df(synthetic_raw_df):
    """
    A fully feature-engineered version of the synthetic dataset, ready
    for sequence building / scaling tests.
    """
    from src.preprocessing import clean_raw_data
    from src.feature_engineering import build_features

    df_clean = clean_raw_data(synthetic_raw_df)
    return build_features(df_clean)