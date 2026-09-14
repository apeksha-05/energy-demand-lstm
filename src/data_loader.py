"""
Data loading for the Panama electricity load dataset.

This module is the single entry point for reading the raw dataset from disk.
It performs basic structural validation (file exists, non-empty, has all
expected columns) but does NOT clean or transform the data — that
responsibility belongs to preprocessing.py.
"""

from pathlib import Path

import pandas as pd

from src.config import RAW_DATASET_PATH, EXPECTED_RAW_COLUMNS
from src.utils import get_logger

logger = get_logger(__name__)


class DataLoadError(Exception):
    """Raised when the raw dataset cannot be loaded or is structurally invalid."""


def load_raw_data(path: Path = RAW_DATASET_PATH) -> pd.DataFrame:
    """
    Load the raw Panama electricity load CSV from disk.

    Parameters
    ----------
    path : Path
        Location of the raw CSV file. Defaults to the configured path.

    Returns
    -------
    pd.DataFrame
        The raw dataset, unmodified except for basic type parsing.

    Raises
    ------
    DataLoadError
        If the file is missing, empty, structurally invalid, or missing
        expected columns.
    """
    if not path.exists():
        raise DataLoadError(
            f"Raw dataset not found at '{path}'. "
            f"Download it from Kaggle "
            f"(ernestojaguilar/shortterm-electricity-load-forecasting-panama) "
            f"and place it at this exact path."
        )

    logger.info("Loading raw dataset from %s", path)

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise DataLoadError(f"Failed to parse CSV at '{path}': {exc}") from exc

    if df.empty:
        raise DataLoadError(f"Raw dataset at '{path}' loaded but contains 0 rows.")

    missing_cols = set(EXPECTED_RAW_COLUMNS) - set(df.columns)
    if missing_cols:
        raise DataLoadError(
            f"Raw dataset is missing expected columns: {missing_cols}. "
            f"The dataset format may have changed — verify the source file."
        )

    logger.info(
        "Loaded raw dataset: %d rows, %d columns", df.shape[0], df.shape[1]
    )
    return df


def inspect_raw_data(df: pd.DataFrame) -> None:
    """
    Print a structural summary of the raw dataframe: columns, dtypes,
    row count, and missing-value counts per column. Used for first-pass
    inspection in Phase 1 — not part of the production pipeline.
    """
    print("Shape:", df.shape)
    print("\nColumns and dtypes:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nFirst 3 rows:")
    print(df.head(3))