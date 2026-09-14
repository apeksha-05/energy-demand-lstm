"""
Central configuration for the Energy Demand LSTM project.
Single source of truth for paths and constants used across all modules.
"""

from pathlib import Path

# --- Project root & data paths -------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DATA_DIR: Path = DATA_DIR / "external"

RAW_DATASET_FILENAME: str = "continuous_dataset.csv"
RAW_DATASET_PATH: Path = RAW_DATA_DIR / RAW_DATASET_FILENAME

MODELS_DIR: Path = PROJECT_ROOT / "models"
EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments"

# --- Reproducibility ------------------------------------------------------------
RANDOM_SEED: int = 42

# --- Expected raw columns --------------------------------------------------------
# Filled in after Phase 1 inspection confirms exact column names/casing.
EXPECTED_RAW_COLUMNS: list[str] = [
    "datetime",
    "nat_demand",
    "T2M_toc", "QV2M_toc", "TQL_toc", "W2M_toc",
    "T2M_san", "QV2M_san", "TQL_san", "W2M_san",
    "T2M_dav", "QV2M_dav", "TQL_dav", "W2M_dav",
    "Holiday_ID", "holiday", "school",
]

# --- Column name aliases (for readability elsewhere in the codebase) ------------
DATETIME_COL: str = "datetime"
TARGET_COL: str = "nat_demand"  # electricity demand — the forecasting target

# Per-city weather columns (three stations: Tocumen, Santiago, David)
TEMPERATURE_COLS: list[str] = ["T2M_toc", "T2M_san", "T2M_dav"]
HUMIDITY_COLS: list[str] = ["QV2M_toc", "QV2M_san", "QV2M_dav"]
PRECIP_PROXY_COLS: list[str] = ["TQL_toc", "TQL_san", "TQL_dav"]
WIND_COLS: list[str] = ["W2M_toc", "W2M_san", "W2M_dav"]

HOLIDAY_ID_COL: str = "Holiday_ID"
HOLIDAY_FLAG_COL: str = "holiday"
SCHOOL_FLAG_COL: str = "school"

# --- Engineered feature groups (populated after Phase 4) ------------------------
WEATHER_FEATURES: list[str] = ["temp_avg", "humidity_avg", "precip_avg", "wind_avg"]

CYCLICAL_FEATURES: list[str] = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
]

CALENDAR_FLAG_FEATURES: list[str] = ["is_weekend", "holiday", "school"]

LAG_FEATURES: list[str] = ["lag_1", "lag_24", "lag_48", "lag_168"]

ROLLING_FEATURES: list[str] = ["rolling_mean_24", "rolling_std_24", "rolling_mean_168"]

# Full multivariate feature set used as model input (excludes the target itself,
# which models will predict — TARGET_COL is included separately when needed,
# e.g. for lag/rolling computation or as the sequence's "y").
MODEL_FEATURE_COLUMNS: list[str] = (
    WEATHER_FEATURES + CYCLICAL_FEATURES + CALENDAR_FLAG_FEATURES
    + LAG_FEATURES + ROLLING_FEATURES
)

# --- Univariate feature subset (Phase 9 comparison) ------------------------------
UNIVARIATE_FEATURE_COLUMNS: list[str] = LAG_FEATURES + ROLLING_FEATURES

# --- Sequence generation ---------------------------------------------------------
DEFAULT_LOOKBACK: int = 72      # hours of history fed to the model
FORECAST_HORIZON: int = 24       # hours ahead to predict (direct multi-output)
LOOKBACK_CANDIDATES: list[int] = [24, 48, 72, 168]  # for Phase 5/16 comparison

# --- Neural network training defaults --------------------------------------------
BATCH_SIZE: int = 256
EPOCHS: int = 30
LEARNING_RATE: float = 1e-3
RNN_HIDDEN_SIZE: int = 64
EARLY_STOPPING_PATIENCE: int = 5

LSTM_NUM_LAYERS: int = 2
LSTM_DROPOUT: float = 0.2

TRANSFORMER_D_MODEL: int = 64
TRANSFORMER_NUM_HEADS: int = 4
TRANSFORMER_NUM_LAYERS: int = 1
TRANSFORMER_DIM_FEEDFORWARD: int = 128