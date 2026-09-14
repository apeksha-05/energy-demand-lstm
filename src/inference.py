"""
Model saving/loading and inference pipeline for the final selected model
(single-layer, multivariate LSTM).

Training and inference are deliberately separated: this module NEVER
fits a scaler or trains a model — it only loads previously-saved
artifacts and applies them identically to how they were used at
training time, preventing any train/inference preprocessing mismatch.
"""

import json
import pickle
from pathlib import Path
from src.preprocessing import inverse_transform_column
from src.preprocessing import clean_raw_data, inverse_transform_column

import torch

from src.config import MODELS_DIR, MODEL_FEATURE_COLUMNS, TARGET_COL, DEFAULT_LOOKBACK, FORECAST_HORIZON, RNN_HIDDEN_SIZE
from src.models.lstm import LSTMForecaster
from src.utils import get_logger

logger = get_logger(__name__)

MODEL_WEIGHTS_PATH = MODELS_DIR / "lstm_final.pt"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
FEATURE_CONFIG_PATH = MODELS_DIR / "feature_config.json"


class InferenceError(Exception):
    """Raised when saving or loading model artifacts fails."""


def save_model_artifacts(model, scaler, scale_columns: list[str]) -> None:
    """
    Save the trained model's weights, the fitted scaler, and the exact
    feature configuration used at training time — everything inference
    needs to reproduce identical preprocessing later.

    Parameters
    ----------
    model : nn.Module
        The trained LSTMForecaster.
    scaler : MinMaxScaler
        The scaler fit on training data (never re-fit at inference time).
    scale_columns : list[str]
        The exact column list/order the scaler was fit on.
    """
    MODELS_DIR.mkdir(exist_ok=True)

    torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
    logger.info("Saved model weights to %s", MODEL_WEIGHTS_PATH)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Saved scaler to %s", SCALER_PATH)

    feature_config = {
        "model_feature_columns": MODEL_FEATURE_COLUMNS,
        "scale_columns": scale_columns,
        "target_column": TARGET_COL,
        "lookback": DEFAULT_LOOKBACK,
        "horizon": FORECAST_HORIZON,
        "hidden_size": RNN_HIDDEN_SIZE,
        "num_features": len(MODEL_FEATURE_COLUMNS),
    }
    with open(FEATURE_CONFIG_PATH, "w") as f:
        json.dump(feature_config, f, indent=2)
    logger.info("Saved feature config to %s", FEATURE_CONFIG_PATH)


def load_model_artifacts():
    """
    Load the saved model, scaler, and feature config for inference.

    Returns
    -------
    tuple[nn.Module, MinMaxScaler, dict]
        The reconstructed model (weights loaded, in eval mode), the
        fitted scaler, and the feature configuration dict.

    Raises
    ------
    InferenceError
        If any required artifact file is missing.
    """
    for path in [MODEL_WEIGHTS_PATH, SCALER_PATH, FEATURE_CONFIG_PATH]:
        if not path.exists():
            raise InferenceError(
                f"Missing model artifact: {path}. "
                f"Run the training pipeline and save_model_artifacts() first."
            )

    with open(FEATURE_CONFIG_PATH) as f:
        feature_config = json.load(f)

    model = LSTMForecaster(
        num_features=feature_config["num_features"],
        hidden_size=feature_config["hidden_size"],
        horizon=feature_config["horizon"],
    )
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, weights_only=True))
    model.eval()

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    logger.info("Loaded model artifacts successfully")
    return model, scaler, feature_config


import numpy as np
import pandas as pd

from src.data_loader import load_raw_data
from src.preprocessing import clean_raw_data
from src.feature_engineering import build_features


def prepare_forecast_input(
    as_of_timestamp: str,
    feature_config: dict,
    scaler,
) -> tuple[np.ndarray, pd.Timestamp]:
    """
    Build a single model-ready input sequence for forecasting the 24 hours
    following `as_of_timestamp`, using real historical data up to and
    including that point.

    Applies the EXACT SAME cleaning, feature engineering, and scaling as
    training — never refits anything, only transforms.

    Parameters
    ----------
    as_of_timestamp : str
        A timestamp string (e.g. "2020-01-15 14:00:00") that must exist
        in the historical dataset. The forecast will predict the 24
        hours immediately following this point.
    feature_config : dict
        Loaded via load_model_artifacts() — contains lookback, feature
        columns, scale columns, etc.
    scaler : MinMaxScaler
        The fitted scaler loaded via load_model_artifacts().

    Returns
    -------
    tuple[np.ndarray, pd.Timestamp]
        X of shape (1, lookback, num_features), ready for model input,
        and the parsed as_of_timestamp for reference.

    Raises
    ------
    InferenceError
        If the timestamp doesn't exist in the data, or there isn't
        enough history before it to build a full lookback window.
    """
    as_of = pd.Timestamp(as_of_timestamp)
    lookback = feature_config["lookback"]
    feature_columns = feature_config["model_feature_columns"]
    scale_columns = feature_config["scale_columns"]

    df_raw = load_raw_data()
    df_clean = clean_raw_data(df_raw)
    df_features = build_features(df_clean)

    if as_of not in df_features.index:
        raise InferenceError(
            f"Timestamp '{as_of}' not found in historical data. "
            f"Available range: {df_features.index.min()} to {df_features.index.max()}."
        )

    window_df = df_features.loc[:as_of].tail(lookback)
    if len(window_df) < lookback:
        raise InferenceError(
            f"Insufficient historical data before '{as_of}': "
            f"need {lookback} hours, found only {len(window_df)}."
        )

    window_scaled = window_df.copy()
    window_scaled[scale_columns] = scaler.transform(window_df[scale_columns].values)

    X = window_scaled[feature_columns].values.astype(np.float32)
    X = X.reshape(1, lookback, len(feature_columns))  # (1, lookback, num_features)

    return X, as_of


def generate_forecast(as_of_timestamp: str) -> pd.DataFrame:
    """
    Full end-to-end inference: load artifacts, prepare input, predict,
    inverse-transform, and return a readable forecast dataframe.

    Parameters
    ----------
    as_of_timestamp : str
        Timestamp to forecast the next 24 hours FROM.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, predicted_demand_mw. One row per forecast
        horizon step (24 rows).
    """
    model, scaler, feature_config = load_model_artifacts()

    X, as_of = prepare_forecast_input(as_of_timestamp, feature_config, scaler)

    import torch
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(X).float()).numpy()  # (1, horizon)

    scale_columns = feature_config["scale_columns"]
    target_column = feature_config["target_column"]

    from src.preprocessing import inverse_transform_column
    pred_mw = inverse_transform_column(pred_scaled, scaler, scale_columns, target_column)
    pred_mw = pred_mw.flatten()  # (horizon,)

    horizon = feature_config["horizon"]
    forecast_timestamps = pd.date_range(
        start=as_of + pd.Timedelta(hours=1), periods=horizon, freq="h"
    )

    result = pd.DataFrame({
        "timestamp": forecast_timestamps,
        "predicted_demand_mw": pred_mw,
    })
    logger.info("Generated %d-hour forecast starting from %s", horizon, as_of)
    return result


def check_recent_anomalies(as_of_timestamp: str, lookback_hours: int = 168) -> pd.DataFrame:
    """
    Check whether actual demand in the recent past (leading up to
    as_of_timestamp) contained any anomalies, using the same rolling
    z-score detector as Phase 17. This gives dashboard users useful
    context ("has anything unusual happened recently?") alongside the
    forward-looking forecast.

    Note: this checks PAST actual-vs-predicted residuals, not the future
    forecast itself — we cannot detect anomalies in a forecast that
    hasn't happened yet, since anomaly detection requires a real
    "actual" value to compare against.

    Returns
    -------
    pd.DataFrame
        Anomaly detection results (see src.anomaly.detector.detect_anomalies)
        for the recent historical window, or an empty dataframe if
        insufficient history exists for the check.
    """
    from src.anomaly.detector import compute_residuals, detect_anomalies, DEFAULT_ROLLING_WINDOW

    model, scaler, feature_config = load_model_artifacts()
    as_of = pd.Timestamp(as_of_timestamp)
    lookback = feature_config["lookback"]

    df_raw = load_raw_data()
    df_clean = clean_raw_data(df_raw)
    df_features = build_features(df_clean)

    # Need enough history to both build sequences AND have a rolling window of residuals
    required_hours = lookback + lookback_hours + DEFAULT_ROLLING_WINDOW
    history = df_features.loc[:as_of].tail(required_hours)

    if len(history) < required_hours:
        logger.info("Insufficient history for anomaly check (need %d, have %d) — skipping", required_hours, len(history))
        return pd.DataFrame()

    from src.sequence_builder import create_sequences
    scale_columns = feature_config["scale_columns"]
    feature_columns = feature_config["model_feature_columns"]
    target_column = feature_config["target_column"]

    history_scaled = history.copy()
    history_scaled[scale_columns] = scaler.transform(history[scale_columns].values)

    X_hist, y_hist_scaled = create_sequences(
        history_scaled, feature_columns, target_column, lookback=lookback, horizon=1
    )

    import torch
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(X_hist).float()).numpy()
        pred_scaled_h1 = pred_scaled[:, 0:1]  # just the first horizon step for a clean series

    from src.preprocessing import inverse_transform_column
    y_true_mw = inverse_transform_column(y_hist_scaled, scaler, scale_columns, target_column).flatten()
    y_pred_mw = inverse_transform_column(pred_scaled_h1, scaler, scale_columns, target_column).flatten()

    residuals = compute_residuals(y_true_mw, y_pred_mw)
    anomaly_results = detect_anomalies(residuals)
    return anomaly_results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        as_of_arg = sys.argv[1]
    else:
        # Default: use a known-good timestamp from the dataset for a quick demo run
        as_of_arg = "2020-01-15 12:00:00"

    print(f"Generating forecast as of {as_of_arg}...\n")
    forecast = generate_forecast(as_of_arg)
    print(forecast.to_string(index=False))

    print("\nChecking for recent anomalies...")
    anomalies = check_recent_anomalies(as_of_arg)
    if len(anomalies) > 0 and anomalies["is_anomaly"].any():
        print(f"\n{anomalies['is_anomaly'].sum()} anomaly(ies) detected in recent history.")
    else:
        print("\nNo anomalies detected in recent history.")
        
        
def compute_residuals_for_range(start_timestamp: str, end_timestamp: str) -> pd.DataFrame:
    """
    Compute actual-vs-predicted (hour+1) residuals for every hour in a
    given historical date range, for anomaly detection / dashboard display.

    Parameters
    ----------
    start_timestamp, end_timestamp : str
        Inclusive range boundaries. Must both exist in the historical
        dataset, and there must be enough history before start_timestamp
        to build the model's required lookback window.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, actual, predicted, residual — one row per
        hour in [start_timestamp, end_timestamp].
    """
    model, scaler, feature_config = load_model_artifacts()
    lookback = feature_config["lookback"]
    feature_columns = feature_config["model_feature_columns"]
    scale_columns = feature_config["scale_columns"]
    target_column = feature_config["target_column"]

    start = pd.Timestamp(start_timestamp)
    end = pd.Timestamp(end_timestamp)

    df_raw = load_raw_data()
    df_clean = clean_raw_data(df_raw)
    df_features = build_features(df_clean)

    # Need `lookback` hours of history BEFORE start, plus the range itself
    history = df_features.loc[:end]
    history = history[history.index >= (start - pd.Timedelta(hours=lookback))]

    if history.index.min() > start - pd.Timedelta(hours=lookback):
        raise InferenceError(
            f"Insufficient historical data before '{start}' to build a "
            f"{lookback}-hour lookback window."
        )

    history_scaled = history.copy()
    history_scaled[scale_columns] = scaler.transform(history[scale_columns].values)

    from src.sequence_builder import create_sequences
    X_hist, y_hist_scaled = create_sequences(
        history_scaled, feature_columns, target_column, lookback=lookback, horizon=1
    )

    import torch
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(X_hist).float()).numpy()[:, 0:1]

    y_true_mw = inverse_transform_column(y_hist_scaled, scaler, scale_columns, target_column).flatten()
    y_pred_mw = inverse_transform_column(pred_scaled, scaler, scale_columns, target_column).flatten()

    # The sequences start `lookback` rows into `history`, so timestamps
    # align starting from history.index[lookback]
    result_timestamps = history.index[lookback : lookback + len(y_true_mw)]

    result = pd.DataFrame({
        "timestamp": result_timestamps,
        "actual": y_true_mw,
        "predicted": y_pred_mw,
        "residual": y_true_mw - y_pred_mw,
    })

    # Trim to the exact requested [start, end] range (history may include
    # extra lookback rows before start)
    result = result[(result["timestamp"] >= start) & (result["timestamp"] <= end)].reset_index(drop=True)
    return result       


def compute_residuals_for_range(start_timestamp: str, end_timestamp: str) -> pd.DataFrame:
    """
    Compute actual-vs-predicted (hour+1) residuals for every hour in a
    given historical date range, for anomaly detection / dashboard display.

    Parameters
    ----------
    start_timestamp, end_timestamp : str
        Inclusive range boundaries. Must both exist in the historical
        dataset, and there must be enough history before start_timestamp
        to build the model's required lookback window.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, actual, predicted, residual — one row per
        hour in [start_timestamp, end_timestamp].
    """
    model, scaler, feature_config = load_model_artifacts()
    lookback = feature_config["lookback"]
    feature_columns = feature_config["model_feature_columns"]
    scale_columns = feature_config["scale_columns"]
    target_column = feature_config["target_column"]

    start = pd.Timestamp(start_timestamp)
    end = pd.Timestamp(end_timestamp)

    df_raw = load_raw_data()
    df_clean = clean_raw_data(df_raw)
    df_features = build_features(df_clean)

    history = df_features.loc[:end]
    history = history[history.index >= (start - pd.Timedelta(hours=lookback))]

    if history.index.min() > start - pd.Timedelta(hours=lookback):
        raise InferenceError(
            f"Insufficient historical data before '{start}' to build a "
            f"{lookback}-hour lookback window."
        )

    history_scaled = history.copy()
    history_scaled[scale_columns] = scaler.transform(history[scale_columns].values)

    from src.sequence_builder import create_sequences
    X_hist, y_hist_scaled = create_sequences(
        history_scaled, feature_columns, target_column, lookback=lookback, horizon=1
    )

    import torch
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(X_hist).float()).numpy()[:, 0:1]

    y_true_mw = inverse_transform_column(y_hist_scaled, scaler, scale_columns, target_column).flatten()
    y_pred_mw = inverse_transform_column(pred_scaled, scaler, scale_columns, target_column).flatten()

    result_timestamps = history.index[lookback : lookback + len(y_true_mw)]

    result = pd.DataFrame({
        "timestamp": result_timestamps,
        "actual": y_true_mw,
        "predicted": y_pred_mw,
        "residual": y_true_mw - y_pred_mw,
    })

    result = result[(result["timestamp"] >= start) & (result["timestamp"] <= end)].reset_index(drop=True)
    return result