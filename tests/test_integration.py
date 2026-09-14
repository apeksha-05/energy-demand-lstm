"""
Integration test: exercises the full pipeline end-to-end on synthetic
data — clean -> feature engineer -> split -> scale -> sequence -> train
(1 tiny epoch) -> predict -> inverse transform -> evaluate. Confirms
every stage's OUTPUT correctly feeds the next stage's INPUT.
"""

import numpy as np

from src.preprocessing import clean_raw_data, chronological_split, fit_scaler, apply_scaler, inverse_transform_column
from src.feature_engineering import build_features
from src.sequence_builder import create_sequences_for_all_splits
from src.models.lstm import LSTMForecaster
from src.training.trainer import train_model, predict
from src.evaluation.metrics import evaluate_all


def test_full_pipeline_end_to_end(synthetic_raw_df):
    df_clean = clean_raw_data(synthetic_raw_df)
    df_features = build_features(df_clean)
    splits = chronological_split(df_features, train_frac=0.6, val_frac=0.2)

    feature_columns = ["T2M_toc", "humidity_avg", "hour_sin", "hour_cos", "lag_1"]
    target_column = "nat_demand"
    scale_columns = feature_columns + [target_column]

    scaler = fit_scaler(splits.train, scale_columns)
    train_scaled = apply_scaler(splits.train, scaler, scale_columns)
    val_scaled = apply_scaler(splits.val, scaler, scale_columns)
    test_scaled = apply_scaler(splits.test, scaler, scale_columns)

    sequences = create_sequences_for_all_splits(
        train_scaled, val_scaled, test_scaled, feature_columns, target_column,
        lookback=12, horizon=3,
    )
    X_train, y_train = sequences["train"]
    X_val, y_val = sequences["val"]
    X_test, y_test = sequences["test"]

    assert X_train.shape[0] > 0 and X_val.shape[0] > 0 and X_test.shape[0] > 0

    model = LSTMForecaster(num_features=len(feature_columns), hidden_size=8, horizon=3)
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=2, batch_size=8)

    assert history["epochs_trained"] >= 1
    assert len(history["train_loss"]) == history["epochs_trained"]

    preds_scaled = predict(model, X_test)
    assert preds_scaled.shape == y_test.shape

    y_true_mw = inverse_transform_column(y_test, scaler, scale_columns, target_column)
    y_pred_mw = inverse_transform_column(preds_scaled, scaler, scale_columns, target_column)

    metrics = evaluate_all(y_true_mw, y_pred_mw)
    assert "MAE" in metrics and np.isfinite(metrics["MAE"])
    assert metrics["MAE"] >= 0