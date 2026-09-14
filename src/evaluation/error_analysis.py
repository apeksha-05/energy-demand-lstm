"""
Error analysis utilities for diagnosing WHERE and WHY a forecasting
model's errors occur — not just single aggregate metrics.
"""

import numpy as np
import pandas as pd

from src.evaluation.metrics import mae, rmse


def error_by_horizon_step(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Compute MAE and RMSE separately for each step of the forecast horizon.

    Parameters
    ----------
    y_true : np.ndarray, shape (num_samples, horizon)
    y_pred : np.ndarray, shape (num_samples, horizon)

    Returns
    -------
    pd.DataFrame
        One row per horizon step (1-indexed for readability), columns
        "horizon_step", "MAE", "RMSE".
    """
    horizon = y_true.shape[1]
    rows = []
    for step in range(horizon):
        step_mae = mae(y_true[:, step], y_pred[:, step])
        step_rmse = rmse(y_true[:, step], y_pred[:, step])
        rows.append({"horizon_step": step + 1, "MAE": step_mae, "RMSE": step_rmse})
    return pd.DataFrame(rows)

def error_by_segment(
    df_segment_info: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    segment_column: str,
) -> pd.DataFrame:
    """
    Compute MAE/RMSE separately for each distinct value of a segmenting
    column (e.g., is_weekend, holiday, a binned hour-of-day, or a binned
    temperature range).

    Parameters
    ----------
    df_segment_info : pd.DataFrame
        A dataframe aligned row-for-row with y_true/y_pred (same length,
        same order), containing the column to segment by.
    y_true : np.ndarray, shape (n,)
    y_pred : np.ndarray, shape (n,)
    segment_column : str
        Column in df_segment_info to group by.

    Returns
    -------
    pd.DataFrame
        One row per segment value, columns: segment value, MAE, RMSE, count.
    """
    if len(df_segment_info) != len(y_true):
        raise ValueError(
            f"Length mismatch: df_segment_info has {len(df_segment_info)} rows, "
            f"y_true has {len(y_true)}"
        )

    temp = df_segment_info.copy()
    temp["_y_true"] = y_true
    temp["_y_pred"] = y_pred
    temp["_abs_error"] = np.abs(temp["_y_true"] - temp["_y_pred"])
    temp["_sq_error"] = (temp["_y_true"] - temp["_y_pred"]) ** 2

    grouped = temp.groupby(segment_column).agg(
        MAE=("_abs_error", "mean"),
        RMSE=("_sq_error", lambda x: np.sqrt(np.mean(x))),
        count=("_abs_error", "size"),
    ).reset_index()

    return grouped

def permutation_feature_importance(
    model,
    X: np.ndarray,
    y_true_scaled: np.ndarray,
    feature_names: list[str],
    predict_fn,
    n_repeats: int = 3,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Compute permutation feature importance for a sequence model.

    For each feature, shuffle its values across the BATCH dimension
    (i.e., mix up which sample each feature-value came from, at every
    timestep) while leaving every other feature and the sequence's
    temporal order untouched. Measure how much the loss (MSE) increases
    versus the unpermuted baseline. A large increase means the model
    relies heavily on that feature; near-zero change means the model
    doesn't use it much (at least not in a way this test can detect).

    Parameters
    ----------
    model : nn.Module
        A trained forecasting model.
    X : np.ndarray, shape (num_samples, lookback, num_features)
    y_true_scaled : np.ndarray, shape (num_samples, horizon)
        Ground truth in the SAME scale the model was trained on (scaled),
        since we're comparing relative MSE degradation, not reporting
        MW-unit errors here.
    feature_names : list[str]
        Names corresponding to the feature (last) dimension of X, in order.
    predict_fn : callable
        Function like src.training.trainer.predict(model, X) -> np.ndarray.
    n_repeats : int
        Number of random shuffles per feature, averaged for stability.

    Returns
    -------
    pd.DataFrame
        Columns: feature, baseline_mse, permuted_mse, importance
        (permuted_mse - baseline_mse), sorted descending by importance.
    """
    rng = np.random.RandomState(random_seed)

    baseline_pred = predict_fn(model, X)
    baseline_mse = float(np.mean((y_true_scaled - baseline_pred) ** 2))

    results = []
    for feat_idx, feat_name in enumerate(feature_names):
        permuted_mses = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            shuffle_idx = rng.permutation(X.shape[0])
            X_permuted[:, :, feat_idx] = X[shuffle_idx, :, feat_idx]

            permuted_pred = predict_fn(model, X_permuted)
            permuted_mse = float(np.mean((y_true_scaled - permuted_pred) ** 2))
            permuted_mses.append(permuted_mse)

        avg_permuted_mse = float(np.mean(permuted_mses))
        results.append({
            "feature": feat_name,
            "baseline_mse": baseline_mse,
            "permuted_mse": avg_permuted_mse,
            "importance": avg_permuted_mse - baseline_mse,
        })

    return pd.DataFrame(results).sort_values("importance", ascending=False).reset_index(drop=True)