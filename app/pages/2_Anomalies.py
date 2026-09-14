"""
Anomalies page — user selects a date range, views detected anomalies
using the rolling z-score residual detector (Phase 17).
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dashboard_utils import get_full_dataset
from src.inference import compute_residuals_for_range
from src.anomaly.detector import detect_anomalies, severity_label, DEFAULT_ROLLING_WINDOW, DEFAULT_Z_THRESHOLD

st.set_page_config(page_title="Anomalies", page_icon="🚨", layout="wide")
st.title("🚨 Anomaly Detection")
st.markdown(
    "Detects unusual demand using a **rolling z-score of forecasting residuals** "
    "(actual minus predicted). See Phase 17 methodology: the threshold adapts to "
    "recent local volatility rather than using one fixed cutoff for the whole series."
)

df = get_full_dataset()
min_date, max_date = df.index.min(), df.index.max()

col1, col2, col3 = st.columns(3)
with col1:
    start_date = st.date_input("Start date", value=(max_date - pd.Timedelta(days=60)).date(), min_value=min_date.date(), max_value=max_date.date())
with col2:
    end_date = st.date_input("End date", value=max_date.date(), min_value=min_date.date(), max_value=max_date.date())
with col3:
    z_threshold = st.slider("Sensitivity (z-threshold)", 2.0, 5.0, value=DEFAULT_Z_THRESHOLD, step=0.5,
                              help="Lower = more sensitive (flags more points). Higher = stricter.")

if pd.Timestamp(start_date) >= pd.Timestamp(end_date):
    st.error("Start date must be before end date.")
    st.stop()

start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date) + pd.Timedelta(hours=23)

try:
    with st.spinner("Computing residuals and detecting anomalies..."):
        residual_df = compute_residuals_for_range(
            start_ts.strftime("%Y-%m-%d %H:%M:%S"), end_ts.strftime("%Y-%m-%d %H:%M:%S")
        )

        if len(residual_df) < DEFAULT_ROLLING_WINDOW + 1:
            st.warning(
                f"Selected range is too short for reliable detection "
                f"(need at least {DEFAULT_ROLLING_WINDOW + 1} hours). Please pick a longer range."
            )
            st.stop()

        anomaly_results = detect_anomalies(
            residual_df["residual"].values, window=DEFAULT_ROLLING_WINDOW, z_threshold=z_threshold
        )
        anomaly_results["timestamp"] = residual_df["timestamp"].values
        anomaly_results["actual"] = residual_df["actual"].values
        anomaly_results["predicted"] = residual_df["predicted"].values
        anomaly_results["severity"] = anomaly_results["z_score"].apply(
            lambda z: severity_label(z) if pd.notna(z) else "—"
        )
except Exception as e:
    st.error(f"Failed to compute anomalies: {e}")
    st.stop()

flagged = anomaly_results[anomaly_results["is_anomaly"]]

col_a, col_b = st.columns(2)
with col_a:
    st.metric("Total Hours Analyzed", len(anomaly_results))
with col_b:
    st.metric("Anomalies Detected", len(flagged))

# --- Chart: actual vs predicted with anomalies marked -----------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=anomaly_results["timestamp"], y=anomaly_results["actual"], mode="lines", name="Actual", line=dict(color="steelblue")))
fig.add_trace(go.Scatter(x=anomaly_results["timestamp"], y=anomaly_results["predicted"], mode="lines", name="Predicted", line=dict(color="gray", dash="dot")))
if len(flagged) > 0:
    fig.add_trace(go.Scatter(
        x=flagged["timestamp"], y=flagged["actual"], mode="markers", name="Anomaly",
        marker=dict(color="red", size=10, symbol="circle"),
    ))
fig.update_layout(title="Actual vs Predicted Demand with Detected Anomalies", xaxis_title="Time", yaxis_title="Demand (MW)", height=500)
st.plotly_chart(fig, use_container_width=True)

# --- Table ---------------------------------------------------------------------
st.subheader("Detected Anomalies")
if len(flagged) == 0:
    st.success("No anomalies detected in the selected range at this sensitivity.")
else:
    display_table = flagged[["timestamp", "actual", "predicted", "residual", "z_score", "severity"]].copy()
    display_table.columns = ["Timestamp", "Actual (MW)", "Predicted (MW)", "Residual (MW)", "Z-Score", "Severity"]
    for col in ["Actual (MW)", "Predicted (MW)", "Residual (MW)", "Z-Score"]:
        display_table[col] = display_table[col].round(2)
    st.dataframe(display_table, use_container_width=True, hide_index=True)