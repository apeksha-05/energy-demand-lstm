"""
Energy Demand Forecasting Dashboard — Home page.

Run with: streamlit run app/app.py
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dashboard_utils import get_full_dataset, get_model_artifacts
from src.inference import generate_forecast
from src.config import TARGET_COL

st.set_page_config(
    page_title="Energy Demand Forecasting",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Panama Electricity Demand Forecasting")
st.caption("Multivariate LSTM-based demand forecasting and anomaly detection")

# --- Load data and model (cached) -------------------------------------------------
try:
    df = get_full_dataset()
    model, scaler, feature_config = get_model_artifacts()
except Exception as e:
    st.error(f"Failed to load data or model artifacts: {e}")
    st.info("Make sure you've run the training pipeline and saved model artifacts (Phase 20) before launching the dashboard.")
    st.stop()

latest_timestamp = df.index.max()
latest_demand = df.loc[latest_timestamp, TARGET_COL]

# --- Top-level metrics row ----------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Latest Recorded Demand", f"{latest_demand:,.0f} MW")

with col2:
    st.metric("Latest Data Timestamp", latest_timestamp.strftime("%Y-%m-%d %H:%M"))

with col3:
    st.metric("Model", "Single-layer LSTM")

with col4:
    st.metric("Test Set MAE", "67.42 MW")

st.divider()

# --- Quick forecast preview: next 24 hours from the latest available data ---------
st.subheader("Next 24-Hour Forecast (from latest available data)")

try:
    forecast = generate_forecast(latest_timestamp.strftime("%Y-%m-%d %H:%M:%S"))
    st.line_chart(forecast.set_index("timestamp")["predicted_demand_mw"])

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Next Hour Forecast", f"{forecast.iloc[0]['predicted_demand_mw']:,.0f} MW")
    with col_b:
        st.metric("24h Average Forecast", f"{forecast['predicted_demand_mw'].mean():,.0f} MW")

except Exception as e:
    st.warning(f"Could not generate a live forecast from the latest timestamp: {e}")

st.divider()
st.markdown(
    """
    ### About this project
    This dashboard forecasts Panama's national electricity demand using a
    multivariate LSTM trained on 5.5 years of hourly demand, weather, and
    calendar data. Use the sidebar to explore forecasts in detail, view
    detected anomalies, compare model architectures, and browse the
    underlying data.
    """
)