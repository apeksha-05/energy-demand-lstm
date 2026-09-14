"""
Forecast page — user selects a date/time and views the resulting
24-hour forecast with historical context.
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dashboard_utils import get_full_dataset, get_model_artifacts
from src.inference import generate_forecast
from src.config import TARGET_COL

st.set_page_config(page_title="Forecast", page_icon="📈", layout="wide")
st.title("📈 Demand Forecast")

df = get_full_dataset()
model, scaler, feature_config = get_model_artifacts()

min_date = df.index.min()
max_date = df.index.max()

st.markdown("Select a date and hour to forecast the **next 24 hours** from that point.")

col1, col2 = st.columns(2)
with col1:
    selected_date = st.date_input(
        "Date",
        value=max_date.date(),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )
with col2:
    selected_hour = st.slider("Hour of day", 0, 23, value=max_date.hour)

as_of = pd.Timestamp(selected_date) + pd.Timedelta(hours=selected_hour)

if as_of not in df.index:
    st.error(f"No data available for {as_of}. Please pick a different date/time.")
    st.stop()

try:
    forecast = generate_forecast(as_of.strftime("%Y-%m-%d %H:%M:%S"))
except Exception as e:
    st.error(f"Forecast generation failed: {e}")
    st.stop()

# --- Historical context: show the 72h lookback window + the forecast together ---
lookback_hours = feature_config["lookback"]
history_window = df.loc[:as_of, TARGET_COL].tail(lookback_hours)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=history_window.index, y=history_window.values,
    mode="lines", name="Recent Actual Demand", line=dict(color="steelblue"),
))
fig.add_trace(go.Scatter(
    x=forecast["timestamp"], y=forecast["predicted_demand_mw"],
    mode="lines+markers", name="Forecast (next 24h)", line=dict(color="orange", dash="dash"),
))
as_of_str = as_of.strftime("%Y-%m-%d %H:%M:%S")
fig.add_shape(
    type="line",
    x0=as_of_str, x1=as_of_str,
    y0=0, y1=1,
    yref="paper",  # spans the full height of the plot regardless of y-axis scale
    line=dict(color="gray", dash="dot"),
)
fig.add_annotation(
    x=as_of_str, y=1, yref="paper",
    text="Forecast start", showarrow=False, yshift=10,
)
fig.update_layout(
    title=f"Demand: {lookback_hours}h History + 24h Forecast (as of {as_of})",
    xaxis_title="Time", yaxis_title="Demand (MW)",
    height=500,
)
st.plotly_chart(fig, use_container_width=True)

# --- Summary metrics ---
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("Hour+1 Forecast", f"{forecast.iloc[0]['predicted_demand_mw']:,.0f} MW")
with col_b:
    st.metric("Peak Forecast (24h)", f"{forecast['predicted_demand_mw'].max():,.0f} MW")
with col_c:
    st.metric("Trough Forecast (24h)", f"{forecast['predicted_demand_mw'].min():,.0f} MW")

st.info(
    "Note: forecast accuracy decreases with horizon distance — hour+1 "
    "predictions are typically more reliable than hour+24 (see Phase 12 "
    "error analysis: ~28% higher error at hour+24 vs hour+1)."
)

with st.expander("View forecast as a table"):
    st.dataframe(forecast, use_container_width=True)