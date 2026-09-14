"""
Data Exploration page — interactive EDA recreating Phase 2's key
findings: demand distribution, hourly/weekly/monthly patterns, and
weather relationships.
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dashboard_utils import get_full_dataset
from src.config import TARGET_COL

st.set_page_config(page_title="Data Exploration", page_icon="🔍", layout="wide")
st.title("🔍 Data Exploration")
st.markdown(
    "Explore the historical patterns underlying the forecasting model — "
    "these findings (Phase 2 of the project) directly motivated the "
    "feature engineering and model design choices used throughout."
)

df = get_full_dataset()

# --- Date range filter -----------------------------------------------------------
min_date, max_date = df.index.min().date(), df.index.max().date()
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("From", value=min_date, min_value=min_date, max_value=max_date, key="explore_start")
with col2:
    end_date = st.date_input("To", value=max_date, min_value=min_date, max_value=max_date, key="explore_end")

df_range = df.loc[str(start_date):str(end_date)]

if len(df_range) == 0:
    st.warning("No data in the selected range.")
    st.stop()

st.divider()

# --- Full time series -------------------------------------------------------------
st.subheader("Demand Over Time")
fig_ts = px.line(df_range, x=df_range.index, y=TARGET_COL, labels={"x": "Date", TARGET_COL: "Demand (MW)"})
fig_ts.update_layout(height=400)
st.plotly_chart(fig_ts, use_container_width=True)

# --- Distribution ------------------------------------------------------------------
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Demand Distribution")
    fig_hist = px.histogram(df_range, x=TARGET_COL, nbins=60, labels={TARGET_COL: "Demand (MW)"})
    fig_hist.update_layout(height=350)
    st.plotly_chart(fig_hist, use_container_width=True)

with col_b:
    st.subheader("Summary Statistics")
    stats = df_range[TARGET_COL].describe()
    st.dataframe(stats.to_frame(name="Demand (MW)").style.format("{:.1f}"), use_container_width=True)

st.divider()

# --- Hourly / weekly / monthly patterns --------------------------------------------
st.subheader("Recurring Patterns")
tab1, tab2, tab3 = st.tabs(["By Hour of Day", "By Day of Week", "By Month"])

with tab1:
    hourly_avg = df_range.groupby(df_range.index.hour)[TARGET_COL].mean().reset_index()
    hourly_avg.columns = ["Hour", "Average Demand (MW)"]
    fig_hour = px.bar(hourly_avg, x="Hour", y="Average Demand (MW)")
    fig_hour.update_layout(height=400)
    st.plotly_chart(fig_hour, use_container_width=True)
    st.caption("Peak demand typically occurs midday (~11am-2pm); trough overnight (~3-6am).")

with tab2:
    dow_series = df_range.groupby(df_range.index.dayofweek)[TARGET_COL].mean()
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_avg = pd.DataFrame({
        "Day": [dow_labels[i] for i in dow_series.index],
        TARGET_COL: dow_series.values,
    })
    fig_dow = px.bar(dow_avg, x="Day", y=TARGET_COL, labels={TARGET_COL: "Average Demand (MW)"})
    fig_dow.update_layout(height=400)
    st.plotly_chart(fig_dow, use_container_width=True)
    st.caption("Weekend demand (especially Sunday) is consistently lower than weekdays.")

with tab3:
    month_avg = df_range.groupby(df_range.index.month)[TARGET_COL].mean().reset_index()
    month_avg.columns = ["Month", "Average Demand (MW)"]
    fig_month = px.bar(month_avg, x="Month", y="Average Demand (MW)")
    fig_month.update_layout(height=400)
    st.plotly_chart(fig_month, use_container_width=True)
    st.caption("Panama's tropical climate produces relatively flat seasonality across months.")

st.divider()

# --- Weather relationship ----------------------------------------------------------
st.subheader("Weather Relationship")
weather_var = st.selectbox("Weather variable", ["temp_avg", "humidity_avg", "precip_avg", "wind_avg"])

sample_df = df_range.sample(min(5000, len(df_range)), random_state=42)
fig_weather = px.scatter(
    sample_df, x=weather_var, y=TARGET_COL, opacity=0.3,
    labels={weather_var: weather_var.replace("_", " ").title(), TARGET_COL: "Demand (MW)"},
)
fig_weather.update_layout(height=450)
st.plotly_chart(fig_weather, use_container_width=True)

correlation = df_range[weather_var].corr(df_range[TARGET_COL])
st.metric(f"Correlation ({weather_var} vs demand)", f"{correlation:.3f}")

st.divider()

# --- Holiday / school effect --------------------------------------------------------
st.subheader("Holiday & School-Day Effects")
col_c, col_d = st.columns(2)
with col_c:
    holiday_avg = df_range.groupby("holiday")[TARGET_COL].mean()
    st.metric("Normal Day Avg", f"{holiday_avg.get(0, float('nan')):,.0f} MW")
    st.metric("Holiday Avg", f"{holiday_avg.get(1, float('nan')):,.0f} MW")
with col_d:
    school_avg = df_range.groupby("school")[TARGET_COL].mean()
    st.metric("Non-School Day Avg", f"{school_avg.get(0, float('nan')):,.0f} MW")
    st.metric("School Day Avg", f"{school_avg.get(1, float('nan')):,.0f} MW")