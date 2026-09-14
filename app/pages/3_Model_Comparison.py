"""
Model Comparison page — displays the full comparison of all architectures
tested in this project (Phases 6-16), loaded directly from the saved
experiment results (never recomputed or invented here).
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

st.set_page_config(page_title="Model Comparison", page_icon="📊", layout="wide")
st.title("📊 Model Comparison")
st.markdown(
    "All metrics below are real, measured results from training and evaluating "
    "each architecture on the same test set (see Phases 6–16). No results are "
    "invented or estimated."
)

comparison_path = Path(__file__).resolve().parent.parent.parent / "experiments" / "final_model_comparison.csv"

if not comparison_path.exists():
    st.error(f"Comparison file not found at {comparison_path}. Run Phase 16's notebook first.")
    st.stop()

df = pd.read_csv(comparison_path, index_col=0)
df.index.name = "Model"

# --- Highlight the selected production model -----------------------------------
PRODUCTION_MODEL = "LSTM"
EXCLUDED_MODEL = "BiLSTM (experiment)"

st.subheader("Full Comparison Table")
st.caption(
    f"✅ **{PRODUCTION_MODEL}** is the selected production model. "
    f"⚠️ **{EXCLUDED_MODEL}** scored marginally better but is excluded from "
    f"production for architectural reasons (see Phase 11)."
)


def highlight_rows(row):
    if row.name == PRODUCTION_MODEL:
        return ["background-color: #1a3d1a"] * len(row)
    elif row.name == EXCLUDED_MODEL:
        return ["background-color: #3d2f1a"] * len(row)
    return [""] * len(row)


styled = df.style.apply(highlight_rows, axis=1).format(
    {"MAE": "{:.2f}", "RMSE": "{:.2f}", "MAPE": "{:.2f}", "sMAPE": "{:.2f}", "R2": "{:.3f}"}
)
st.dataframe(styled, use_container_width=True)

st.divider()

# --- Bar chart: MAE comparison ---------------------------------------------------
st.subheader("MAE Comparison Across Models")
df_sorted = df.sort_values("MAE")

fig_mae = px.bar(
    df_sorted, x=df_sorted.index, y="MAE",
    color=df_sorted.index.map(lambda m: "Production" if m == PRODUCTION_MODEL else ("Excluded" if m == EXCLUDED_MODEL else "Other")),
    color_discrete_map={"Production": "#2ecc71", "Excluded": "#f39c12", "Other": "#7f8c8d"},
    labels={"x": "Model", "color": "Status"},
)
fig_mae.update_layout(xaxis_title="Model", yaxis_title="MAE (MW)", height=500, xaxis_tickangle=-45)
st.plotly_chart(fig_mae, use_container_width=True)

# --- Scatter: accuracy vs cost trade-off ----------------------------------------
st.subheader("Accuracy vs. Computational Cost Trade-off")
if "num_parameters" in df.columns:
    df_params = df[df["num_parameters"] != "—"].copy()
    df_params["num_parameters"] = pd.to_numeric(df_params["num_parameters"], errors="coerce")
    df_params = df_params.dropna(subset=["num_parameters"])

    fig_scatter = px.scatter(
        df_params, x="num_parameters", y="MAE", text=df_params.index,
        labels={"num_parameters": "Number of Parameters", "MAE": "MAE (MW)"},
    )
    fig_scatter.update_traces(textposition="top center", marker=dict(size=12))
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.caption(
        "Note how Stacked LSTM and Transformer sit in the upper-right "
        "(more parameters, worse accuracy) — added complexity was not "
        "justified for this dataset (see Phases 10 and 15)."
    )

st.divider()

with st.expander("Read the full model selection rationale"):
    selection_path = Path(__file__).resolve().parent.parent.parent / "experiments" / "final_model_selection.txt"
    if selection_path.exists():
        st.text(selection_path.read_text())
    else:
        st.info("Selection rationale file not found.")