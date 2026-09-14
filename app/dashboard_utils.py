"""
Shared utilities for the Streamlit dashboard: cached data/model loading
so expensive operations (loading the CSV, loading the model) don't
re-run on every user interaction.
"""

import sys
from pathlib import Path

import streamlit as st

# Make the project's src/ package importable when Streamlit runs this file directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_raw_data
from src.preprocessing import clean_raw_data
from src.feature_engineering import build_features
from src.inference import load_model_artifacts
from src.config import TARGET_COL


@st.cache_data
def get_full_dataset():
    """
    Load and feature-engineer the full historical dataset once, cached
    across dashboard interactions (Streamlit re-runs the whole script on
    every widget interaction, so caching expensive I/O is essential).
    """
    df_raw = load_raw_data()
    df_clean = clean_raw_data(df_raw)
    df_features = build_features(df_clean)
    return df_features


@st.cache_resource
def get_model_artifacts():
    """
    Load the trained model, scaler, and feature config once, cached as a
    resource (not data) since it includes a PyTorch model object.
    """
    return load_model_artifacts()