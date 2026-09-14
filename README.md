# Multivariate Energy Demand Forecasting and Anomaly Detection Using LSTM

A complete, end-to-end deep learning system for forecasting Panama's national electricity demand and detecting anomalous consumption patterns -- built as a comparative study across 8 neural sequence architectures (RNN, LSTM, Stacked LSTM, BiLSTM, GRU, Attention-LSTM, Transformer) plus classical baselines.

---

## Project Overview

This project forecasts hourly electricity demand 24 hours into the future using 5.5 years (2015-2020) of historical demand, weather, and calendar data for Panama. Beyond forecasting, it detects anomalous demand events using forecasting residuals, and packages everything into an interactive Streamlit dashboard.

The project is explicitly designed as **both** a working application **and** a practical study of why LSTM works for time-series forecasting -- every architectural decision is measured empirically, not assumed.

## Problem Statement

Accurate short-term electricity demand forecasting is essential for grid operators to balance supply and demand, schedule generation, and detect abnormal events (equipment failures, unexpected consumption spikes/drops) before they cause instability. This project builds a system that:
- Forecasts demand 24 hours ahead using historical demand, weather, and calendar features.
- Detects anomalous demand automatically using model residuals.
- Compares 8+ modeling approaches empirically to justify architecture choice.

## Objectives

- Build a complete, leakage-safe time-series ML pipeline from raw data to deployed dashboard.
- Empirically compare RNN, LSTM, GRU, Attention, and Transformer architectures on the same task.
- Demonstrate deep understanding of *why* LSTM works (gating, cell state, vanishing gradients) rather than treating it as a black box.
- Detect real anomalies using forecasting residuals with a defensible, dynamic thresholding method.
- Ship a usable, interactive forecasting dashboard.

## Dataset

**Source**: [Short-term Electricity Load Forecasting (Panama Case Study)](https://www.kaggle.com/datasets/ernestojaguilar/shortterm-electricity-load-forecasting-panama), originally published as academic research data by Aguilar Madrid (2021) on Mendeley Data.

- **48,048 hourly observations**, January 2015 - June 2020.
- **Target**: national electricity demand (MW).
- **Features**: temperature, humidity, precipitation proxy, and wind speed across 3 weather stations; holiday and school-day flags.
- No missing values or duplicate timestamps in the raw data (verified in this project's EDA).

## Features Engineered

- **National weather averages** (collapsing 3 stations into 4 features) to reduce multicollinearity.
- **Cyclical calendar encodings** (`sin`/`cos` of hour, day-of-week, month) to correctly represent circular time.
- **Lag features** (`lag_1/24/48/168`) and **rolling statistics** (`rolling_mean_24/std_24/mean_168`), computed leakage-safely (shifted before rolling).

## Architecture

```
Raw Data -> Cleaning -> Feature Engineering -> Chronological Split -> Scaling
    -> Sequence Generation (72h lookback, 24h horizon)
    -> Model Training (8 architectures compared)
    -> Evaluation & Error Analysis -> Anomaly Detection
    -> Final Model Selection -> Saved Artifacts -> Inference Pipeline -> Dashboard
```

Training and inference are strictly separated: inference only ever loads previously-fit scalers and trained weights, never refitting anything -- preventing any train/inference preprocessing mismatch.

## Why LSTM?

Vanilla RNNs suffer from vanishing gradients over long sequences -- the training signal for long-range dependencies (e.g., weekly seasonality, 168 hours back) shrinks exponentially as it backpropagates through time. LSTM solves this with a **cell state** -- an additive memory pathway modified by three gates (forget, input, output) -- that allows gradients to flow across long sequences largely unimpeded, rather than being repeatedly squashed through a nonlinearity at every timestep.

This project empirically confirms this: Simple RNN (MAE 76.72) underperforms both GRU (71.91) and LSTM (67.42) on identical data, directly demonstrating gating's real, measurable value.

### The Mathematics

```
f_t = sigmoid(W_f . [h_(t-1), x_t] + b_f)      # Forget gate
i_t = sigmoid(W_i . [h_(t-1), x_t] + b_i)      # Input gate
C~_t = tanh(W_C . [h_(t-1), x_t] + b_C)        # Candidate cell state
C_t = f_t * C_(t-1) + i_t * C~_t               # Cell state update (additive!)
o_t = sigmoid(W_o . [h_(t-1), x_t] + b_o)      # Output gate
h_t = o_t * tanh(C_t)                          # Hidden state
```

The cell state update is the key: it's dominated by element-wise multiplication and addition, not a repeated nonlinear transformation -- this is what lets gradients survive long backward passes.

## Models Compared

| Model | MAE (MW) | RMSE (MW) | R2 | Parameters |
|---|---|---|---|---|
| Naive (persistence) | 165.83 | 213.04 | -0.330 | -- |
| Moving Average | 132.92 | 165.12 | 0.201 | -- |
| Linear Regression | 92.60 | 117.56 | 0.595 | -- |
| Simple RNN | 76.72 | 102.87 | 0.690 | 7,064 |
| Univariate LSTM | 79.76 | 106.03 | 0.670 | 20,248 |
| **LSTM (multivariate) -- SELECTED** | **67.42** | **93.77** | **0.742** | **23,576** |
| Stacked LSTM | 69.66 | 97.96 | 0.718 | 56,856 |
| BiLSTM (experiment only) | 66.39 | 91.86 | 0.752 | 47,128 |
| GRU | 71.91 | 97.21 | 0.723 | 18,072 |
| Attention-LSTM | 86.77 | 112.72 | 0.627 | 27,800 |
| Transformer | 73.77 | 103.45 | 0.686 | 36,376 |

Full results: [`experiments/final_model_comparison.csv`](experiments/final_model_comparison.csv)

### Key Findings

- **Every neural sequence model beats every non-sequence baseline** -- validating the core premise of using recurrent architectures for this task.
- **Gating matters**: LSTM/GRU clearly beat vanilla RNN.
- **More complexity is not automatically better**: Stacked LSTM (2.4x more parameters) and Transformer (more parameters, 3x longer training) both *underperformed* the single-layer LSTM -- added capacity wasn't justified by this dataset's size/structure.
- **BiLSTM scored marginally best** (66.39 vs 67.42 MAE) but is **excluded from production**: real-time forecasting deployment has no legitimate access to "future" data, and the ~1.5% gain doesn't justify the architectural risk and 2x compute cost.
- **Multivariate features add real value**: multivariate LSTM beat univariate LSTM by ~15.5% MAE, confirming weather/calendar features carry genuine predictive signal.

## Anomaly Detection

Uses a **dynamic rolling z-score of forecasting residuals** (actual minus predicted), rather than a fixed threshold, since demand volatility varies by time of day. The detector successfully and independently re-identified a real extreme demand event (January 20, 2019) and the onset of Panama's COVID-19 demand disruption (April 2020) -- both previously found only through manual data inspection during EDA.

```
z(t) = (residual(t) - rolling_mean(t)) / rolling_std(t)
anomaly(t) = |z(t)| > 3.0
```

## Known Limitations

- **Peak-hour forecasts (11am-2pm) are ~71% less accurate** than trough-hour forecasts (51.4 vs 87.9 MAE) -- the model struggles most exactly when demand is highest.
- **Forecast accuracy degrades non-monotonically with horizon distance**, worsening ~28% from hour+1 to hour+24, modulated by the daily demand cycle rather than degrading smoothly.
- **COVID-19 caused a genuine, unforecastable distribution shift** in March-June 2020 that no model in this comparison could anticipate -- a fundamental limitation of any historically-trained model facing an unprecedented event.
- Rolling/dynamic anomaly thresholds can become desensitized during sustained volatile periods (documented explicitly in [Phase 17 analysis](experiments/anomaly_detection_summary.json)).

## Installation

**Requirements**: Python 3.11, Git, ~500MB disk space.

```powershell
git clone https://github.com/apeksha-05/energy-demand-lstm.git
cd energy-demand-lstm
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Get the dataset** (not included in this repo due to size):
1. Download from [Kaggle](https://www.kaggle.com/datasets/ernestojaguilar/shortterm-electricity-load-forecasting-panama).
2. Extract and place `continuous_dataset.csv` in `data/raw/`.

## Usage

**Train the final model** (produces `models/lstm_final.pt`, `models/scaler.pkl`, `models/feature_config.json`):
```powershell
jupyter notebook notebooks/15_final_model_training.ipynb
```
Then run all cells.

**Run inference from the command line:**
```powershell
python -m src.inference "2020-01-15 12:00:00"
```

**Run the dashboard:**
```powershell
streamlit run app/app.py
```

**Run tests:**
```powershell
pytest tests/ -v
```

## Project Structure

```
energy-demand-lstm/
|-- data/                  # raw/interim/processed (raw data not included -- see Installation)
|-- notebooks/             # 15 notebooks, one per experimental phase
|-- src/                   # production pipeline code
|   |-- models/            # 8 architecture implementations
|   |-- training/          # shared training loop, early stopping
|   |-- evaluation/        # metrics, error analysis
|   |-- anomaly/           # residual-based anomaly detector
|   `-- inference.py       # save/load artifacts, forecast generation
|-- app/                   # Streamlit dashboard (5 pages)
|-- tests/                 # pytest suite (36 tests)
|-- experiments/           # saved results -- real, measured, never invented
`-- models/                # trained artifacts (not included -- regenerate via notebook)
```

## Technologies Used

Python 3.11, PyTorch, pandas, NumPy, scikit-learn, Streamlit, Plotly, Matplotlib/Seaborn, pytest

## Future Improvements

- Complementary absolute-magnitude anomaly threshold alongside the relative/rolling one.
- Systematic hyperparameter tuning (Attention-LSTM was under-converged within compute budget; worth revisiting with more epochs).
- Real-time weather API integration for true live forecasting rather than historical "as-of" queries.
- Ablation study on per-city vs. averaged weather features.

## Author

Built as a final-year machine learning project by **Apeksha**.

## License

This project is licensed under the MIT License -- see the [LICENSE](LICENSE) file for details.