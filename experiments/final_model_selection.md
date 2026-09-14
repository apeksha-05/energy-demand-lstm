# Final Model Selection

## Decision
Single-layer, multivariate LSTM selected as the production forecasting model.

- Lookback: 72 hours
- Forecast horizon: 24 hours (direct multi-output)
- Hidden size: 64
- Parameters: 23,576
- Test set performance: MAE=67.42 MW, RMSE=93.77 MW, MAPE=5.77%, R²=0.742

## Models compared (11 total)
See experiments/final_model_comparison.csv for full metrics.

## Rationale
1. **Accuracy**: 2nd best of 11 models (MAE 67.42), only 1.5% behind BiLSTM.
2. **Deployment-appropriateness**: Unlike BiLSTM, uses only past data —
   suitable for real-time forecasting where no future data exists.
3. **Cost**: 23,576 parameters vs BiLSTM's 47,128, Stacked LSTM's 56,856,
   Transformer's 36,376 — best accuracy-per-parameter among gated
   recurrent models.
4. **Stability**: Consistent, reproducible convergence across every
   retraining in this project (Phases 8, 9, 12, 17, 18, 19).
5. **Generalization**: Consistent performance across weekday/weekend/
   holiday segments (Phase 18); known weaknesses (peak hours, COVID-19
   period) are explainable distribution-shift limitations shared by
   every model tested, not unique flaws.
6. **Interpretability**: Permutation importance (Phase 19) successfully
   extracted genuine, actionable insight from this exact model.

## Models explicitly excluded and why
- **BiLSTM**: Marginally better accuracy, but architecturally
  inappropriate for real-time deployment (Phase 11).
- **Stacked LSTM**: More parameters, worse accuracy — overfitting
  (Phase 10).
- **Transformer**: More parameters, much slower, worse accuracy —
  dataset likely too modest to leverage its flexibility advantage
  (Phase 15).
- **Attention-LSTM**: Inconclusive due to under-convergence within our
  compute-constrained epoch budget (Phase 13) — not a definitive
  rejection of attention mechanisms for this task.

## Known limitations (see Phase 18 error analysis)
- Peak-hour (11am-2pm) forecasts are ~71% less accurate than trough-hour
  forecasts.
- Performance degraded significantly during the COVID-19 demand
  disruption (March-June 2020), a genuine out-of-distribution event no
  model in this comparison could have anticipated.
- Forecast accuracy degrades ~28% from hour+1 to hour+24 of the horizon
  (Phase 12), and non-monotonically due to the daily demand cycle.