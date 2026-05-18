# Output Files

## Directory Structure

```
outputs/
├── checkpoints/
│   └── hantacast_best.pt       # Best model checkpoint (PyTorch format)
├── metrics/
│   ├── metrics.json            # Training metrics (SEIRD params, residual scale)
│   └── evaluation_metrics.json # Evaluation metrics (MAE, RMSE, MAPE, etc.)
├── predictions/
│   ├── test_predictions.csv    # Rolling-origin test predictions (if data permits)
│   └── forecast_150day.csv     # Future forecast (e.g., 150-day horizon)
└── logs/
    ├── train.log               # Training log
    ├── evaluate.log            # Evaluation log
    └── forecast.log            # Forecast log
```

## Checkpoint Format

`hantacast_best.pt` is a PyTorch checkpoint containing:
- MixLinear model state_dict
- MixLinear normalization statistics (mean, std)
- SEIRD best-fit parameters and compartment state
- Model configuration (lookback, horizon, seed)

## Metrics JSON Format

`metrics.json`:
```json
{
  "model": "HantaCast",
  "lookback": 3,
  "horizon": 1,
  "epochs": 80,
  "mixlinear_signal_scale": 0.15,
  "seird_best_params": { ... },
  "seird_residual_scale": 1.0,
  "checkpoint_path": "..."
}
```

`evaluation_metrics.json`:
```json
{
  "model": "HantaCast",
  "status": "evaluated",
  "MAE": 0.0,
  "RMSE": 0.0,
  "MAPE": 0.0,
  "SMAPE": 0.0,
  "MASE": 0.0,
  "peak_timing_error": 0.0,
  "peak_magnitude_error": 0.0,
  "final_cumulative_error": 0.0,
  "poisson_deviance": 0.0,
  "n_predictions": 0,
  "n_origins": 0
}
```

If the dataset is too small for rolling-origin evaluation, the status will be `"insufficient_data"`.

## Prediction CSV Format

`test_predictions.csv`:
- `forecast_origin_date` — Date of forecast origin
- `target_date` — Date being forecast
- `horizon_step` — Step within forecast horizon
- `lookback` — Lookback window used
- `horizon` — Forecast horizon
- `new_cases_pred` — Predicted new cases
- `actual_new_cases` — Actual new cases
- `model` — Model name (HantaCast)

`forecast_150day.csv`:
- `date` — Forecast date
- `day_index` — Day index
- `forecast_median` — Median forecast (non-negative integer)
- `forecast_lower` — 2.5th percentile
- `forecast_upper` — 97.5th percentile
- `interval_type` — Type of prediction interval
- `interval_source` — Source of interval estimation
