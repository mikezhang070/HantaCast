# HantaCast

HantaCast: A Deep Mechanistically Constrained Hybrid Model for Hantavirus Case Trend Forecasting.

## Model Description

HantaCast integrates a **MixLinear-based deep temporal signal learner** with **SEIRD-constrained epidemiological dynamics** into a unified hybrid model.

## Directory Structure

```
HantaCast_train_minimal/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── data/
│   ├── README_data.md
│   ├── raw/
│   └── processed/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── preprocess.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── hantacast.py
│   │   ├── mixlinear.py
│   │   ├── seird_dynamics.py
│   │   └── behavior_response.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── metrics.py
│   └── utils/
│       ├── __init__.py
│       ├── seed.py
│       └── io.py
├── scripts/
│   ├── train_hantacast.py
│   ├── evaluate_hantacast.py
│   ├── forecast_hantacast.py
│   └── run_all.py
└── outputs/
    ├── README_outputs.md
    ├── checkpoints/
    ├── metrics/
    ├── predictions/
    └── logs/
```

## Environment Setup

```bash
pip install -r requirements.txt
```

## Data

See `data/README_data.md` for dataset details.

The training uses a standardized aggregate case-count time series located at `data/processed/case_timeseries_standardized.csv`. Raw input files are in `data/raw/`.

**All data is aggregate-level only. No personally identifiable information is included.**

## Train + Evaluate (single command)

```bash
# Full training + rolling-origin evaluation
python scripts/train_hantacast.py --config configs/default.yaml

# Smoke test (1 epoch, quick verification)
python scripts/train_hantacast.py --config configs/default.yaml --smoke-test
```

Outputs:
- `outputs/checkpoints/hantacast_best.pt` — Model checkpoint
- `outputs/metrics/metrics.json` — Training metrics (SEIRD params, residual scale)
- `outputs/metrics/evaluation_metrics.json` — Evaluation metrics (MAE, RMSE, MAPE, etc.)
- `outputs/predictions/test_predictions.csv` — Rolling-origin prediction details
- `outputs/logs/train.log` — Full training + evaluation log

Console output includes per-epoch loss and a final metrics table.

## Forecasting

```bash
python scripts/forecast_hantacast.py --config configs/default.yaml --checkpoint outputs/checkpoints/hantacast_best.pt --horizon 150
```

Outputs:
- `outputs/predictions/forecast_150day.csv`

Forecast columns:
- `date`, `day_index` — Date and day index
- `forecast_median` — Median forecast (non-negative integer)
- `forecast_lower` — 2.5th percentile (non-negative integer)
- `forecast_upper` — 97.5th percentile (non-negative integer)
- `interval_type`, `interval_source` — Interval metadata

## One-Click Run

```bash
python scripts/run_all.py
python scripts/run_all.py --smoke-test
```

Runs training + evaluation + 150-day forecast sequentially.

## Reproducibility

- All random seeds are fixed via `configs/default.yaml` (seed: 42).
- Deterministic PyTorch settings are enabled.
- Time-ordered data splits are used (no random shuffling).
- Model architecture is fully specified in source code.

