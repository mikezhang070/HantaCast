"""Hyperparameter tuning for HantaCast."""

from __future__ import annotations

import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.hantacast import HantaCast
from src.utils.seed import set_global_seed

log = logging.getLogger(__name__)

# Tuning grid — focused on small-data overfitting prevention
TUNE_GRID = {
    "lookback": [3, 5],
    "mixlinear_dropout": [0.1, 0.3, 0.5],
    "mixlinear_wd": [0.001, 0.01, 0.05],
    "mixlinear_lr": [0.005, 0.01],
    "seird_mixlinear_scale": [0.01, 0.05, 0.10, 0.15],
}


def _rolling_origin_mae(model_cls, df: pd.DataFrame, lookback: int, horizon: int, seed: int, **kwargs) -> float:
    """Compute rolling-origin MAE for a single HantaCast configuration."""
    set_global_seed(seed)
    df = df.copy()
    df["new_cases"] = pd.to_numeric(df["new_cases"], errors="coerce").fillna(0.0)
    n_obs = len(df)

    start_origin = lookback + horizon + 1
    if start_origin > n_obs - horizon:
        return float("inf")

    errors = []
    n_origins = 0
    for origin in range(start_origin, n_obs - horizon + 1):
        train_slice = df.iloc[:origin].copy()
        test_slice = df.iloc[origin : origin + horizon].copy()
        future_covariates = test_slice[
            [col for col in ["date", "day_index", "intervention_index", "mobility_index",
                             "flight_volume", "behavior_response_index", "location"]
             if col in test_slice.columns]
        ].copy()

        try:
            model = model_cls(lookback=lookback, horizon=horizon, random_state=seed, **kwargs)
            model.fit(train_slice)
            pred_df = model.predict(horizon=horizon, future_covariates=future_covariates).iloc[:len(test_slice)]
            y_pred = pred_df["forecast_median"].to_numpy(dtype=float)
            y_true = test_slice["new_cases"].to_numpy(dtype=float)
            errors.extend(np.abs(y_true - y_pred).tolist())
            n_origins += 1
        except Exception as e:
            log.warning(f"  Config failed at origin {origin}: {e}")
            continue

    if n_origins == 0:
        return float("inf")
    return float(np.mean(errors))


def tune_hantacast(
    df: pd.DataFrame,
    config: dict,
    output_dir: str | Path = "outputs",
) -> dict:
    """Run hyperparameter search for HantaCast.

    Evaluates each configuration with rolling-origin MAE.
    Returns the best config and a sorted results table.
    """
    output_dir = Path(output_dir)
    seed = int(config.get("seed", 42))
    horizon = int(config.get("training", {}).get("horizon", 1))

    # Generate grid
    keys = list(TUNE_GRID.keys())
    values = list(TUNE_GRID.values())
    total = 1
    for v in values:
        total *= len(v)
    print(f"\n  Tuning HantaCast over {total} configurations ...")

    results = []
    best_mae = float("inf")
    best_config = None

    for idx, combo in enumerate(itertools.product(*values)):
        params = dict(zip(keys, combo))
        lookback = params.pop("lookback")
        mixlinear_dropout = params.pop("mixlinear_dropout")
        mixlinear_wd = params.pop("mixlinear_wd")
        mixlinear_lr = params.pop("mixlinear_lr")
        seird_mixlinear_scale = params.pop("seird_mixlinear_scale")

        mae = _rolling_origin_mae(
            HantaCast, df, lookback, horizon, seed,
            mixlinear_dropout=mixlinear_dropout,
            mixlinear_wd=mixlinear_wd,
            mixlinear_lr=mixlinear_lr,
            seird_mixlinear_scale=seird_mixlinear_scale,
        )

        result = {
            "config_id": idx + 1,
            "lookback": lookback,
            "dropout": mixlinear_dropout,
            "wd": mixlinear_wd,
            "lr": mixlinear_lr,
            "signal_scale": seird_mixlinear_scale,
            "MAE": round(mae, 6),
        }
        results.append(result)

        if mae < best_mae:
            best_mae = mae
            best_config = {
                "lookback": lookback,
                "mixlinear_dropout": mixlinear_dropout,
                "mixlinear_wd": mixlinear_wd,
                "mixlinear_lr": mixlinear_lr,
                "seird_mixlinear_scale": seird_mixlinear_scale,
            }

        if (idx + 1) % 10 == 0 or idx + 1 == total:
            print(f"    {idx+1}/{total} done  |  best MAE so far: {best_mae:.4f}")

    results_df = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)
    results_df.to_csv(output_dir / "metrics" / "hantacast_tuning.csv", index=False)

    # Print top 10
    print(f"\n  Top 10 HantaCast configurations:")
    print(f"  {'ID':<5} {'L':<4} {'dropout':<8} {'wd':<8} {'lr':<8} {'scale':<8} {'MAE':<8}")
    print(f"  {'-'*55}")
    for _, row in results_df.head(10).iterrows():
        print(f"  {int(row['config_id']):<5} {int(row['lookback']):<4} "
              f"{float(row['dropout']):<8.3f} {float(row['wd']):<8.4f} "
              f"{float(row['lr']):<8.4f} {float(row['signal_scale']):<8.3f} "
              f"{float(row['MAE']):<8.4f}")

    print(f"\n  Best config: L={best_config['lookback']}, "
          f"dropout={best_config['dropout']}, wd={best_config['mixlinear_wd']}, "
          f"lr={best_config['mixlinear_lr']}, signal_scale={best_config['seird_mixlinear_scale']}")
    print(f"  Best MAE: {best_mae:.4f}")

    return best_config
