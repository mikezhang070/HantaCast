"""Evaluation utilities for HantaCast and all baselines."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.baselines import build_all_baselines
from src.models.hantacast import HantaCast
from src.utils.io import ensure_dir, save_dataframe, write_json
from src.utils.metrics import compute_all_metrics

log = logging.getLogger(__name__)


def evaluate_all_models(
    model: HantaCast,
    test_df: pd.DataFrame,
    config: dict,
    output_dir: str | Path = "outputs",
) -> pd.DataFrame:
    """Evaluate HantaCast + all baselines with rolling-origin.

    Returns a comparison DataFrame sorted by MAE.
    """
    output_dir = Path(output_dir)
    ensure_dir(output_dir / "metrics")
    ensure_dir(output_dir / "predictions")

    seed = int(config.get("seed", 42))
    lookback = int(config.get("training", {}).get("lookback", 3))
    horizon = int(config.get("training", {}).get("horizon", 1))

    df = test_df.copy()
    df["new_cases"] = pd.to_numeric(df["new_cases"], errors="coerce").fillna(0.0)
    n_obs = len(df)

    # Build all models
    baselines = build_all_baselines(lookback=lookback, horizon=horizon, random_state=seed)
    models: dict[str, object] = {**baselines}  # baselines are fitted fresh each origin
    model_table: dict[str, dict] = {name: {"type": "baseline"} for name in baselines}
    model_table["HantaCast"] = {"type": "hantacast", "model": model}

    print(f"\n  Evaluating {len(models) + 1} models on {n_obs} observations ...")

    all_rows = []
    start_origin = lookback + horizon + 1
    if start_origin > n_obs - horizon:
        log.warning(f"Dataset too small: need {start_origin + horizon} observations for rolling-origin, got {n_obs}")
        return pd.DataFrame()

    n_origins = n_obs - horizon - start_origin + 1
    if n_origins < 1:
        log.warning("No valid origins for rolling-origin evaluation.")
        return pd.DataFrame()

    n_models = len(models) + 1

    # ── Pre-train MixLinear on full data (frozen during evaluation) ──
    hantacast_model = model_table["HantaCast"]["model"]
    hantacast_model.fit_mixlinear_full(df)

    for origin in range(start_origin, n_obs - horizon + 1):
        train_slice = df.iloc[:origin].copy()
        test_slice = df.iloc[origin:origin + horizon].copy()
        future_covariates = test_slice[
            [col for col in ["date", "day_index", "intervention_index", "mobility_index",
                             "flight_volume", "behavior_response_index", "location"]
             if col in test_slice.columns]
        ].copy()

        # Evaluate baselines — fit fresh at each origin
        for name, baseline in models.items():
            try:
                baseline.fit(train_slice)
                pred_df = baseline.predict(horizon=horizon, future_covariates=future_covariates).iloc[:len(test_slice)]
            except Exception as exc:
                log.warning(f"  {name} failed at origin {origin}: {exc}")
                continue
            for i in range(len(pred_df)):
                all_rows.append({
                    "model": name,
                    "forecast_origin_date": train_slice["date"].iloc[-1],
                    "target_date": test_slice["date"].iloc[i],
                    "horizon_step": i + 1,
                    "lookback": lookback,
                    "horizon": horizon,
                    "new_cases_pred": float(pred_df["forecast_median"].iloc[i]),
                    "actual_new_cases": float(test_slice["new_cases"].iloc[i]),
                })

        # Evaluate HantaCast — MixLinear frozen, only SEIRD refit
        try:
            hantacast_model = model_table["HantaCast"]["model"]
            hantacast_model.fit_seird_only(train_slice)
            pred_df = hantacast_model.predict(horizon=horizon, future_covariates=future_covariates).iloc[:len(test_slice)]
        except Exception as exc:
            log.warning(f"  HantaCast failed at origin {origin}: {exc}")
            continue
        for i in range(len(pred_df)):
            all_rows.append({
                "model": "HantaCast",
                "forecast_origin_date": train_slice["date"].iloc[-1],
                "target_date": test_slice["date"].iloc[i],
                "horizon_step": i + 1,
                "lookback": lookback,
                "horizon": horizon,
                "new_cases_pred": float(pred_df["forecast_median"].iloc[i]),
                "actual_new_cases": float(test_slice["new_cases"].iloc[i]),
            })

        if origin == start_origin or origin % max(1, n_origins // 3) == 0 or origin == n_obs - horizon:
            print(f"    origin {origin - start_origin + 1}/{n_origins} done")

    if not all_rows:
        log.warning("No evaluation results generated.")
        return pd.DataFrame()

    details_df = pd.DataFrame(all_rows)
    insample = pd.to_numeric(df["new_cases"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    # Compute per-model metrics
    comparison_rows = []
    for model_name, group in details_df.groupby("model", sort=True):
        y_true = group["actual_new_cases"].to_numpy(dtype=float)
        y_pred = group["new_cases_pred"].to_numpy(dtype=float)
        metrics = compute_all_metrics(y_true, y_pred, insample=insample)
        comparison_rows.append({
            "model": model_name,
            **metrics,
            "n_predictions": int(len(group)),
            "n_origins": int(group["forecast_origin_date"].nunique()),
        })

    comparison_df = pd.DataFrame(comparison_rows).sort_values("MAE").reset_index(drop=True)
    comparison_df["MAE_rank"] = comparison_df["MAE"].rank(method="dense", ascending=True).astype(int)
    comparison_df["RMSE_rank"] = comparison_df["RMSE"].rank(method="dense", ascending=True).astype(int)

    # Save
    save_dataframe(details_df, output_dir / "predictions" / "test_predictions.csv")
    save_dataframe(comparison_df, output_dir / "metrics" / "model_comparison.csv")
    write_json(comparison_df.to_dict(orient="records"), output_dir / "metrics" / "evaluation_metrics.json")

    return comparison_df


def _print_comparison_table(comparison_df: pd.DataFrame):
    """Pretty-print model comparison table."""
    if comparison_df.empty:
        print("  (no results)")
        return

    header = f"  {'Rank':<5} {'Model':<22} {'MAE':>8} {'RMSE':>8} {'MAPE':>8} {'SMAPE':>8} {'MASE':>8} {'N':>5}"
    print(f"\n{'=' * len(header)}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for _, row in comparison_df.iterrows():
        rank = int(row.get("MAE_rank", 0))
        print(
            f"  {rank:<5} "
            f"{str(row['model']):<22} "
            f"{float(row['MAE']):>8.4f} "
            f"{float(row['RMSE']):>8.4f} "
            f"{float(row['MAPE']):>7.1f}% "
            f"{float(row['SMAPE']):>7.1f}% "
            f"{float(row['MASE']):>8.4f} "
            f"{int(row['n_predictions']):>5}"
        )
    print()
