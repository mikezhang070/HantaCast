#!/usr/bin/env python
"""Generate future forecasts from trained HantaCast model.

Usage:
    python scripts/forecast_hantacast.py --config configs/default.yaml --checkpoint outputs/checkpoints/hantacast_best.pt --horizon 150
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import load_standardized_data
from src.models.hantacast import HantaCast
from src.utils.io import ensure_dir, save_dataframe, setup_logger
from src.utils.seed import set_global_seed

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate HantaCast forecasts")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/hantacast_best.pt", help="Path to model checkpoint")
    parser.add_argument("--horizon", type=int, default=150, help="Forecast horizon (days)")
    parser.add_argument("--step-horizon", type=int, default=1, help="Recursive step size")
    parser.add_argument("--data", type=str, default=None, help="Override training data path")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = int(config.get("seed", 42))
    set_global_seed(seed)

    # Setup logging
    output_root = Path(config.get("output", {}).get("root", "outputs"))
    log_path = output_root / "logs" / "forecast.log"
    setup_logger(log_path)

    log.info("=" * 60)
    log.info(f"HantaCast Forecast — {args.horizon}-day horizon")
    log.info(f"Config: {config_path.resolve()}")
    log.info(f"Checkpoint: {args.checkpoint}")
    log.info(f"Horizon: {args.horizon} days")
    log.info(f"Step horizon: {args.step_horizon}")
    log.info("=" * 60)

    # Check checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        msg = f"Checkpoint not found: {ckpt_path}. Run train_hantacast.py first."
        log.error(msg)
        print(f"ERROR: {msg}")
        return

    # Load data
    data_path = args.data or config.get("data", {}).get("train_path", "data/processed/case_timeseries_standardized.csv")
    log.info(f"Loading data from: {data_path}")
    df = load_standardized_data(data_path)
    log.info(f"Data loaded: {len(df)} observations")

    # Build and load model
    model = HantaCast(
        lookback=int(config.get("training", {}).get("lookback", 5)),
        horizon=int(config.get("training", {}).get("horizon", 1)),
        random_state=seed,
    )
    model.load_checkpoint(ckpt_path)
    model.mixlinear.train_df = df.copy()
    model.train_df = df.copy()
    model.seird.train_df = df.copy()

    # Pre-train MixLinear on full data for stable signal
    model.fit_mixlinear_full(df)
    model._is_fitted = True

    # Generate forecast
    n_samples = int(config.get("forecast", {}).get("n_samples", 80))
    forecast = model.forecast(
        total_horizon=args.horizon,
        n_samples=n_samples,
    )

    # Save
    output_path = output_root / "predictions" / f"forecast_{args.horizon}day.csv"
    save_dataframe(forecast, output_path)
    log.info(f"Forecast saved to {output_path}")
    log.info(f"Forecast shape: {forecast.shape}")
    log.info(f"Forecast columns: {list(forecast.columns)}")

    # Print summary
    print(f"\nForecast saved to: {output_path.resolve()}")
    print(f"Forecast head ({args.horizon} days):")
    print(forecast[["date", "forecast_median", "forecast_lower", "forecast_upper"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
