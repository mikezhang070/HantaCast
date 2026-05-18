#!/usr/bin/env python
"""Evaluate HantaCast model.

Usage:
    python scripts/evaluate_hantacast.py --config configs/default.yaml --checkpoint outputs/checkpoints/hantacast_best.pt
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
from src.training.evaluate import _print_comparison_table, evaluate_all_models
from src.utils.io import setup_logger
from src.utils.seed import set_global_seed

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate HantaCast model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/hantacast_best.pt", help="Path to model checkpoint")
    parser.add_argument("--data", type=str, default=None, help="Override test data path")
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
    log_path = output_root / "logs" / "evaluate.log"
    setup_logger(log_path)

    log.info("=" * 60)
    log.info("HantaCast Evaluation")
    log.info(f"Config: {config_path.resolve()}")
    log.info(f"Checkpoint: {args.checkpoint}")
    log.info("=" * 60)

    # Check checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        msg = f"Checkpoint not found: {ckpt_path}. Run train_hantacast.py first."
        log.error(msg)
        result = {"status": "no_checkpoint", "message": msg}
        write_json(result, output_root / "metrics" / "evaluation_metrics.json")
        print(f"ERROR: {msg}")
        return

    # Load data
    data_path = args.data or config.get("data", {}).get("test_path", "data/processed/case_timeseries_standardized.csv")
    log.info(f"Loading data from: {data_path}")
    test_df = load_standardized_data(data_path)
    log.info(f"Data loaded: {len(test_df)} observations")

    # Build and load model
    model = HantaCast(
        lookback=int(config.get("training", {}).get("lookback", 3)),
        horizon=int(config.get("training", {}).get("horizon", 1)),
        random_state=seed,
    )
    model.load_checkpoint(ckpt_path)
    model.mixlinear.train_df = test_df.copy()
    model.train_df = test_df.copy()

    # Evaluate all models
    comparison_df = evaluate_all_models(model, test_df, config, output_dir=output_root)
    if comparison_df.empty:
        log.warning("No evaluation results — dataset too small.")
    else:
        _print_comparison_table(comparison_df)


if __name__ == "__main__":
    main()
