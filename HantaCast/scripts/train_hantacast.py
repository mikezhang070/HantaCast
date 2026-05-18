#!/usr/bin/env python
"""Train and evaluate HantaCast model.

Usage:
    python scripts/train_hantacast.py --config configs/default.yaml
    python scripts/train_hantacast.py --config configs/default.yaml --smoke-test
    python scripts/train_hantacast.py --config configs/default.yaml --skip-eval
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
from src.training.train import train_hantacast
from src.training.tune import tune_hantacast
from src.utils.io import setup_logger
from src.utils.seed import set_global_seed

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train, tune, and evaluate HantaCast model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--smoke-test", action="store_true", help="Run a minimal smoke test (1 epoch)")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning before training")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation after training")
    parser.add_argument("--data", type=str, default=None, help="Override training data path")
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = int(config.get("seed", 42))
    set_global_seed(seed)

    output_root = Path(config.get("output", {}).get("root", "outputs"))
    log_path = output_root / "logs" / "train.log"
    setup_logger(log_path)

    log.info("=" * 60)
    log.info("HantaCast — Train + Evaluate")
    log.info(f"Config: {config_path.resolve()}")
    log.info(f"Smoke test: {args.smoke_test}")
    log.info(f"Seed: {seed}")
    log.info("=" * 60)

    # ── Load data ─────────────────────────────────────────────────
    data_path = args.data or config.get("data", {}).get("train_path", "data/processed/case_timeseries_standardized.csv")
    log.info(f"Loading data from: {data_path}")
    df = load_standardized_data(data_path)
    log.info(f"Data loaded: {len(df)} observations, {df.shape[1]} columns")
    print(f"\n  Data: {len(df)} days, {df.shape[1]} columns")

    # ── Optional: Tune ────────────────────────────────────────────
    if args.tune:
        best = tune_hantacast(df, config, output_dir=output_root)
        # Apply best config
        config["training"]["lookback"] = best["lookback"]
        if "mixlinear" not in config.get("model", {}):
            config["model"]["mixlinear"] = {}
        config["model"]["mixlinear"]["dropout"] = best["mixlinear_dropout"]
        config["model"]["seird"] = config.get("model", {}).get("seird", {})
        config["model"]["seird"]["mixlinear_signal_scale"] = best["seird_mixlinear_scale"]
        config["training"]["weight_decay"] = best["mixlinear_wd"]
        config["training"]["learning_rate"] = best["mixlinear_lr"]
        print(f"\n  Using best config for training...")

    # ── Train ─────────────────────────────────────────────────────
    print(f"\n  ═══ Training HantaCast ═══")
    model, train_log = train_hantacast(
        train_df=df,
        config=config,
        output_dir=output_root,
        smoke_test=args.smoke_test,
    )

    # ── Print training summary ────────────────────────────────────
    print(f"\n  Training complete — {train_log.get('epochs_run', '?')} epochs")
    print(f"  Residual scale: {train_log['seird_residual_scale']:.4f}")
    print(f"  SEIRD params: beta0={train_log['seird_best_params'].get('beta0', '?')}, "
          f"sigma={train_log['seird_best_params'].get('sigma', '?')}, "
          f"gamma={train_log['seird_best_params'].get('gamma', '?')}")

    # ── Evaluate ──────────────────────────────────────────────────
    if args.skip_eval:
        print("  Evaluation skipped (--skip-eval).")
        return

    print(f"\n  ═══ Evaluating all models (rolling-origin) ═══")

    eval_model = HantaCast(
        lookback=int(config.get("training", {}).get("lookback", 3)),
        horizon=int(config.get("training", {}).get("horizon", 1)),
        random_state=seed,
    )
    eval_model.load_checkpoint(output_root / "checkpoints" / "hantacast_best.pt")
    eval_model.mixlinear.train_df = df.copy()
    eval_model.train_df = df.copy()

    comparison_df = evaluate_all_models(eval_model, df, config, output_dir=output_root)

    if comparison_df.empty:
        print("  ⚠ Dataset too small for rolling-origin evaluation.")
        return

    _print_comparison_table(comparison_df)

    print(f"  Checkpoint:     {output_root / 'checkpoints' / 'hantacast_best.pt'}")
    print(f"  Comparison:     {output_root / 'metrics' / 'model_comparison.csv'}")
    print(f"  All predictions: {output_root / 'predictions' / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()
