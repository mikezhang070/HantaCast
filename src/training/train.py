"""Training utilities for HantaCast."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.models.hantacast import HantaCast
from src.utils.io import ensure_dir, write_json

log = logging.getLogger(__name__)


def train_hantacast(
    train_df: pd.DataFrame,
    config: dict,
    output_dir: str | Path = "outputs",
    smoke_test: bool = False,
) -> tuple[HantaCast, dict]:
    """Train HantaCast model.

    Parameters
    ----------
    train_df : pd.DataFrame
        Standardized training data.
    config : dict
        Configuration dictionary.
    output_dir : str | Path
        Output directory for checkpoints and logs.
    smoke_test : bool
        If True, run a minimal training (1 epoch) for smoke testing.

    Returns
    -------
    model : HantaCast
        Trained model.
    train_log : dict
        Training log with metrics.
    """
    output_dir = Path(output_dir)
    ensure_dir(output_dir / "checkpoints")
    ensure_dir(output_dir / "logs")

    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    seed = int(config.get("seed", 42))

    epochs = 1 if smoke_test else int(train_cfg.get("epochs", 80))

    model = HantaCast(
        lookback=int(train_cfg.get("lookback", 3)),
        horizon=int(train_cfg.get("horizon", 1)),
        random_state=seed,
        mixlinear_epochs=epochs,
        mixlinear_lr=float(train_cfg.get("learning_rate", 0.01)),
        mixlinear_wd=float(train_cfg.get("weight_decay", 0.001)),
        mixlinear_dropout=float(model_cfg.get("mixlinear", {}).get("dropout", 0.1)),
        mixlinear_mc_samples=int(model_cfg.get("mixlinear", {}).get("mc_samples", 50)),
        seird_population=float(model_cfg.get("seird", {}).get("total_population", 147.0)),
        seird_mixlinear_scale=float(model_cfg.get("seird", {}).get("mixlinear_signal_scale", 0.15)),
        seird_interval_samples=int(model_cfg.get("seird", {}).get("interval_samples", 120)),
        seird_param_grid=model_cfg.get("seird", {}).get("param_grid", None),
    )

    if smoke_test:
        log.info("SMOKE TEST MODE: running 1 epoch only")

    model.fit(train_df)

    # Save checkpoint
    ckpt_path = output_dir / "checkpoints" / "hantacast_best.pt"
    model.save_checkpoint(ckpt_path)

    # Build training log
    train_log = {
        "model": "HantaCast",
        "lookback": model.lookback,
        "horizon": model.horizon,
        "epochs": epochs,
        "epochs_run": model.mixlinear.epochs_run,
        "mixlinear_signal_scale": model.seird.mixlinear_signal_scale,
        "seird_best_params": model.seird.best_params,
        "seird_residual_scale": float(model.seird.residual_scale),
        "checkpoint_path": str(ckpt_path.resolve()),
        "smoke_test": smoke_test,
    }
    metrics_path = output_dir / "metrics" / "metrics.json"
    write_json(train_log, metrics_path)
    log.info(f"Training metrics saved to {metrics_path}")

    return model, train_log
