#!/usr/bin/env python
"""One-click run: train → evaluate → forecast HantaCast.

Usage:
    python scripts/run_all.py
    python scripts/run_all.py --smoke-test
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.io import ensure_dir, setup_logger

log = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent


def run_step(step_name: str, cmd: list[str]) -> int:
    print(f"\n{'=' * 60}")
    print(f"STEP: {step_name}")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd, cwd=SCRIPTS_DIR.parent)
    if result.returncode != 0:
        print(f"FAILED: {step_name} (exit code {result.returncode})")
    else:
        print(f"COMPLETED: {step_name}")
    return result.returncode


def check_output_files(output_dir: str) -> list[str]:
    output_dir = Path(output_dir)
    expected = [
        output_dir / "checkpoints" / "hantacast_best.pt",
        output_dir / "metrics" / "metrics.json",
        output_dir / "metrics" / "evaluation_metrics.json",
        output_dir / "predictions" / "forecast_150day.csv",
        output_dir / "predictions" / "test_predictions.csv",
        output_dir / "logs" / "train.log",
    ]
    missing = [str(f) for f in expected if not f.exists()]
    return missing


def main():
    parser = argparse.ArgumentParser(description="Run all HantaCast steps")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test mode (1 epoch)")
    parser.add_argument("--horizon", type=int, default=150, help="Forecast horizon in days")
    args = parser.parse_args()

    output_dir = "outputs"
    ensure_dir(output_dir)
    setup_logger(Path(output_dir) / "logs" / "run_all.log")

    smoke_flag = ["--smoke-test"] if args.smoke_test else []
    config_arg = "--config configs/default.yaml"

    # Step 1: Train
    rc = run_step("1/3: Training", [
        sys.executable, str(SCRIPTS_DIR / "train_hantacast.py"), config_arg, *smoke_flag
    ])
    if rc != 0 and not args.smoke_test:
        print("Training failed. Stopping.")
        return

    # Step 2: Evaluate
    rc = run_step("2/3: Evaluation", [
        sys.executable, str(SCRIPTS_DIR / "evaluate_hantacast.py"), config_arg,
        "--checkpoint", f"{output_dir}/checkpoints/hantacast_best.pt"
    ])

    # Step 3: Forecast
    rc = run_step("3/3: Forecast", [
        sys.executable, str(SCRIPTS_DIR / "forecast_hantacast.py"), config_arg,
        "--checkpoint", f"{output_dir}/checkpoints/hantacast_best.pt",
        "--horizon", str(args.horizon),
    ])

    # Check output files
    print(f"\n{'=' * 60}")
    print("OUTPUT FILE CHECK")
    print(f"{'=' * 60}")
    missing = check_output_files(output_dir)
    if missing:
        print("MISSING FILES:")
        for f in missing:
            print(f"  - {f}")
    else:
        print("All expected output files present.")
        print(f"  checkpoints/hantacast_best.pt")
        print(f"  metrics/metrics.json")
        print(f"  metrics/evaluation_metrics.json")
        print(f"  predictions/forecast_{args.horizon}day.csv")
        print(f"  predictions/test_predictions.csv")
        print(f"  logs/train.log")

    print(f"\nFull run complete. Output directory: {Path(output_dir).resolve()}")


if __name__ == "__main__":
    main()
