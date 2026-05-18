from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


DEFAULT_FEATURE_ORDER = [
    "new_cases",
    "cumulative_cases",
    "intervention_index",
    "mobility_index",
    "flight_volume",
    "behavior_response_index",
    "day_index",
]


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in DEFAULT_FEATURE_ORDER if col in df.columns]


def make_supervised_windows(
    df: pd.DataFrame,
    lookback: int,
    horizon: int,
    target_col: str = "new_cases",
    feature_columns: List[str] | None = None,
) -> Dict[str, object]:
    if feature_columns is None:
        feature_columns = select_feature_columns(df)
    work = df.copy()
    for col in feature_columns + [target_col]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=feature_columns + [target_col]).reset_index(drop=True)

    X, y, target_dates = [], [], []
    if len(work) < lookback + horizon:
        return {
            "X": np.empty((0, lookback, len(feature_columns)), dtype=float),
            "y": np.empty((0, horizon), dtype=float),
            "target_dates": [],
            "feature_columns": feature_columns,
        }

    values = work[feature_columns].to_numpy(dtype=float)
    target = work[target_col].to_numpy(dtype=float)
    dates = pd.to_datetime(work["date"])
    for start in range(0, len(work) - lookback - horizon + 1):
        X.append(values[start : start + lookback])
        y.append(target[start + lookback : start + lookback + horizon])
        target_dates.append(dates.iloc[start + lookback : start + lookback + horizon].dt.strftime("%Y-%m-%d").tolist())
    return {
        "X": np.asarray(X, dtype=float),
        "y": np.asarray(y, dtype=float),
        "target_dates": target_dates,
        "feature_columns": feature_columns,
    }


def load_standardized_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in DEFAULT_FEATURE_ORDER:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def split_train_val_test(
    df: pd.DataFrame,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    test_size = max(1, int(n * test_ratio))
    val_size = max(0, int(n * val_ratio))
    train_size = n - val_size - test_size
    if train_size < 1:
        train_size = max(1, n - 1)
        val_size = max(0, n - train_size)
    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:train_size + val_size].copy() if val_size > 0 else train_df.copy()
    test_df = df.iloc[train_size + val_size:].copy() if train_size + val_size < n else val_df.copy()
    return train_df, val_df, test_df
