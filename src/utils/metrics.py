"""Metrics for forecast evaluation."""

from __future__ import annotations

from typing import Dict

import numpy as np


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-8, np.nan, np.abs(y_true))
    if np.isnan(denom).all():
        return 0.0
    value = np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0
    return float(0.0 if np.isnan(value) else value)


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom < 1e-8, np.nan, denom)
    if np.isnan(denom).all():
        return 0.0
    value = np.nanmean(2.0 * np.abs(y_true - y_pred) / denom) * 100.0
    return float(0.0 if np.isnan(value) else value)


def mase(y_true, y_pred, insample) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    insample = np.asarray(insample, dtype=float)
    if insample.size < 2:
        return float("nan")
    naive_scale = np.mean(np.abs(np.diff(insample)))
    if naive_scale < 1e-8:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / naive_scale)


def peak_timing_error(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return float("nan")
    return float(abs(int(np.argmax(y_true)) - int(np.argmax(y_pred))))


def peak_magnitude_error(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return float("nan")
    return float(abs(float(np.max(y_true)) - float(np.max(y_pred))))


def final_cumulative_error(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return float("nan")
    return float(abs(float(np.sum(y_true)) - float(np.sum(y_pred))))


def correlation(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def poisson_deviance(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 1e-8)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(
            y_true == 0,
            y_pred,
            y_true * np.log(np.maximum(y_true, 1e-8) / y_pred) - (y_true - y_pred),
        )
    return float(2.0 * np.nansum(term))


def compute_all_metrics(y_true, y_pred, insample=None) -> Dict[str, float]:
    metrics = {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "SMAPE": smape(y_true, y_pred),
        "correlation": correlation(y_true, y_pred),
        "peak_timing_error": peak_timing_error(y_true, y_pred),
        "peak_magnitude_error": peak_magnitude_error(y_true, y_pred),
        "final_cumulative_error": final_cumulative_error(y_true, y_pred),
        "poisson_deviance": poisson_deviance(y_true, y_pred),
    }
    metrics["MASE"] = float("nan") if insample is None else mase(y_true, y_pred, insample)
    return metrics
