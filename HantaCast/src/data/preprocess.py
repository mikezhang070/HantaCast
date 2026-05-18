from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.utils.io import (
    extract_numeric_value,
    locate_input_files,
    parse_date_like,
    read_csv_safe,
    save_dataframe,
    warn,
)
from src.models.behavior_response import build_behavior_response_index


STANDARD_COLUMNS = [
    "date",
    "day_index",
    "new_cases",
    "cumulative_cases",
    "active_cases",
    "intervention_index",
    "mobility_index",
    "flight_volume",
    "behavior_response_index",
    "location",
    "source_note",
    "is_imputed",
]


def _choose_column(columns, candidates):
    lowered = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _parse_case_table(case_df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str], dict]:
    if case_df.empty:
        raise ValueError("Case timeseries table is required to build the standardized series.")

    date_col = _choose_column(case_df.columns, ["date", "report_date", "day"])
    if date_col is None:
        raise ValueError("No date-like column detected in case timeseries table.")

    cumulative_col = _choose_column(
        case_df.columns,
        ["cumulative_cases", "total_cases", "cases", "confirmed_cases", "confirmed"],
    )
    new_col = _choose_column(case_df.columns, ["new_cases", "daily_cases", "daily_new_cases"])
    deaths_col = _choose_column(case_df.columns, ["deaths", "death_count"])
    note_col = _choose_column(case_df.columns, ["notes"])
    source_col = _choose_column(case_df.columns, ["reporting_source", "source", "source_id"])

    work = case_df.copy()
    work["date"] = work[date_col].map(parse_date_like)
    work = work.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    if cumulative_col is not None:
        work["cumulative_cases"] = pd.to_numeric(work[cumulative_col], errors="coerce")
    if new_col is not None:
        work["new_cases"] = pd.to_numeric(work[new_col], errors="coerce")

    construction = {
        "new_cases": f"original column: {new_col}" if new_col is not None else "derived_from_cumulative_difference",
        "cumulative_cases": f"original column: {cumulative_col}" if cumulative_col else "derived_from_new_cases_cumsum",
    }

    if "new_cases" not in work.columns and "cumulative_cases" in work.columns:
        work["new_cases"] = work["cumulative_cases"].diff().fillna(work["cumulative_cases"]).clip(lower=0)
    if "cumulative_cases" not in work.columns and "new_cases" in work.columns:
        work["cumulative_cases"] = work["new_cases"].fillna(0).cumsum()

    full_dates = pd.date_range(work["date"].min(), work["date"].max(), freq="D")
    standardized = pd.DataFrame({"date": full_dates})
    standardized = standardized.merge(
        work[["date", "new_cases", "cumulative_cases"] + ([deaths_col] if deaths_col else [])],
        on="date",
        how="left",
    )
    standardized["cumulative_cases"] = standardized["cumulative_cases"].ffill()
    standardized["new_cases"] = standardized["cumulative_cases"].diff().fillna(standardized["cumulative_cases"]).clip(lower=0)

    if deaths_col:
        standardized["deaths"] = pd.to_numeric(standardized[deaths_col], errors="coerce").ffill().fillna(0)
        standardized["active_cases"] = (standardized["cumulative_cases"] - standardized["deaths"]).clip(lower=0)
    else:
        active_window = min(14, max(3, len(standardized) // 2 or 3))
        standardized["active_cases"] = standardized["new_cases"].rolling(active_window, min_periods=1).sum()

    source_map = work.set_index("date")
    standardized["source_note"] = standardized["date"].map(
        lambda d: " | ".join(
            str(source_map.loc[d, col])
            for col in [source_col, note_col]
            if col is not None and d in source_map.index and pd.notna(source_map.loc[d, col])
        )
        if d in source_map.index
        else "forward-filled between report dates"
    )
    standardized["is_imputed"] = ~standardized["date"].isin(work["date"])

    target_cols = [col for col in ["new_cases", "cumulative_cases", "active_cases"] if col in standardized.columns]
    return standardized, target_cols, construction


def _classify_intervention(text: str) -> dict:
    lower = str(text or "").lower()
    intervention_type = "unknown_or_unclassified"
    intervention_index = 0.2
    mobility_effect = 0.0
    testing_quarantine_effect = 0.0
    environmental_effect = 0.0
    behavioral_effect = 0.1

    if any(key in lower for key in ["travel", "arrival", "departure", "flight", "repatriation", "disembark", "evacuation"]):
        intervention_type = "travel_or_mobility_restriction"
        intervention_index = 0.45
        mobility_effect = 0.6
        behavioral_effect = 0.35
    if any(key in lower for key in ["test", "screen", "confirm", "lab", "notification"]):
        intervention_type = "testing_or_screening"
        intervention_index = max(intervention_index, 0.35)
        testing_quarantine_effect = max(testing_quarantine_effect, 0.55)
    if any(key in lower for key in ["quarantine", "isolation", "monitor", "high risk"]):
        intervention_type = "quarantine_or_isolation"
        intervention_index = max(intervention_index, 0.8)
        testing_quarantine_effect = max(testing_quarantine_effect, 0.9)
    if any(key in lower for key in ["disinfection", "environment", "sanit"]):
        intervention_type = "environmental_control"
        intervention_index = max(intervention_index, 0.35)
    if any(key in lower for key in ["report", "who", "ecdc", "update", "monitoring"]):
        intervention_type = "reporting_or_monitoring" if intervention_type == "unknown_or_unclassified" else intervention_type
        intervention_index = max(intervention_index, 0.25)

    return {
        "intervention_type": intervention_type,
        "intervention_index": float(min(intervention_index, 1.0)),
        "mobility_effect": float(min(mobility_effect, 1.0)),
        "testing_quarantine_effect": float(min(testing_quarantine_effect, 1.0)),
        "environmental_effect": float(min(environmental_effect, 1.0)),
        "behavioral_effect": float(min(behavioral_effect, 1.0)),
    }


def _build_intervention_series(base_dates: pd.Series, timeline_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    timeline = pd.DataFrame({"date": base_dates.copy()})
    timeline["intervention_index"] = 0.0

    if timeline_df.empty:
        timeline["intervention_index"] = 0.2
        timeline["intervention_imputed"] = True
        dictionary = pd.DataFrame([
            {"raw_event_text": "", "date": pd.to_datetime(base_dates).min().strftime("%Y-%m-%d"),
             "intervention_type": "unknown_or_unclassified", "intervention_index": 0.2,
             "mobility_effect": 0.0, "testing_quarantine_effect": 0.0,
             "environmental_effect": 0.0, "behavioral_effect": 0.1,
             "source": "imputed_default", "is_imputed": True}
        ])
        return timeline, dictionary

    work = timeline_df.copy()
    date_col = _choose_column(work.columns, ["date"])
    work["event_date"] = work[date_col].map(parse_date_like)
    event_type = work["event_type"].astype(str) if "event_type" in work.columns else pd.Series("", index=work.index, dtype="object")
    event_summary = work["event_summary"].astype(str) if "event_summary" in work.columns else pd.Series("", index=work.index, dtype="object")
    work["raw_event_text"] = (event_type + " " + event_summary).str.strip()
    work["event_text"] = work["raw_event_text"].str.lower()
    work = work.dropna(subset=["event_date"])

    event_scores = []
    dictionary_rows = []
    for _, row in work.iterrows():
        text = row["event_text"]
        event_meta = _classify_intervention(text)
        event_scores.append((row["event_date"], float(event_meta["intervention_index"])))
        dictionary_rows.append({
            "raw_event_text": str(row["raw_event_text"]),
            "date": pd.to_datetime(row["event_date"]).strftime("%Y-%m-%d"),
            **event_meta,
            "source": "original timeline",
            "is_imputed": False,
        })

    event_df = pd.DataFrame(event_scores, columns=["date", "score"])
    if event_df.empty:
        timeline["intervention_index"] = 0.2
        timeline["intervention_imputed"] = True
        return timeline, pd.DataFrame(dictionary_rows)

    merged = timeline.merge(event_df.groupby("date")["score"].max().reset_index(), on="date", how="left")
    merged["intervention_index"] = merged["score"].fillna(0.0).cummax()
    if merged["intervention_index"].max() <= 0:
        merged["intervention_index"] = 0.2
        merged["intervention_imputed"] = True
    else:
        merged["intervention_imputed"] = False
    return merged.drop(columns=["score"]), pd.DataFrame(dictionary_rows)


def _build_mobility_series(base_dates: pd.Series, flights_df: pd.DataFrame) -> pd.DataFrame:
    mobility = pd.DataFrame({"date": base_dates.copy()})
    mobility["flight_volume"] = 0.0
    mobility["mobility_index"] = 0.0
    mobility["mobility_imputed"] = True
    if flights_df.empty:
        return mobility

    work = flights_df.copy()
    date_col = _choose_column(work.columns, ["date_or_window", "date"])
    count_col = _choose_column(work.columns, ["reported_count", "count", "volume"])
    work["date"] = work[date_col].map(parse_date_like)
    work["flight_volume"] = work[count_col].map(lambda x: extract_numeric_value(x, strategy="mean")) if count_col else np.nan
    work = work.dropna(subset=["date"])
    grouped = work.groupby("date")["flight_volume"].sum(min_count=1).reset_index()
    mobility = mobility.merge(grouped, on="date", how="left", suffixes=("", "_observed"))
    mobility["flight_volume"] = mobility["flight_volume_observed"].fillna(0.0)
    max_volume = mobility["flight_volume"].max()
    mobility["mobility_index"] = mobility["flight_volume"] / max_volume if max_volume and max_volume > 0 else 0.0
    mobility["mobility_imputed"] = mobility["flight_volume_observed"].isna()
    return mobility.drop(columns=["flight_volume_observed"])


def _extract_population_total(model_ready_df: pd.DataFrame) -> float:
    if model_ready_df.empty or "variable" not in model_ready_df.columns:
        return 147.0
    mask = model_ready_df["variable"].astype(str).str.contains("N_ship_initial", case=False, na=False)
    if not mask.any():
        return 147.0
    return extract_numeric_value(model_ready_df.loc[mask, "value"].iloc[0], strategy="first") or 147.0


def build_standardized_timeseries(
    base_dir: str | Path = ".",
    output_path: str | Path = "outputs/processed/case_timeseries_standardized.csv",
    config: dict | None = None,
) -> Tuple[pd.DataFrame, dict]:
    file_map = locate_input_files(base_dir)
    case_df = read_csv_safe(file_map.get("case_timeseries"))
    model_ready_df = read_csv_safe(file_map.get("model_ready"))
    flights_df = read_csv_safe(file_map.get("repatriation_flights"))
    timeline_df = read_csv_safe(file_map.get("timeline"))

    standardized, target_cols, construction = _parse_case_table(case_df)
    intervention_df, _intervention_dictionary = _build_intervention_series(standardized["date"], timeline_df)
    mobility_df = _build_mobility_series(standardized["date"], flights_df)
    behavior_df = build_behavior_response_index(
        standardized["date"],
        intervention_df[["date", "intervention_index"]],
        mobility_df[["date", "mobility_index"]],
        timeline_df,
        config=config,
        output_path="outputs/processed/behavior_response_index.csv",
    )
    behavior_df["date"] = pd.to_datetime(behavior_df["date"])

    standardized = standardized.merge(intervention_df[["date", "intervention_index", "intervention_imputed"]], on="date", how="left")
    standardized = standardized.merge(mobility_df[["date", "mobility_index", "flight_volume", "mobility_imputed"]], on="date", how="left")
    standardized = standardized.merge(
        behavior_df[["date", "behavior_response_index", "is_imputed"]].rename(columns={"is_imputed": "behavior_imputed"}),
        on="date", how="left")
    standardized["intervention_index"] = standardized["intervention_index"].fillna(0.2).clip(0, 1)
    standardized["mobility_index"] = standardized["mobility_index"].fillna(0.0).clip(0, 1)
    standardized["flight_volume"] = standardized["flight_volume"].fillna(0.0)
    standardized["behavior_response_index"] = standardized["behavior_response_index"].fillna(0.25).clip(0, 1)
    standardized["day_index"] = np.arange(len(standardized), dtype=int)
    standardized["location"] = "MV Hondius aggregate"
    standardized["is_imputed"] = (
        standardized["is_imputed"].fillna(False)
        | standardized["intervention_imputed"].fillna(False)
        | standardized["mobility_imputed"].fillna(True)
        | standardized["behavior_imputed"].fillna(True)
    )
    standardized["population_total"] = _extract_population_total(model_ready_df)
    standardized["source_note"] = standardized["source_note"].fillna("derived from available aggregate data")

    for column in STANDARD_COLUMNS:
        if column not in standardized.columns:
            standardized[column] = np.nan

    standardized = standardized[
        STANDARD_COLUMNS + [col for col in standardized.columns if col not in STANDARD_COLUMNS]
    ].copy()
    standardized["date"] = pd.to_datetime(standardized["date"]).dt.strftime("%Y-%m-%d")
    save_dataframe(standardized, output_path)

    metadata = {
        "standardized_path": str(Path(output_path).resolve()),
        "available_target_columns": target_cols,
        "input_files": {key: str(path) if path else None for key, path in file_map.items()},
        "n_rows": int(len(standardized)),
        "n_days": int(len(standardized)),
        "date_range": f"{standardized['date'].iloc[0]} to {standardized['date'].iloc[-1]}",
    }
    return standardized, metadata


def build_future_covariates(standardized_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = standardized_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    horizon = int(horizon)
    tail = df.tail(max(horizon, 3)).copy().reset_index(drop=True)
    repeats = []
    for step in range(horizon):
        base_row = tail.iloc[min(step, len(tail) - 1)].copy()
        base_row["date"] = df["date"].iloc[-1] + pd.Timedelta(days=step + 1)
        base_row["day_index"] = int(df["day_index"].iloc[-1]) + step + 1
        repeats.append(base_row)
    future_df = pd.DataFrame(repeats)
    future_df["date"] = future_df["date"].dt.strftime("%Y-%m-%d")
    covariate_cols = [col for col in [
        "date", "day_index", "intervention_index", "mobility_index",
        "flight_volume", "behavior_response_index", "location"
    ] if col in future_df.columns]
    return future_df[covariate_cols]
