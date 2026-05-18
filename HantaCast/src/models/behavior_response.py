from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.io import parse_date_like, save_dataframe


COMPLIANCE_KEYWORDS = {
    "quarantine": 0.9, "isolation": 0.9, "monitor": 0.7, "screen": 0.65,
    "testing": 0.65, "disembark": 0.55, "repatriation": 0.5, "evacuation": 0.55,
}
UNCERTAINTY_KEYWORDS = {
    "notification": 0.55, "update": 0.4, "reported": 0.45, "suspected": 0.7,
    "probable": 0.6, "inconclusive": 0.75, "died": 0.8, "death": 0.8, "critically ill": 0.8,
}
MOBILITY_RESPONSE_KEYWORDS = {
    "flight": 0.8, "travel": 0.65, "arrival": 0.55, "departure": 0.5,
    "repatriation": 0.8, "evacuation": 0.8, "ship leaves": 0.55,
}


def _keyword_score(text: str, keyword_map: dict[str, float]) -> float:
    text = str(text).lower()
    matched = [weight for keyword, weight in keyword_map.items() if keyword in text]
    if not matched:
        return 0.0
    return float(min(1.0, max(matched) + 0.1 * max(len(matched) - 1, 0)))


def encode_policy_text_to_behavior_score(text: str) -> dict[str, float]:
    text = str(text or "")
    return {
        "compliance_pressure_score": _keyword_score(text, COMPLIANCE_KEYWORDS),
        "uncertainty_score": _keyword_score(text, UNCERTAINTY_KEYWORDS),
        "mobility_response_score": _keyword_score(text, MOBILITY_RESPONSE_KEYWORDS),
    }


def build_behavior_response_index(
    base_dates: pd.Series,
    intervention_df: pd.DataFrame,
    mobility_df: pd.DataFrame,
    timeline_df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    output_path: str | Path = "outputs/processed/behavior_response_index.csv",
) -> pd.DataFrame:
    config = config or {}
    behavior_cfg = config.get("behavior_response", {})
    weights = behavior_cfg.get("weights", {
        "intervention_index": 0.45, "mobility_index": 0.20, "event_intensity_score": 0.35,
    })

    behavior_df = pd.DataFrame({"date": pd.to_datetime(base_dates)})
    behavior_df["compliance_pressure_score"] = 0.0
    behavior_df["uncertainty_score"] = 0.0
    behavior_df["mobility_response_score"] = 0.0
    behavior_df["event_intensity_score"] = 0.0
    behavior_df["source"] = "imputed_default"
    behavior_df["is_imputed"] = True

    if not timeline_df.empty:
        work = timeline_df.copy()
        date_col = "date" if "date" in work.columns else work.columns[0]
        work["event_date"] = work[date_col].map(parse_date_like)
        text_cols = [col for col in ["event_type", "event_summary", "model_tag"] if col in work.columns]
        work["event_text"] = work[text_cols].astype(str).agg(" ".join, axis=1).str.strip()
        work = work.dropna(subset=["event_date"])
        score_rows = []
        for _, row in work.iterrows():
            scores = encode_policy_text_to_behavior_score(row["event_text"])
            event_intensity = float(min(1.0,
                0.5 * scores["compliance_pressure_score"]
                + 0.3 * scores["uncertainty_score"]
                + 0.2 * scores["mobility_response_score"]))
            score_rows.append({"date": pd.to_datetime(row["event_date"]), **scores, "event_intensity_score": event_intensity})
        if score_rows:
            score_df = pd.DataFrame(score_rows).groupby("date", as_index=False).max()
            behavior_df = behavior_df.merge(score_df, on="date", how="left", suffixes=("", "_timeline"))
            for col in ["compliance_pressure_score", "uncertainty_score", "mobility_response_score", "event_intensity_score"]:
                behavior_df[col] = behavior_df[f"{col}_timeline"].fillna(0.0)
                behavior_df = behavior_df.drop(columns=[f"{col}_timeline"])
            behavior_df["source"] = "timeline_keyword_coding"
            behavior_df["is_imputed"] = False

    intervention = intervention_df[["date", "intervention_index"]].copy()
    intervention["date"] = pd.to_datetime(intervention["date"])
    mobility = mobility_df[["date", "mobility_index"]].copy()
    mobility["date"] = pd.to_datetime(mobility["date"])
    behavior_df = behavior_df.merge(intervention, on="date", how="left")
    behavior_df = behavior_df.merge(mobility, on="date", how="left")
    behavior_df["intervention_index"] = pd.to_numeric(behavior_df["intervention_index"], errors="coerce").fillna(0.2)
    behavior_df["mobility_index"] = pd.to_numeric(behavior_df["mobility_index"], errors="coerce").fillna(0.0)

    behavior_df["behavior_response_index"] = (
        float(weights.get("intervention_index", 0.45)) * behavior_df["intervention_index"]
        + float(weights.get("mobility_index", 0.20)) * behavior_df["mobility_index"]
        + float(weights.get("event_intensity_score", 0.35)) * behavior_df["event_intensity_score"]
    ).clip(0.0, 1.0)

    if behavior_df["behavior_response_index"].max() <= 0:
        behavior_df["behavior_response_index"] = 0.25
        behavior_df["source"] = "imputed_default"
        behavior_df["is_imputed"] = True

    behavior_df["date"] = pd.to_datetime(behavior_df["date"]).dt.strftime("%Y-%m-%d")
    output_cols = ["date", "behavior_response_index", "compliance_pressure_score",
                   "uncertainty_score", "mobility_response_score", "source", "is_imputed"]
    save_dataframe(behavior_df[output_cols], output_path)
    return behavior_df[output_cols]
