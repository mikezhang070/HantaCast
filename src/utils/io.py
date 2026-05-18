from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


EXPECTED_FILE_ALIASES: Dict[str, list[str]] = {
    "case_timeseries": ["mv_hondius_case_timeseries.csv", "Case_Timeseries.csv"],
    "model_ready": ["mv_hondius_model_ready.csv", "Model_Ready.csv"],
    "repatriation_flights": ["mv_hondius_repatriation_flights.csv", "Repatriation_Flights.csv"],
    "research_long_format": ["mv_hondius_research_data_long_format.csv"],
    "timeline": ["Cruise_Timeline.csv"],
    "summary": ["Quick_Summary.csv"],
    "readme_table": ["README.csv"],
}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def warn(message: str) -> None:
    warnings.warn(message, stacklevel=2)


def write_json(data: dict, path: str | Path) -> Path:
    path = ensure_parent(path)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def find_first_existing(search_roots, candidates) -> Optional[Path]:
    checked = []
    for root in search_roots:
        for name in candidates:
            direct = Path(root) / name
            checked.append(direct)
            if direct.exists():
                return direct
            matches = list(Path(root).rglob(name))
            if matches:
                return matches[0]
    return None


def locate_input_files(base_dir: str | Path = ".") -> Dict[str, Optional[Path]]:
    base_dir = Path(base_dir).resolve()
    search_roots = [base_dir / "data" / "raw", base_dir / "data", base_dir]
    located = {}
    for key, aliases in EXPECTED_FILE_ALIASES.items():
        located[key] = find_first_existing(search_roots, aliases)
    return located


def read_csv_safe(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def parse_date_like(value) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    text = str(value).strip()
    if not text:
        return pd.NaT
    match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if match:
        return pd.to_datetime(match.group(0), errors="coerce")
    return pd.to_datetime(text, errors="coerce")


def extract_numeric_value(value, strategy: str = "mean") -> float:
    if pd.isna(value):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if text in {"unknown", "nan", "none", ""}:
        return float("nan")
    numbers = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        return float("nan")
    if "-" in text and len(numbers) >= 2:
        return sum(numbers[:2]) / 2.0
    if "+" in text or strategy == "sum":
        return float(sum(numbers))
    return float(numbers[0] if strategy == "first" else sum(numbers) / len(numbers))


def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    path = ensure_parent(path)
    df.to_csv(path, index=False)
    return path


def setup_logger(log_path: str | Path) -> None:
    import logging

    log_path = Path(log_path)
    ensure_parent(log_path)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
