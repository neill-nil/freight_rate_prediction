"""Loading and prediction-time cleaning.

Every rule here comes from the EDA (reports/eda/03_data_quality.md, 09_structure.md):
  * weight: negative values are sign flips → abs(); missing → median for the equipment type
  * market_index: a per-date value plus small noise → missing values get the same-date mean
  * corrupted labels (prices ×/÷ 2–5.5) are removed from training only, by a two-pass rule
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
EQUIPMENT = ["Dry Van", "Reefer", "Flatbed"]

LANE_OUTLIER_LO, LANE_OUTLIER_HI = 0.5, 2.0  # price per mile vs the lane's typical value
MODEL_OUTLIER_LOG = 0.4  # |log residual| vs the structural model (≈ ×1.5 or ÷1.5)


def load_train() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "train_test.csv", parse_dates=["date"])


def load_validation() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "validation.csv", parse_dates=["date"])


def load_december() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "december_chart_inputs.csv", parse_dates=["date"])


def daily_market_index(*frames: pd.DataFrame) -> pd.Series:
    """Mean market_index per date over the given frames (inputs only, no labels)."""
    allf = pd.concat([f[["date", "market_index"]] for f in frames])
    return allf.groupby("date")["market_index"].mean()


def weight_fill_values(train: pd.DataFrame) -> dict[str, float]:
    return train.assign(w=train["weight"].abs()).groupby("equipment")["w"].median().to_dict()


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """One (lat, lon) per city, from any frame that has coordinate columns."""
    parts = []
    for f in frames:
        for role in ("pickup", "delivery"):
            if f"{role}_lat" in f:
                parts.append(f[[role, f"{role}_lat", f"{role}_lon"]].set_axis(["city", "lat", "lon"], axis=1))
    return pd.concat(parts).drop_duplicates("city").set_index("city").sort_index()


def clean_inputs(df: pd.DataFrame, daily_mi: pd.Series, weight_fill: dict[str, float], coords: pd.DataFrame | None = None) -> pd.DataFrame:
    """Apply the cleaning rules to any frame (train, test, validation, December)."""
    out = df.copy()
    if coords is not None and "pickup_lat" not in out:  # December file has no coordinates
        for role in ("pickup", "delivery"):
            out[f"{role}_lat"] = out[role].map(coords["lat"])
            out[f"{role}_lon"] = out[role].map(coords["lon"])
    if "market_index" not in out:  # December file has no market_index
        out["market_index"] = np.nan
    out["weight_missing"] = out["weight"].isna()
    out["weight_abs"] = out["weight"].abs().fillna(out["equipment"].map(weight_fill))
    out["mi_missing"] = out["market_index"].isna()
    out["mi"] = out["market_index"].fillna(out["date"].map(daily_mi))
    return out


def lane_relative_rate(df: pd.DataFrame) -> pd.Series:
    """Price per mile ÷ typical price per mile for the same lane and equipment (medians, so robust)."""
    rpm = df["posted_rate"] / df["distance"]
    lane_eq = rpm.groupby([df["pickup"], df["delivery"], df["equipment"]]).median().unstack("equipment")
    premium = {"Dry Van": 1.0}
    for e in EQUIPMENT[1:]:
        if e in lane_eq and "Dry Van" in lane_eq:
            premium[e] = float((lane_eq[e] / lane_eq["Dry Van"]).dropna().median())
    prem = df["equipment"].map(premium)
    lane_base = (rpm / prem).groupby([df["pickup"], df["delivery"]]).transform("median")
    return rpm / (lane_base * prem)


def flag_corrupted(df: pd.DataFrame, lane_bounds: tuple[float, float] = (LANE_OUTLIER_LO, LANE_OUTLIER_HI),
                   model_log: float | None = MODEL_OUTLIER_LOG) -> pd.Series:
    """Two-pass corrupted-label rule on a cleaned, labelled frame.

    Pass 1: price per mile far from the lane's typical value (catches almost all).
    Pass 2: far from the structural model's prediction (catches corrupted prices on thin lanes where
            the corrupted value is itself the lane median). model_log=None skips pass 2.
    """
    from .models import StructuralModel  # local import avoids a cycle

    rel = lane_relative_rate(df)
    pass1 = (rel < lane_bounds[0]) | (rel > lane_bounds[1])
    if model_log is None:
        return pass1
    m = StructuralModel(trend="continue", unseen="mean").fit(df[~pass1])
    resid = np.log(df["posted_rate"].values) - m.predict_log(df)
    pass2 = pd.Series(np.abs(resid) > model_log, index=df.index)
    return pass1 | pass2
