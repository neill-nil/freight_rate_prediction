"""Feature helpers: calendar/time terms and neighbour-based city effects."""
from __future__ import annotations

import numpy as np
import pandas as pd

T0 = pd.Timestamp("2025-01-01")


def months_since_start(dates: pd.Series) -> np.ndarray:
    """Time index in 30-day 'months' since 2025-01-01 (for the slow upward drift)."""
    return ((pd.to_datetime(dates) - T0).dt.days / 30.0).values


def days_to_quarter_end(dates: pd.Series) -> np.ndarray:
    d = pd.to_datetime(dates)
    return ((d + pd.offsets.QuarterEnd(0)) - d).dt.days.values


def quarter_ramp(dates: pd.Series, ramp_days: float) -> np.ndarray:
    """0 until `ramp_days` before quarter end, then rising linearly to 1 on the last day (eda/10)."""
    return np.clip(1 - days_to_quarter_end(dates) / ramp_days, 0, None)


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(h))


def knn_city_effect(city: str, effects: pd.Series, coords: pd.DataFrame, k: int = 5) -> float:
    """Distance-weighted mean effect of the k nearest cities that have an estimated effect (eda/11)."""
    known = effects.index
    d = haversine_miles(coords.loc[city, "lat"], coords.loc[city, "lon"], coords.loc[known, "lat"].values, coords.loc[known, "lon"].values)
    order = np.argsort(d)[:k]
    w = 1 / np.maximum(d[order], 1.0)
    return float(np.sum(w * effects.values[order]) / w.sum())
