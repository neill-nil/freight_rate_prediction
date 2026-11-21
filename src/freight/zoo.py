"""A shared, leak-free feature matrix so different model types can be compared fairly.

Every model gets the same information:
  * the engineered terms from the EDA (curved distance, clipped weight, market, equipment, quarter-end ramp
    and its per-equipment versions, damped time trend) plus raw coordinates and days to quarter end
  * the city and lane price levels the structural model learns, and the structural model's own prediction
These last ones are *target encodings* (built from prices), so a row must never see its own price:
  * training rows: computed 5-fold inside the training data (levels from the other 4/5)
  * test rows: from the structural model fitted on all training data
Target for every model: log(price per mile); prediction = exp(output) × distance.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .evaluation import FoldData, score_fold
from .features import days_to_quarter_end, knn_city_effect, months_since_start, quarter_ramp
from .models import StructuralModel

BASE_FEATURES = ["log_distance", "z2", "z3", "weight_abs", "weight_clip", "mi", "reefer", "flatbed", "days_to_qe",
                 "ramp", "ramp_reefer", "ramp_flatbed", "t", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
LEVEL_FEATURES = ["city_pickup", "city_delivery", "lane"]
STRUCT_FEATURE = ["struct_pred"]  # structural model's log price-per-mile prediction (for stacking / residual models)


class FeatureBuilder:
    def __init__(self, structural_params: dict, coords: pd.DataFrame, inner_folds: int = 5, seed: int = 0) -> None:
        self.params, self.coords, self.inner_folds, self.seed = structural_params, coords, inner_folds, seed

    def _base(self, df: pd.DataFrame) -> pd.DataFrame:
        z = np.log(df["distance"].values) - 7.0
        rf = (df["equipment"] == "Reefer").values.astype(float)
        fb = (df["equipment"] == "Flatbed").values.astype(float)
        r = quarter_ramp(df["date"], self.params.get("ramp_days", 28))
        t = months_since_start(df["date"])
        t = np.where(t > self.t_max_, self.t_max_ + 0.5 * (t - self.t_max_), t)  # damped beyond training (modelling 01)
        lo, hi = self.params.get("weight_bounds", (20_000, 40_000))
        return pd.DataFrame({
            "log_distance": z + 7.0, "z2": z**2, "z3": z**3,
            "weight_abs": df["weight_abs"].values, "weight_clip": np.clip(df["weight_abs"].values, lo, hi),
            "mi": df["mi"].values, "reefer": rf, "flatbed": fb, "days_to_qe": days_to_quarter_end(df["date"]),
            "ramp": r, "ramp_reefer": r * rf, "ramp_flatbed": r * fb, "t": t,
            "pickup_lat": df["pickup_lat"].values, "pickup_lon": df["pickup_lon"].values,
            "delivery_lat": df["delivery_lat"].values, "delivery_lon": df["delivery_lon"].values,
        }, index=df.index)

    def _levels(self, m: StructuralModel, df: pd.DataFrame) -> pd.DataFrame:
        eff = m.city_effects_ - m.city_effects_.mean()
        cache: dict[str, float] = {}

        def city_level(city: str) -> float:
            if city not in cache:
                cache[city] = float(eff[city]) if city in eff.index else knn_city_effect(city, eff, self.coords, m.knn_k)
            return cache[city]

        lane = pd.Series(m._lane_key(df)).map(m.lane_effects_).fillna(0.0).values if m.lane_shrink is not None else 0.0
        return pd.DataFrame({
            "city_pickup": df["pickup"].map(city_level).values, "city_delivery": df["delivery"].map(city_level).values,
            "lane": lane, "struct_pred": m.predict_log(df) - np.log(df["distance"].values),
        }, index=df.index)

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        self.t_max_ = float(months_since_start(train["date"]).max())
        levels = pd.DataFrame(index=train.index, columns=LEVEL_FEATURES + STRUCT_FEATURE, dtype=float)
        for a, b in KFold(self.inner_folds, shuffle=True, random_state=self.seed).split(train):
            m = StructuralModel(coords=self.coords, **self.params).fit(train.iloc[a])
            levels.iloc[b] = self._levels(m, train.iloc[b]).values
        self.full_ = StructuralModel(coords=self.coords, **self.params).fit(train)
        return pd.concat([self._base(train), levels], axis=1)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([self._base(df), self._levels(self.full_, df)], axis=1)


def build_fold_matrices(fd: FoldData, structural_params: dict, coords: pd.DataFrame) -> list[dict]:
    """Feature matrices for every fold, computed once and reused by all model types."""
    mats = []
    for f in fd.folds:
        fb = FeatureBuilder(structural_params, coords)
        Xtr = fb.fit_transform(f["train"])
        Xte = fb.transform(f["test"])
        mats.append({"Xtr": Xtr, "ytr": np.log(f["train"]["posted_rate"].values / f["train"]["distance"].values),
                     "Xte": Xte, "logd_te": np.log(f["test"]["distance"].values)})
    return mats


def evaluate_estimator(fd: FoldData, mats: list[dict], make_est: Callable[[], object], label: str, cols: list[str],
                       residual: bool = False) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Fit an sklearn-style regressor on log price-per-mile per fold and score it like every other model.

    residual=True: the estimator learns only what the structural prediction leaves over (stacking on top of it).
    """
    rows, preds = [], []
    for f, m in zip(fd.folds, mats):
        y = m["ytr"] - m["Xtr"]["struct_pred"].values if residual else m["ytr"]
        est = make_est().fit(m["Xtr"][cols], y)
        out = est.predict(m["Xte"][cols]) + (m["Xte"]["struct_pred"].values if residual else 0.0)
        p = np.exp(out + m["logd_te"])
        preds.append(p)
        rows.append(score_fold(f, p, label))
    return pd.DataFrame(rows), preds
