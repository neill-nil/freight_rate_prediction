"""Models, from simplest to most flexible. All share `fit(df)` / `predict(df)` (price in $).

`df` must already be cleaned with `data.clean_inputs`; training frames must have corrupted labels removed.

1. NaiveRatePerMile  : one typical price per mile × distance (the simplest sensible baseline)
2. StructuralModel   : the multiplicative structure found in the EDA, fitted by least squares on log price
                       log(price) = f(distance) + weight + market + equipment + city + quarter-end ramp + drift
3. GBMModel          : gradient-boosted trees on the same cleaned inputs (flexible, no assumed shape)
4. HybridModel       : structural model + trees fitted to what the structural model leaves over
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import days_to_quarter_end, knn_city_effect, months_since_start, quarter_ramp

EQUIP_CODE = {"Dry Van": 0, "Reefer": 1, "Flatbed": 2}


class NaiveRatePerMile:
    name = "naive rate per mile"

    def fit(self, df: pd.DataFrame) -> "NaiveRatePerMile":
        self.rpm_ = float(np.median(df["posted_rate"] / df["distance"]))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.rpm_ * df["distance"].values


class StructuralModel:
    """Least squares on log price with the shapes found in eda/09 and eda/10.

    Parameters
    ----------
    distance_degree : polynomial degree in log distance (1 = power law; eda/09 found 3 best)
    weight_bounds   : weight only matters inside this band (eda/09: 20k–40k lb); None = log weight
    use_weight      : include weight at all
    use_market      : include market_index (linear)
    use_cities      : one shared effect per city (same whether pickup or delivery)
    ramp_days       : quarter-end ramp length in days (0 = no ramp; eda/10: 28)
    trend           : 'none' | 'continue' (extrapolate the drift) | 'freeze' (hold it at its last training value)
                      | 'damped' (continue at half speed: the average of 'continue' and 'freeze')
    unseen          : how to price a city absent from training: 'mean' (average city) | 'knn' (nearest neighbours)
    """

    def __init__(self, distance_degree: int = 3, weight_bounds: tuple[float, float] | None = (20_000, 40_000),
                 use_weight: bool = True, use_market: bool = True, use_cities: bool = True, ramp_days: float = 28,
                 trend: str = "continue", unseen: str = "knn", knn_k: int = 5, coords: pd.DataFrame | None = None,
                 ramp_by_equipment: bool = False, drift_by_equipment: bool = False, lane_shrink: float | None = None) -> None:
        self.distance_degree, self.weight_bounds, self.use_market = distance_degree, weight_bounds, use_market
        self.use_weight = use_weight
        self.ramp_by_equipment, self.drift_by_equipment = ramp_by_equipment, drift_by_equipment
        self.lane_shrink = lane_shrink
        self.use_cities, self.ramp_days, self.trend = use_cities, ramp_days, trend
        self.unseen, self.knn_k, self.coords = unseen, knn_k, coords

    # design matrix ------------------------------------------------------------------------------------
    def _design(self, df: pd.DataFrame, t: np.ndarray) -> np.ndarray:
        z = np.log(df["distance"].values) - 7.0  # centred for numerical stability
        cols = [z**p for p in range(1, self.distance_degree + 1)]
        if self.use_weight and self.weight_bounds is None:
            cols.append(np.log(df["weight_abs"].values))
        elif self.use_weight:
            lo, hi = self.weight_bounds
            cols.append(np.clip(df["weight_abs"].values, lo, hi) / 1e4)
        if self.use_market:
            cols.append(df["mi"].values)
        reefer = (df["equipment"] == "Reefer").values.astype(float)
        flatbed = (df["equipment"] == "Flatbed").values.astype(float)
        cols += [reefer, flatbed]
        if self.ramp_days:
            r = quarter_ramp(df["date"], self.ramp_days)
            cols.append(r)
            if self.ramp_by_equipment:  # quarter-end rush is larger for Reefer and Flatbed (found in modelling 02)
                cols += [r * reefer, r * flatbed]
        if self.trend != "none":
            cols.append(t)
            if self.drift_by_equipment:
                cols += [t * reefer, t * flatbed]
        X = np.column_stack(cols)
        if self.use_cities:
            X = np.column_stack([X, self._city_dummies(df)])
        return X

    def _city_dummies(self, df: pd.DataFrame) -> np.ndarray:
        idx = {c: i for i, c in enumerate(self.cities_)}
        M = np.zeros((len(df), len(self.cities_)))
        rows = np.arange(len(df))
        for role in ("pickup", "delivery"):
            j = df[role].map(idx)
            ok = j.notna().values
            M[rows[ok], j[ok].astype(int).values] += 1
        return M[:, 1:]  # first city is the reference level (effect 0)

    # fit / predict --------------------------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "StructuralModel":
        self.cities_ = sorted(set(df["pickup"]) | set(df["delivery"]))
        t = months_since_start(df["date"])
        self.t_max_ = float(t.max())
        X = self._design(df, t)
        A = np.column_stack([np.ones(len(df)), X])
        self.coef_, *_ = np.linalg.lstsq(A, np.log(df["posted_rate"].values), rcond=None)
        n_city = len(self.cities_) - 1 if self.use_cities else 0
        city_coef = self.coef_[len(self.coef_) - n_city:] if n_city else np.array([])
        self.city_effects_ = pd.Series(np.r_[0.0, city_coef], index=self.cities_) if self.use_cities else pd.Series(dtype=float)
        if self.lane_shrink is not None:
            # each directed lane's own level beyond its two cities: mean residual shrunk toward 0
            # (sum / (n + λ)), so a lane with few loads gets only a small adjustment; unseen lanes get 0
            resid = np.log(df["posted_rate"].values) - A @ self.coef_
            g = pd.DataFrame({"lane": self._lane_key(df), "r": resid}).groupby("lane")["r"].agg(["sum", "count"])
            self.lane_effects_ = g["sum"] / (g["count"] + self.lane_shrink)
        return self

    @staticmethod
    def _lane_key(df: pd.DataFrame) -> np.ndarray:
        return (df["pickup"] + "→" + df["delivery"]).values

    def _time_index(self, df: pd.DataFrame) -> np.ndarray:
        t = months_since_start(df["date"])
        if self.trend == "freeze":
            return np.minimum(t, self.t_max_)
        if self.trend == "damped":
            return np.where(t > self.t_max_, self.t_max_ + 0.5 * (t - self.t_max_), t)
        return t

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        X = self._design(df, self._time_index(df))
        out = np.column_stack([np.ones(len(df)), X]) @ self.coef_
        if self.use_cities:  # unseen cities got a zero dummy row; give them an estimated effect instead
            known = set(self.cities_)
            est: dict[str, float] = {}
            for role in ("pickup", "delivery"):
                for city in set(df[role]) - known:
                    if city not in est:
                        if self.unseen == "knn" and self.coords is not None and city in self.coords.index:
                            est[city] = knn_city_effect(city, self.city_effects_, self.coords, self.knn_k)
                        else:
                            est[city] = float(self.city_effects_.mean())
                out = out + df[role].map(est).fillna(0.0).values
        if self.lane_shrink is not None:
            out = out + pd.Series(self._lane_key(df)).map(self.lane_effects_).fillna(0.0).values
        return out

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(df))


GBM_FEATURES = ["log_distance", "equipment_code", "weight_abs", "mi", "pickup_lat", "pickup_lon",
                "delivery_lat", "delivery_lon", "days_to_quarter_end"]


def gbm_frame(df: pd.DataFrame, use_trend: bool = False) -> pd.DataFrame:
    X = pd.DataFrame({
        "log_distance": np.log(df["distance"].values),
        "equipment_code": df["equipment"].map(EQUIP_CODE).values,
        "weight_abs": df["weight_abs"].values,
        "mi": df["mi"].values,
        "pickup_lat": df["pickup_lat"].values, "pickup_lon": df["pickup_lon"].values,
        "delivery_lat": df["delivery_lat"].values, "delivery_lon": df["delivery_lon"].values,
        "days_to_quarter_end": days_to_quarter_end(df["date"]),
    })
    if use_trend:
        X["months"] = months_since_start(df["date"])
    return X


def make_hgb(params: dict | None = None, features: list[str] | None = None) -> HistGradientBoostingRegressor:
    p = {"max_iter": 600, "learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 50, "l2_regularization": 0.0}
    p.update(params or {})
    cat = ["equipment_code"] if features is None or "equipment_code" in features else None
    return HistGradientBoostingRegressor(categorical_features=cat, early_stopping=False, random_state=0, **p)


class GBMModel:
    """Gradient-boosted trees on log price (target='log_rate') or log price per mile (target='log_rpm').

    log_rpm removes most of the steep distance trend first, so the piecewise-constant trees only have to
    model a gentle slope (a known weakness of trees on smooth trends).
    """

    def __init__(self, params: dict | None = None, target: str = "log_rpm", use_trend: bool = False) -> None:
        self.params, self.target, self.use_trend = params, target, use_trend

    def _offset(self, df: pd.DataFrame) -> np.ndarray:
        return np.log(df["distance"].values) if self.target == "log_rpm" else 0.0

    def fit(self, df: pd.DataFrame) -> "GBMModel":
        y = np.log(df["posted_rate"].values) - self._offset(df)
        self.model_ = make_hgb(self.params).fit(gbm_frame(df, self.use_trend), y)
        return self

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(gbm_frame(df, self.use_trend)) + self._offset(df)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(df))


class HybridModel:
    """Structural model first; trees then learn whatever pattern is left in its residuals.

    `features` restricts what the residual trees may look at (used to diagnose what they learn).
    """

    def __init__(self, structural: StructuralModel, params: dict | None = None, features: list[str] | None = None) -> None:
        self.structural, self.params, self.features = structural, params, features

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        X = gbm_frame(df)
        return X if self.features is None else X[self.features]

    def fit(self, df: pd.DataFrame) -> "HybridModel":
        self.structural.fit(df)
        resid = np.log(df["posted_rate"].values) - self.structural.predict_log(df)
        self.model_ = make_hgb(self.params, self.features).fit(self._X(df), resid)
        return self

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        return self.structural.predict_log(df) + self.model_.predict(self._X(df))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(df))
