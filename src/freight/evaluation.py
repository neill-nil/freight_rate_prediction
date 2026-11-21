"""Time-aware evaluation that mimics the real task.

The real task: learn from Jan–Oct, predict the next two months (Nov–Dec), where 12% of loads involve a
city never seen in training. Each fold copies that:
  * train on all months up to a cut-off, test on the following TWO months (forward-chaining)
  * hide 4 random cities from the training part (≈ 12% of test loads then involve an unseen city)
  * clean inputs and remove corrupted labels using the training part only
  * score on test rows with genuine labels (corrupted labels can't be predicted by anyone), and also
    report all rows for transparency
Metrics: MAPE and RMSE (the two agreed measures) plus MAE and median APE.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .data import clean_inputs, daily_market_index, flag_corrupted, weight_fill_values

FOLDS = [  # (name, last training day, last test day)
    ("train Jan–May → test Jun–Jul", "2025-05-31", "2025-07-31"),
    ("train Jan–Jun → test Jul–Aug", "2025-06-30", "2025-08-31"),
    ("train Jan–Jul → test Aug–Sep", "2025-07-31", "2025-09-30"),
    ("train Jan–Aug → test Sep–Oct", "2025-08-31", "2025-10-31"),
]
N_HIDDEN_CITIES = 4


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ape = np.abs(p - y) / y
    return {"MAPE (%)": 100 * ape.mean(), "RMSE ($)": float(np.sqrt(np.mean((p - y) ** 2))),
            "MAE ($)": float(np.mean(np.abs(p - y))), "median APE (%)": 100 * float(np.median(ape))}


class FoldData:
    """Pre-computes each fold's cleaned train/test frames once, so many models can be compared cheaply."""

    def __init__(self, train_raw: pd.DataFrame, val_raw: pd.DataFrame, folds=FOLDS, n_hidden: int = N_HIDDEN_CITIES, seed: int = 0,
                 flag_fn: Callable[[pd.DataFrame], pd.Series] = flag_corrupted) -> None:
        """flag_fn: rule that marks corrupted labels in each fold's TRAINING part (scored test rows never change)."""
        wf_all = weight_fill_values(train_raw)
        full = clean_inputs(train_raw, daily_market_index(train_raw, val_raw), wf_all)
        self.genuine = ~flag_corrupted(full)  # used ONLY to choose which test rows to score
        self.folds = []
        for i, (name, train_end, test_end) in enumerate(folds):
            tr = train_raw[train_raw.date <= train_end]
            te = train_raw[(train_raw.date > train_end) & (train_raw.date <= test_end)]
            rng = np.random.default_rng(seed + i)
            cities = sorted(set(tr.pickup) | set(tr.delivery))
            hidden = set(rng.choice(cities, n_hidden, replace=False)) if n_hidden else set()
            tr = tr[~(tr.pickup.isin(hidden) | tr.delivery.isin(hidden))]
            dmi = daily_market_index(tr, te)
            wf = weight_fill_values(tr)
            trc, tec = clean_inputs(tr, dmi, wf), clean_inputs(te, dmi, wf)
            trc = trc[~flag_fn(trc)]
            self.folds.append({
                "name": name, "train": trc, "test": tec, "hidden": sorted(hidden),
                "unseen": (tec.pickup.isin(hidden) | tec.delivery.isin(hidden)).values,
                "genuine": self.genuine.reindex(tec.index).values,
                "month2": (tec.date > (pd.Timestamp(train_end) + pd.offsets.MonthEnd(1))).values,
            })

    def evaluate(self, make_model: Callable[[], object], label: str = "") -> tuple[pd.DataFrame, list[np.ndarray]]:
        rows, preds = [], []
        for f in self.folds:
            m = make_model().fit(f["train"])
            p = m.predict(f["test"])
            preds.append(p)
            rows.append(score_fold(f, p, label))
        return pd.DataFrame(rows), preds

    def score_predictions(self, preds: list[np.ndarray], label: str) -> pd.DataFrame:
        """Score ready-made per-fold predictions (used for ensembles)."""
        return pd.DataFrame([score_fold(f, p, label) for f, p in zip(self.folds, preds)])


def score_fold(f: dict, p: np.ndarray, label: str) -> dict:
    """All metrics for one fold's predictions (genuine test rows; seen/unseen cities; month 1/2)."""
    y = f["test"]["posted_rate"].values
    g = f["genuine"]
    row = {"model": label, "fold": f["name"], **metrics(y[g], p[g])}
    row["MAPE seen cities (%)"] = metrics(y[g & ~f["unseen"]], p[g & ~f["unseen"]])["MAPE (%)"]
    row["MAPE unseen cities (%)"] = metrics(y[g & f["unseen"]], p[g & f["unseen"]])["MAPE (%)"]
    row["MAPE month 1 (%)"] = metrics(y[g & ~f["month2"]], p[g & ~f["month2"]])["MAPE (%)"]
    row["MAPE month 2 (%)"] = metrics(y[g & f["month2"]], p[g & f["month2"]])["MAPE (%)"]
    row["bias (%)"] = 100 * float(np.median(p[g] / y[g] - 1))
    row["RMSE all rows ($)"] = metrics(y, p)["RMSE ($)"]
    return row


def summarise(per_fold: pd.DataFrame) -> pd.DataFrame:
    """Average over folds, plus the worst fold's MAPE (stability)."""
    cols = [c for c in per_fold.columns if c not in ("model", "fold")]
    s = per_fold.groupby("model", sort=False)[cols].mean()
    s["worst-fold MAPE (%)"] = per_fold.groupby("model", sort=False)["MAPE (%)"].max()
    return s
