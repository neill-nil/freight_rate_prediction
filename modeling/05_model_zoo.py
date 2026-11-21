"""Modelling step 5: standard model types on the same features, each tuned, scored on the same future-month folds.

Fairness: every model gets the same leak-free feature matrix (src/freight/zoo.py): the EDA-engineered terms,
raw coordinates, days to quarter end, and city / lane price levels learned without seeing each row's own
price. Linear models get the formula-style subset; flexible models get all 20 features. Target: log price per mile.
Each model type gets its own small tuning search on the same 4 folds; the best setting per type goes on
the leaderboard. Structural v3 (modelling 04) is the reference to beat.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lightgbm as lgb  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.linear_model import Lasso, LinearRegression, Ridge  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from freight.data import city_coordinates, load_train, load_validation  # noqa: E402
from freight.evaluation import FoldData, summarise  # noqa: E402
from freight.models import StructuralModel  # noqa: E402
from freight.reporting import BLUE, INK_2, ORANGE, Report, set_style  # noqa: E402
from freight.zoo import BASE_FEATURES, LEVEL_FEATURES, build_fold_matrices, evaluate_estimator  # noqa: E402

LINEAR_COLS = ["log_distance", "z2", "z3", "weight_clip", "mi", "reefer", "flatbed", "ramp", "ramp_reefer", "ramp_flatbed", "t"] + LEVEL_FEATURES
ALL_COLS = BASE_FEATURES + LEVEL_FEATURES
COLS = ["MAPE (%)", "RMSE ($)", "MAE ($)", "MAPE seen cities (%)", "MAPE unseen cities (%)", "MAPE month 2 (%)", "bias (%)", "worst-fold MAPE (%)"]


def grid(space: dict, n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out, seen = [], set()
    total = int(np.prod([len(v) for v in space.values()]))
    while len(out) < min(n, total):
        p = {k: v[rng.integers(len(v))] for k, v in space.items()}
        key = tuple(sorted((k, str(v)) for k, v in p.items()))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def scaled(est):
    return lambda: make_pipeline(StandardScaler(), est())


MODEL_TYPES = {
    # name: (feature set, factory(params) -> estimator, search space, n settings)
    "Linear regression (least squares)": (LINEAR_COLS, lambda p: scaled(lambda: LinearRegression())(), {"none": [0]}, 1),
    "Ridge regression": (LINEAR_COLS, lambda p: make_pipeline(StandardScaler(), Ridge(alpha=p["alpha"])), {"alpha": [0.01, 0.1, 1, 10, 100]}, 5),
    "Lasso regression": (LINEAR_COLS, lambda p: make_pipeline(StandardScaler(), Lasso(alpha=p["alpha"], max_iter=20_000)), {"alpha": [1e-5, 1e-4, 1e-3, 1e-2]}, 4),
    "k-nearest neighbours": (ALL_COLS, lambda p: make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=p["k"], weights="distance", n_jobs=-1)), {"k": [5, 15, 30, 60]}, 4),
    "Random forest": (ALL_COLS, lambda p: RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0, **p),
                      {"max_features": [0.3, 0.5, 0.8], "min_samples_leaf": [1, 5, 20]}, 6),
    "Extra trees": (ALL_COLS, lambda p: ExtraTreesRegressor(n_estimators=200, n_jobs=-1, random_state=0, **p),
                    {"max_features": [0.5, 0.8, 1.0], "min_samples_leaf": [1, 5, 20]}, 4),
    "Gradient boosting (scikit-learn HGB)": (ALL_COLS, lambda p: HistGradientBoostingRegressor(early_stopping=False, random_state=0, **p),
                                             {"learning_rate": [0.03, 0.06, 0.1], "max_iter": [500, 1000, 1500], "max_leaf_nodes": [15, 31, 63],
                                              "min_samples_leaf": [20, 50, 100], "l2_regularization": [0.0, 1.0, 10.0]}, 8),
    "LightGBM": (ALL_COLS, lambda p: lgb.LGBMRegressor(verbose=-1, random_state=0, n_jobs=-1, subsample_freq=1, **p),
                 {"learning_rate": [0.02, 0.05, 0.1], "n_estimators": [500, 1000, 2000], "num_leaves": [15, 31, 63],
                  "min_child_samples": [10, 30, 100], "reg_lambda": [0.0, 1.0, 10.0], "colsample_bytree": [0.6, 0.8, 1.0], "subsample": [0.7, 1.0]}, 12),
    "XGBoost": (ALL_COLS, lambda p: xgb.XGBRegressor(tree_method="hist", n_jobs=-1, random_state=0, **p),
                {"learning_rate": [0.02, 0.05, 0.1], "n_estimators": [500, 1000, 2000], "max_depth": [4, 6, 8],
                 "min_child_weight": [1, 5, 20], "reg_lambda": [1.0, 10.0], "subsample": [0.7, 1.0], "colsample_bytree": [0.6, 0.8, 1.0]}, 12),
}


def main() -> None:
    set_style()
    t0 = time.time()
    train, val = load_train(), load_validation()
    coords = city_coordinates(train, val)
    fd = FoldData(train, val)
    v3 = pd.read_pickle(ROOT / "outputs" / "feature_choice.pkl")["structural_v3"]
    mats = build_fold_matrices(fd, v3, coords)
    print(f"features built ({time.time() - t0:.0f}s)", flush=True)
    rep = Report("05_model_zoo", "Modelling 05 · Standard model types, each tuned, on the same features",
                 "Same 4 future-month folds with hidden cities as every other step. Every model gets the same leak-free features "
                 "(EDA terms, coordinates, days to quarter end, and city / lane price levels learned without seeing each row's own price). "
                 "Linear models use the formula-style subset; flexible models use all 20. Target: log price per mile.")
    rep.h2("1. Features given to the models")
    rep.bullets([f"**Linear models ({len(LINEAR_COLS)})**: {', '.join(LINEAR_COLS)}",
                 f"**Flexible models ({len(ALL_COLS)})**: {', '.join(ALL_COLS)}",
                 "City and lane levels for training rows come from 5-fold estimates inside the training data, so no row sees its own price. "
                 "Unseen cities get the 5-nearest-neighbour level; unseen lanes get 0."])

    pf_v3, preds_v3 = fd.evaluate(lambda: StructuralModel(coords=coords, **v3), "Structural v3 (reference)")
    best_rows, all_preds, tuning_tables = [pf_v3], {"Structural v3 (reference)": preds_v3}, {}
    for name, (cols, factory, space, n) in MODEL_TYPES.items():
        rows = []
        best = (np.inf, None, None, None)
        for p in grid(space, n, seed=len(name)):
            params = {} if "none" in p else p
            pf, preds = evaluate_estimator(fd, mats, lambda params=params: factory(params), name, cols)
            s = summarise(pf).iloc[0]
            rows.append({**{k: v for k, v in params.items()}, "MAPE (%)": s["MAPE (%)"], "RMSE ($)": s["RMSE ($)"], "worst-fold MAPE (%)": s["worst-fold MAPE (%)"]})
            if s["MAPE (%)"] < best[0]:
                best = (s["MAPE (%)"], params, pf, preds)
            print(f"  {name} {params}: MAPE {s['MAPE (%)']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        tuning_tables[name] = (pd.DataFrame(rows).sort_values("MAPE (%)"), best[1])
        best_rows.append(best[2])
        all_preds[name] = best[3]

    rep.h2("2. Leaderboard (best setting of each model type)")
    board = summarise(pd.concat(best_rows)).sort_values("MAPE (%)")
    rep.table(board[COLS], digits=3)
    ref = board.loc["Structural v3 (reference)"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    y = np.arange(len(board))[::-1]
    colors = [ORANGE if n == "Structural v3 (reference)" else BLUE for n in board.index]
    ax.barh(y, board["MAPE (%)"], color=colors, height=0.6)
    for yi, v in zip(y, board["MAPE (%)"]):
        ax.text(v + 0.02, yi, f"{v:.2f}%", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, board.index)
    ax.set(title="Future-month MAPE by model type (orange = structural v3 reference)", xlabel="MAPE (%)")
    ax.grid(axis="y", visible=False)
    rep.figure(fig, "leaderboard", "Every model type at its best tuned setting, same features, same folds.")
    per_fold = pd.concat(best_rows).pivot(index="fold", columns="model", values="MAPE (%)")[board.index]
    rep.p("MAPE per fold:")
    rep.table(per_fold.T, digits=3)
    beats = [n for n in board.index if n != "Structural v3 (reference)" and board.loc[n, "MAPE (%)"] <= ref["MAPE (%)"] - 0.02 and board.loc[n, "RMSE ($)"] <= ref["RMSE ($)"] - 1.0]
    rep.p("Model types that beat structural v3 by the clear-gain bar (≥ 0.02 MAPE pts and ≥ $1 RMSE): " + (", ".join(beats) if beats else "**none**") + ".")

    rep.h2("3. Tuning details per model type")
    rep.p("Each search ran on the same 4 folds; the table shows every setting tried, best first. Picking the best of several settings on the "
          "same folds flatters the flexible models slightly; the same procedure was applied to all.")
    for name, (tab, best_p) in tuning_tables.items():
        rep.h3(name)
        rep.table(tab, index=False, digits=3)
    rep.save()
    pd.to_pickle({"preds": all_preds, "board": board, "best_params": {k: v[1] for k, v in tuning_tables.items()}}, ROOT / "outputs" / "zoo_preds.pkl")
    pd.to_pickle(mats, ROOT / "outputs" / "fold_matrices.pkl")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
