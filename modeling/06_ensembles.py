"""Modelling step 6: ensembles. Can combining models beat structural v3?

Three standard ways, all scored on the same future-month folds:
  1. averaging: geometric mean of v3 and another model type's predictions (no extra fitting)
  2. stacking by residual learning: a flexible model learns only what v3 leaves over
     (the 'hybrid' idea from modelling 02, now with LightGBM / XGBoost / HGB / random forest, each tuned)
  3. a learned blend: non-negative weights on several models' log predictions, fitted leave-one-fold-out
     (weights for fold k are learned on the other folds only)
Clear-gain bar to replace v3: ≥ 0.02 MAPE points and ≥ $1 RMSE.
"""
from __future__ import annotations

import sys
import time
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from scipy.optimize import nnls  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor  # noqa: E402

from freight.data import city_coordinates, load_train, load_validation  # noqa: E402
from freight.evaluation import FoldData, summarise  # noqa: E402
from freight.reporting import Report, set_style  # noqa: E402
from freight.zoo import BASE_FEATURES, LEVEL_FEATURES, STRUCT_FEATURE, evaluate_estimator  # noqa: E402

COLS = ["MAPE (%)", "RMSE ($)", "MAE ($)", "MAPE seen cities (%)", "MAPE unseen cities (%)", "MAPE month 2 (%)", "bias (%)", "worst-fold MAPE (%)"]
RES_COLS = BASE_FEATURES + LEVEL_FEATURES + STRUCT_FEATURE
REF = "Structural v3 (reference)"


def grid(space: dict, n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out, seen = [], set()
    while len(out) < n:
        p = {k: v[rng.integers(len(v))] for k, v in space.items()}
        key = tuple(sorted((k, str(v)) for k, v in p.items()))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


RESIDUAL_LEARNERS = {
    "v3 + LightGBM on residuals": (lambda p: lgb.LGBMRegressor(verbose=-1, random_state=0, n_jobs=-1, subsample_freq=1, **p),
                                   {"learning_rate": [0.02, 0.05], "n_estimators": [200, 500, 1000], "num_leaves": [7, 15, 31],
                                    "min_child_samples": [50, 100, 300], "reg_lambda": [1.0, 10.0], "colsample_bytree": [0.8, 1.0], "subsample": [0.8, 1.0]}, 8),
    "v3 + XGBoost on residuals": (lambda p: xgb.XGBRegressor(tree_method="hist", n_jobs=-1, random_state=0, **p),
                                  {"learning_rate": [0.02, 0.05], "n_estimators": [200, 500, 1000], "max_depth": [3, 4, 6],
                                   "min_child_weight": [20, 50, 100], "reg_lambda": [1.0, 10.0], "subsample": [0.8, 1.0], "colsample_bytree": [0.8, 1.0]}, 8),
    "v3 + scikit-learn HGB on residuals": (lambda p: HistGradientBoostingRegressor(early_stopping=False, random_state=0, **p),
                                           {"learning_rate": [0.02, 0.05], "max_iter": [200, 400], "max_leaf_nodes": [7, 15, 31],
                                            "min_samples_leaf": [100, 200, 400], "l2_regularization": [1.0, 10.0]}, 6),
    "v3 + random forest on residuals": (lambda p: RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0, **p),
                                        {"max_features": [0.3, 0.5], "min_samples_leaf": [20, 50, 100]}, 3),
}


def gmean(preds: list[list[np.ndarray]]) -> list[np.ndarray]:
    return [np.exp(np.mean([np.log(p[k]) for p in preds], axis=0)) for k in range(len(preds[0]))]


def main() -> None:
    set_style()
    t0 = time.time()
    train, val = load_train(), load_validation()
    fd = FoldData(train, val)
    zoo = pd.read_pickle(ROOT / "outputs" / "zoo_preds.pkl")
    mats = pd.read_pickle(ROOT / "outputs" / "fold_matrices.pkl")
    preds = dict(zoo["preds"])
    rep = Report("06_ensembles", "Modelling 06 · Ensembles: can combining models beat structural v3?",
                 "Same folds, same scoring. Base models are the best tuned setting of each type from Modelling 05.")
    rows = [fd.score_predictions(preds[REF], REF)]

    # 1. averaging ------------------------------------------------------------------------------------------
    rep.h2("1. Averaging (geometric mean of predictions)")
    others = [n for n in zoo["board"].index if n != REF]
    for n in others:
        rows.append(fd.score_predictions(gmean([preds[REF], preds[n]]), f"average: v3 + {n}"))
    top_flex = [n for n in zoo["board"].index if n not in (REF,) and "regression" not in n][:3]
    rows.append(fd.score_predictions(gmean([preds[REF]] + [preds[n] for n in top_flex]), f"average: v3 + top 3 flexible ({', '.join(top_flex)})"))
    avg = summarise(pd.concat(rows)).sort_values("MAPE (%)")
    rep.table(avg[COLS], digits=3)
    errs = pd.DataFrame({n: np.concatenate([np.log(p / f["test"]["posted_rate"].values)[f["genuine"]] for p, f in zip(preds[n], fd.folds)]) for n in [REF] + others})
    rep.p("Why averaging rarely helps here: correlation of the models' log errors with v3's errors (1 = they make the same mistakes):")
    rep.table(errs.corr()[[REF]].drop(REF).sort_values(REF, ascending=False).rename(columns={REF: "error correlation with v3"}), digits=3)

    # 2. residual stacking ---------------------------------------------------------------------------------------
    rep.h2("2. Stacking by residual learning (flexible model on top of v3)")
    rep.p("Each learner sees all 20 features plus v3's own prediction, and learns v3's leftover error. v3's predictions for training rows are "
          "5-fold out-of-fold, so the learner sees honest leftovers, not ones v3 has already memorised.")
    res_rows, res_tables = [], {}
    for name, (factory, space, n) in RESIDUAL_LEARNERS.items():
        best = (np.inf, None, None, None)
        tab = []
        for p in grid(space, n, seed=len(name)):
            pf, pr = evaluate_estimator(fd, mats, lambda p=p: factory(p), name, RES_COLS, residual=True)
            s = summarise(pf).iloc[0]
            tab.append({**p, "MAPE (%)": s["MAPE (%)"], "RMSE ($)": s["RMSE ($)"]})
            if s["MAPE (%)"] < best[0]:
                best = (s["MAPE (%)"], p, pf, pr)
            print(f"  {name} {p}: {s['MAPE (%)']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        res_tables[name] = pd.DataFrame(tab).sort_values("MAPE (%)")
        res_rows.append(best[2])
        preds[name] = best[3]
    res = summarise(pd.concat([rows[0]] + res_rows)).sort_values("MAPE (%)")
    rep.table(res[COLS], digits=3)
    for name, tab in res_tables.items():
        rep.h3(f"Tuning: {name}")
        rep.table(tab, index=False, digits=3)

    # 3. learned blend ---------------------------------------------------------------------------------------------
    rep.h2("3. Learned blend (non-negative weights, fitted leave-one-fold-out)")
    candidates = [REF] + others + list(RESIDUAL_LEARNERS)
    k = len(fd.folds)
    logp = {n: [np.log(preds[n][i]) for i in range(k)] for n in candidates}
    logy = [np.log(f["test"]["posted_rate"].values) for f in fd.folds]
    blend_preds, weights = [], []
    for i in range(k):
        tr_idx = [j for j in range(k) if j != i]
        A = np.vstack([np.column_stack([logp[n][j][fd.folds[j]["genuine"]] for n in candidates]) for j in tr_idx])
        b = np.concatenate([logy[j][fd.folds[j]["genuine"]] for j in tr_idx])
        w, _ = nnls(A, b)
        w = w / w.sum()
        weights.append(w)
        blend_preds.append(np.exp(np.column_stack([logp[n][i] for n in candidates]) @ w))
    wt = pd.DataFrame(weights, columns=candidates, index=[f["name"] for f in fd.folds]).T
    wt = wt[(wt > 0.005).any(axis=1)]
    bl = fd.score_predictions(blend_preds, "learned blend (leave-one-fold-out)")
    rep.p("Weights chosen when each fold is held out (models with weight < 0.5% in every fold omitted):")
    rep.table(wt, digits=3)
    # pairwise residual-stack averages as a robustness check
    pair_rows = [fd.score_predictions(gmean([preds[a], preds[b]]), f"average: {a} + {b}") for a, b in combinations(list(RESIDUAL_LEARNERS), 2)]

    # 4. leaderboard --------------------------------------------------------------------------------------------------
    rep.h2("4. Ensemble leaderboard")
    allres = summarise(pd.concat([rows[0], *rows[1:], *res_rows, bl, *pair_rows])).drop_duplicates().sort_values("MAPE (%)")
    rep.table(allres.head(12)[COLS], digits=3)
    ref = allres.loc[REF]
    beats = [n for n in allres.index if n != REF and allres.loc[n, "MAPE (%)"] <= ref["MAPE (%)"] - 0.02 and allres.loc[n, "RMSE ($)"] <= ref["RMSE ($)"] - 1.0]
    rep.p("Ensembles that beat structural v3 by the clear-gain bar: " + (", ".join(f"**{b}**" for b in beats) if beats else "**none**") + ".")
    best_name = allres.index[0]
    pf_best = pd.concat([rows[0], *rows[1:], *res_rows, bl, *pair_rows]).query("model in [@REF, @best_name]")
    rep.p("Per fold, best ensemble vs v3 (MAPE %):")
    rep.table(pf_best.pivot(index="fold", columns="model", values="MAPE (%)"), digits=3)
    rep.save()
    pd.to_pickle({"preds": preds, "leaderboard": allres, "residual_best": {n: res_tables[n].iloc[0].to_dict() for n in res_tables}}, ROOT / "outputs" / "ensemble_preds.pkl")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
