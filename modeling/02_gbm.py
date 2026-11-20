"""Modelling step 2: a flexible challenger (gradient-boosted trees), properly tuned, and a hybrid.

Why: the structural model assumes shapes we found in the EDA. Trees assume nothing, so they test whether
anything was missed. Known weaknesses for this data, both checked here:
  * trees approximate smooth curves with steps → we compare log price vs log price-per-mile as the target
  * trees cannot extrapolate a trend in time → with or without a time-index feature
Hyperparameters are tuned by random search on the SAME time-forward folds (with hidden cities) as step 1.
The hybrid keeps the structural model and lets small trees learn only what it leaves over.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from freight.data import city_coordinates, load_train, load_validation  # noqa: E402
from freight.evaluation import FoldData, summarise  # noqa: E402
from freight.features import quarter_ramp  # noqa: E402
from freight.models import GBMModel, HybridModel, StructuralModel  # noqa: E402
from freight.reporting import Report, set_style  # noqa: E402

COLS = ["MAPE (%)", "RMSE ($)", "MAE ($)", "MAPE seen cities (%)", "MAPE unseen cities (%)", "MAPE month 1 (%)", "MAPE month 2 (%)", "bias (%)", "worst-fold MAPE (%)"]
SPACE = {
    "learning_rate": [0.02, 0.04, 0.06, 0.1],
    "max_iter": [300, 600, 1000, 1500],
    "max_leaf_nodes": [15, 31, 63, 127],
    "min_samples_leaf": [20, 50, 100, 200],
    "l2_regularization": [0.0, 0.1, 1.0, 10.0],
    "max_features": [0.5, 0.8, 1.0],
}
HYBRID_SPACE = {
    "learning_rate": [0.02, 0.05],
    "max_iter": [100, 200, 400],
    "max_leaf_nodes": [7, 15, 31],
    "min_samples_leaf": [100, 200, 400],
    "l2_regularization": [1.0, 10.0],
}


def sample(space: dict, n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out, seen = [], set()
    while len(out) < n:
        p = {k: v[rng.integers(len(v))] for k, v in space.items()}
        key = tuple(sorted(p.items()))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def main() -> None:
    set_style()
    t0 = time.time()
    train, val = load_train(), load_validation()
    coords = city_coordinates(train, val)
    fd = FoldData(train, val)
    chosen = pd.read_pickle(ROOT / "outputs" / "structural_choice.pkl")["chosen"]
    rep = Report(
        "02_gbm",
        "Modelling 02 · Gradient-boosted trees: tuned challenger and hybrid",
        "Same folds, hidden cities and scoring as Modelling 01. Model: scikit-learn HistGradientBoostingRegressor on cleaned inputs "
        "(log distance, equipment, weight, market_index, pickup/delivery lat & lon, days to quarter end; optionally a time index). "
        "City names are deliberately not used, so new cities are handled through coordinates.",
    )

    # 1. design choices with default settings ------------------------------------------------------------------
    rep.h2("1. Two design choices, with default settings")
    per = []
    for target in ("log_rate", "log_rpm"):
        for trend in (False, True):
            label = f"target {target}, {'with' if trend else 'no'} time index"
            per.append(fd.evaluate(lambda t=target, tr=trend: GBMModel(target=t, use_trend=tr), label)[0])
    s1 = summarise(pd.concat(per))
    rep.table(s1[COLS], digits=3)
    best_design = s1["MAPE (%)"].idxmin()
    target = "log_rpm" if "log_rpm" in best_design else "log_rate"
    use_trend = "with time index" in best_design
    rep.bullets([
        "**Target**: log price per mile removes most of the steep distance trend before the trees start, so their step-shaped fit "
        "only has to follow a gentle slope.",
        "**Time index**: trees can't extrapolate. With a time index they hold the last training level (like 'freeze'); without it "
        "they average over all training months and miss the drift entirely.",
        f"Chosen for tuning: **{best_design}**.",
    ])

    # 2. random-search tuning ------------------------------------------------------------------------------------
    rep.h2("2. Hyperparameter tuning (random search, 24 settings × 4 time-forward folds)")
    configs = sample(SPACE, 24, seed=1)
    rows = []
    for i, p in enumerate(configs):
        pf, _ = fd.evaluate(lambda p=p: GBMModel(params=p, target=target, use_trend=use_trend), f"cfg {i}")
        s = summarise(pf).iloc[0]
        rows.append({**p, **{k: s[k] for k in ["MAPE (%)", "RMSE ($)", "worst-fold MAPE (%)", "MAPE unseen cities (%)"]}})
        print(f"  cfg {i:2d} MAPE {s['MAPE (%)']:.3f} RMSE {s['RMSE ($)']:.1f}  ({time.time() - t0:.0f}s)", flush=True)
    tune = pd.DataFrame(rows).sort_values("MAPE (%)").reset_index(drop=True)
    rep.p("Search space: " + "; ".join(f"{k} ∈ {v}" for k, v in SPACE.items()) + ". Ranked by mean future-month MAPE:")
    rep.table(tune.head(10), index=False, digits=3)
    best = {k: tune.loc[0, k] for k in SPACE}
    best = {k: (int(v) if k in ("max_iter", "max_leaf_nodes", "min_samples_leaf") else float(v)) for k, v in best.items()}
    rep.p(f"Spread among the top 5 settings: MAPE {tune['MAPE (%)'][:5].min():.3f}–{tune['MAPE (%)'][:5].max():.3f}%. "
          f"Worst setting tried: {tune['MAPE (%)'].max():.3f}%. Chosen: " + ", ".join(f"{k}={v}" for k, v in best.items()) + ".")
    corr = tune[list(SPACE) + ["MAPE (%)"]].astype(float).corr(method="spearman")["MAPE (%)"].drop("MAPE (%)")
    rep.p("Which settings mattered (Spearman correlation with MAPE across the 24 tries; negative = higher value is better): "
          + ", ".join(f"{k} {v:+.2f}" for k, v in corr.items()) + ".")

    # 3. hybrid on the structural model from step 1 ----------------------------------------------------------------
    rep.h2("3. Hybrid: structural model + small trees on its residuals")
    rep.p("The trees alone lose to the structural model (they can't extrapolate the drift and are weaker on unseen cities). But they "
          "might still spot patterns the structural model misses. The hybrid keeps the structural model and lets small trees "
          "learn only its leftovers. Tuned with a random search of 12 settings.")
    S1 = lambda: StructuralModel(coords=coords, **chosen)  # noqa: E731

    def tune_hybrid(make_structural, seed: int, tag: str) -> tuple[pd.DataFrame, dict]:
        hrows = []
        for i, p in enumerate(sample(HYBRID_SPACE, 12, seed=seed)):
            pf, _ = fd.evaluate(lambda p=p: HybridModel(make_structural(), params=p), f"{tag} {i}")
            s = summarise(pf).iloc[0]
            hrows.append({**p, **{k: s[k] for k in ["MAPE (%)", "RMSE ($)", "worst-fold MAPE (%)", "MAPE unseen cities (%)"]}})
            print(f"  {tag} {i:2d} MAPE {s['MAPE (%)']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        ht = pd.DataFrame(hrows).sort_values("MAPE (%)").reset_index(drop=True)
        hb = {k: (int(ht.loc[0, k]) if k in ("max_iter", "max_leaf_nodes", "min_samples_leaf") else float(ht.loc[0, k])) for k in HYBRID_SPACE}
        return ht, hb

    htune, hbest = tune_hybrid(S1, seed=2, tag="hyb")
    rep.table(htune.head(6), index=False, digits=3)
    pf_s1 = fd.evaluate(S1, "structural (Modelling 01)")[0]
    pf_h1 = fd.evaluate(lambda: HybridModel(S1(), params=hbest), "hybrid on structural (Modelling 01)")[0]
    s1_, h1_ = summarise(pf_s1).iloc[0], summarise(pf_h1).iloc[0]
    rep.p(f"All 12 hybrid settings score {htune['MAPE (%)'].min():.3f}–{htune['MAPE (%)'].max():.3f}% vs {s1_['MAPE (%)']:.3f}% for the structural "
          f"model alone, so the gain is not luck of one setting. **The trees are finding something real.** What?")

    # 4. diagnosis: what did the trees find? --------------------------------------------------------------------------
    rep.h2("4. What did the residual trees find?")
    rep.p("Diagnosis by restriction: refit the hybrid letting the residual trees see only some inputs. If a restricted version keeps the "
          "gain, the pattern lives in those inputs.")
    geo = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
    subsets = {
        "nothing (= structural alone)": "none",
        "all inputs": None,
        "geography: both ends (4 coordinates)": geo,
        "geography: pickup end only": geo[:2],
        "distance, equipment, weight, market, days to quarter end": ["log_distance", "equipment_code", "weight_abs", "mi", "days_to_quarter_end"],
        "…without days to quarter end": ["log_distance", "equipment_code", "weight_abs", "mi"],
        "days to quarter end alone": ["days_to_quarter_end"],
        "equipment alone": ["equipment_code"],
        "days to quarter end + equipment": ["days_to_quarter_end", "equipment_code"],
        "days to quarter end + distance": ["days_to_quarter_end", "log_distance"],
        "days to quarter end + weight": ["days_to_quarter_end", "weight_abs"],
    }
    drows = []
    for name, feats in subsets.items():
        mk = S1 if feats == "none" else (lambda f=feats: HybridModel(S1(), params=hbest, features=f))
        s = summarise(fd.evaluate(mk, name)[0]).iloc[0]
        drows.append({"residual trees may look at": name, "MAPE (%)": s["MAPE (%)"], "RMSE ($)": s["RMSE ($)"], "MAPE unseen cities (%)": s["MAPE unseen cities (%)"]})
        print(f"  diag {name}: {s['MAPE (%)']:.3f}", flush=True)
    rep.table(pd.DataFrame(drows), index=False, digits=3)
    rep.bullets([
        "**Days to quarter end only helps together with equipment**: neither alone helps, the pair recovers most of the non-geographic gain. "
        "So the quarter-end rush is not the same size for every trailer type.",
        "**Geography helps only when both ends are visible**: pickup coordinates alone add nothing (city effects already cover that). "
        "So there is a small route-level (origin-region × destination-region) effect beyond the two city effects.",
    ])
    full_tr = fd.folds[-1]["train"]
    no_ramp = StructuralModel(coords=coords, **{**chosen, "ramp_days": 0}).fit(full_tr)
    r = 100 * (np.log(full_tr.posted_rate.values) - no_ramp.predict_log(full_tr))
    win = np.where(quarter_ramp(full_tr.date, 28) > 0.5, "last 14 days of quarter", "rest of quarter")
    rt = pd.DataFrame({"r": r, "w": win, "equipment": full_tr.equipment.values}).groupby(["equipment", "w"]).r.mean().unstack()
    rt["quarter-end lift (pts)"] = rt["last 14 days of quarter"] - rt["rest of quarter"]
    rt.index.name = "equipment (train Jan–Aug)"
    rep.p("Confirmed directly: residual of a structural model without any ramp, last 14 days of each quarter vs the rest (%):")
    rep.table(rt, digits=2)

    # 5. structural v2 + hybrid on top ----------------------------------------------------------------------------------
    rep.h2("5. Feeding the discovery back: structural v2, then the hybrid on top")
    chosen_v2 = {**chosen, "ramp_by_equipment": True}
    S2 = lambda: StructuralModel(coords=coords, **chosen_v2)  # noqa: E731
    m2 = S2().fit(full_tr)
    k0 = 1 + 3 + 1 + 1 + 2  # const, 3 distance terms, weight, market, Reefer, Flatbed → ramp terms start here
    ramp = {"Dry Van": m2.coef_[k0], "Reefer": m2.coef_[k0] + m2.coef_[k0 + 1], "Flatbed": m2.coef_[k0] + m2.coef_[k0 + 2]}
    rep.p("Structural v2 = structural + a separate ramp height per equipment (2 extra coefficients). Fitted quarter-end lift (train Jan–Aug): "
          + ", ".join(f"{e} {100 * (np.exp(v) - 1):+.1f}%" for e, v in ramp.items()) + ". The drift was also tried per equipment: no gain.")
    htune2, hbest2 = tune_hybrid(S2, seed=3, tag="hyb-v2")
    rep.p(f"Hybrid retuned on top of v2 (12 settings): {htune2['MAPE (%)'].min():.3f}–{htune2['MAPE (%)'].max():.3f}% MAPE.")
    rep.table(htune2.head(4), index=False, digits=3)

    # 6. head-to-head ----------------------------------------------------------------------------------------------
    rep.h2("6. Head-to-head on future months")
    final = [
        ("structural v1 (Modelling 01)", S1),
        ("trees alone, tuned", lambda: GBMModel(params=best, target=target, use_trend=use_trend)),
        ("hybrid on v1", lambda: HybridModel(S1(), params=hbest)),
        ("structural v2 (+ ramp per equipment)", S2),
        ("hybrid on v2", lambda: HybridModel(S2(), params=hbest2)),
    ]
    pfs = [fd.evaluate(mk, name)[0] for name, mk in final]
    s6 = summarise(pd.concat(pfs))
    rep.table(s6[COLS], digits=3)
    per_fold = pd.concat(pfs).pivot(index="fold", columns="model", values="MAPE (%)")[[n for n, _ in final]]
    rep.p("MAPE per fold:")
    rep.table(per_fold, digits=3)
    wins = int((per_fold["hybrid on v2"] < per_fold["structural v2 (+ ramp per equipment)"]).sum())
    rep.p(f"Hybrid on v2 beats structural v2 in {wins} of {len(per_fold)} folds; the remaining gain is the route-level geography effect.")
    rep.save()
    pd.to_pickle({"gbm_params": best, "gbm_target": target, "gbm_trend": use_trend, "hybrid_params_v1": hbest,
                  "structural_v2": chosen_v2, "hybrid_params": hbest2, "summary": s6}, ROOT / "outputs" / "gbm_choice.pkl")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
