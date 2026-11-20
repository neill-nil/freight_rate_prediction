"""Modelling step 1: baselines and the structural model, built one ingredient at a time.

Why this model first: the EDA showed price is a product of independent effects with ~1.8% normal noise
(eda/09) plus a drift and quarter-end ramp (eda/10). A least-squares fit on log price with exactly those
terms is the simplest model that matches the observed pattern, and every coefficient is readable.

Sections:
  1. how models are judged (time-forward folds with hidden cities)
  2. ladder: add one ingredient at a time, scored on future months
  3. hyperparameters: sweep each one around the EDA-derived value
  4. where the chosen model still errs (by fold, distance, equipment, time)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from freight.data import city_coordinates, load_train, load_validation  # noqa: E402
from freight.evaluation import FoldData, summarise  # noqa: E402
from freight.models import NaiveRatePerMile, StructuralModel  # noqa: E402
from freight.reporting import AQUA, BLUE, INK_2, ORANGE, YELLOW, Report, set_style  # noqa: E402

KEY = ["MAPE (%)", "RMSE ($)", "MAE ($)", "median APE (%)", "MAPE seen cities (%)", "MAPE unseen cities (%)",
       "MAPE month 1 (%)", "MAPE month 2 (%)", "bias (%)", "worst-fold MAPE (%)"]


def main() -> None:
    set_style()
    train, val = load_train(), load_validation()
    coords = city_coordinates(train, val)
    fd = FoldData(train, val)
    rep = Report(
        "01_structural",
        "Modelling 01 · Baselines and the structural model",
        "Every number on this page is measured on **future months the model never saw**, averaged over 4 folds, on test rows "
        "with genuine labels. Lower is better for every column except bias (best near 0).",
    )

    # 1. evaluation design --------------------------------------------------------------------------------
    rep.h2("1. How models are judged")
    rep.p("The real task is: learn from Jan–Oct, predict the next **two** months, where 12% of loads involve a city never seen. "
          "Each fold copies that: train up to a cut-off, test on the next two months, and hide 4 random cities from training. "
          "Cleaning and corrupted-label removal are redone inside each fold using the training part only (no peeking).")
    ft = pd.DataFrame([{
        "fold": f["name"], "train rows (clean)": len(f["train"]), "test rows": len(f["test"]),
        "hidden cities": ", ".join(f["hidden"]), "test rows with an unseen city": f"{100 * f['unseen'].mean():.1f}%",
        "test rows with genuine labels": f"{100 * f['genuine'].mean():.1f}%",
    } for f in fd.folds])
    rep.table(ft, index=False)
    rep.bullets([
        "**MAPE** = average % error (every load counts equally). **RMSE** = typical $ error, dominated by long, expensive loads.",
        "Also shown: MAE ($), median % error, MAPE split by seen/unseen cities and by first/second test month, median bias "
        "(+ = over-predicting), and the worst fold's MAPE (stability).",
        "Corrupted test labels (~1.4%) are excluded from scoring because nobody can predict a randomly ×4 price; "
        "'RMSE all rows' in the detailed tables shows the effect of keeping them.",
    ])

    # 2. ladder ---------------------------------------------------------------------------------------------
    rep.h2("2. Ladder: one ingredient at a time")
    S = lambda **kw: (lambda: StructuralModel(coords=coords, **kw))  # noqa: E731
    ladder = [
        ("0 naive: typical price per mile × distance", lambda: NaiveRatePerMile()),
        ("1 + distance curve + equipment", S(use_weight=False, use_market=False, use_cities=False, ramp_days=0, trend="none")),
        ("2 + weight (20k–40k band)", S(use_market=False, use_cities=False, ramp_days=0, trend="none")),
        ("3 + market_index", S(use_cities=False, ramp_days=0, trend="none")),
        ("4 + city effects (unseen city = average)", S(ramp_days=0, trend="none", unseen="mean")),
        ("5 + quarter-end ramp", S(trend="none", unseen="mean")),
        ("6 + slow drift (continues into test)", S(trend="continue", unseen="mean")),
        ("7 + unseen cities from 5 nearest neighbours", S(trend="continue", unseen="knn")),
    ]
    per = []
    for name, mk in ladder:
        pf, _ = fd.evaluate(mk, name)
        per.append(pf)
    lad = summarise(pd.concat(per))
    rep.table(lad[KEY], digits=2)
    steps = lad["MAPE (%)"].values
    rep.p("What each step bought (MAPE change): " + "; ".join(
        f"{lad.index[i].split(' ', 1)[1]}: {steps[i] - steps[i - 1]:+.2f}" for i in range(1, len(steps))) + ".")
    fig, ax = plt.subplots(figsize=(11, 3.8))
    x = np.arange(len(lad))
    ax.bar(x, lad["MAPE (%)"], color=BLUE, width=0.6)
    for i, v in enumerate(lad["MAPE (%)"]):
        ax.text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=8, color=INK_2)
    ax.axhline(1.5, color=ORANGE, linewidth=1)
    ax.text(len(lad) - 0.5, 1.6, "noise floor ≈ 1.5%", color=ORANGE, fontsize=8, ha="right")
    ax.set_xticks(x, [n.split(" ", 1)[0] for n in lad.index])
    ax.set(title="Future-month MAPE as ingredients are added (step numbers as in the table)", ylabel="MAPE (%)", xlabel="ladder step")
    ax.grid(axis="x", visible=False)
    rep.figure(fig, "ladder", "Distance + equipment do most of the work; weight, city effects and the two time terms close most of the remaining gap. "
               "market_index only pays off once the drift is in the model (step 3 vs 6).")
    MI = 1 + 3 + 1  # coefficient position: const, 3 distance terms, weight, then market_index
    conf = pd.DataFrame([{
        "fold": f["name"],
        "market coefficient, no time terms": StructuralModel(use_cities=False, ramp_days=0, trend="none").fit(f["train"]).coef_[MI],
        "market coefficient, with ramp + drift": StructuralModel(use_cities=False, ramp_days=28, trend="continue").fit(f["train"]).coef_[MI],
    } for f in fd.folds])
    rep.h3("Why adding market_index alone made things worse (step 3)")
    rep.table(conf, index=False, digits=3)
    rep.p("Without time terms, the market coefficient is inflated (it soaks up the upward drift that happens to coincide with the "
          "market's spring rise), so it mispredicts the future (bias −2.9%). With the drift in the model the coefficient settles at "
          "≈ 0.13, the value found in the EDA, and market_index becomes useful. Ingredients interact through confounding, which is why "
          "the order of the ladder matters and why the final model has to include all of them together.")

    # 3. hyperparameters --------------------------------------------------------------------------------------
    rep.h2("3. Hyperparameters: sweep each around its EDA value")
    rep.p("The structural model has only a handful of settings, each suggested by the EDA. Each is swept on its own with the others at "
          "their default (marked ◀), scored on the same future-month folds. Rule: move away from the EDA value only if MAPE improves by "
          "≥ 0.02 points **and** RMSE also improves. Smaller differences are within fold-to-fold noise, and chasing them would tune "
          "to these 4 folds rather than to the future.")
    base = dict(distance_degree=3, weight_bounds=(20_000, 40_000), ramp_days=28, trend="continue", unseen="knn", knn_k=5)
    sweeps = {
        "drift handling": ("trend", ["none", "freeze", "damped", "continue"]),
        "quarter-end ramp length (days)": ("ramp_days", [0, 14, 21, 28, 35, 42]),
        "weight shape": ("weight_bounds", [None, (5_000, 47_500), (15_000, 40_000), (20_000, 40_000), (20_000, 45_000), (25_000, 40_000)]),
        "distance curve (polynomial degree in log distance)": ("distance_degree", [1, 2, 3, 4]),
        "neighbours used for unseen cities": ("knn_k", [1, 3, 5, 8]),
    }
    chosen = dict(base)
    sweep_results = {}
    for title, (param, values) in sweeps.items():
        rows = []
        for v in values:
            kw = {**base, param: v}
            label = {None: "log weight (no band)", (5_000, 47_500): "linear (no band)"}.get(v, v) if param == "weight_bounds" else v
            pf, _ = fd.evaluate(lambda kw=kw: StructuralModel(coords=coords, **kw), str(label))
            s = summarise(pf).iloc[0]
            rows.append({"value": f"{label}{' ◀' if v == base[param] else ''}", **{k: s[k] for k in ["MAPE (%)", "RMSE ($)", "worst-fold MAPE (%)", "MAPE month 2 (%)", "MAPE unseen cities (%)", "bias (%)"]}})
        tab = pd.DataFrame(rows)
        sweep_results[title] = tab
        rep.h3(title)
        rep.table(tab, index=False, digits=3)
        if param == "trend":
            tf = pd.concat([fd.evaluate(lambda v=v: StructuralModel(coords=coords, **{**base, "trend": v}), v)[0] for v in values[1:]])
            tfp = tf.pivot(index="fold", columns="model", values="MAPE (%)")[values[1:]]
            tfp.columns = [f"MAPE {c} (%)" for c in tfp.columns]
            rep.p("Per fold (MAPE %): 'freeze' and 'continue' each win two folds, so the data cannot tell them apart. The month-to-month "
                  "wiggle the EDA found (±0.5%) decides each fold. 'Damped' (half speed, i.e. the average of the two) is a hedge that is "
                  "never the worst.")
            rep.table(tfp, digits=3)
        best_i = int(tab["MAPE (%)"].idxmin())
        default_i = [i for i, v in enumerate(values) if v == base[param]][0]
        gain_mape = tab["MAPE (%)"][default_i] - tab["MAPE (%)"][best_i]
        gain_rmse = tab["RMSE ($)"][default_i] - tab["RMSE ($)"][best_i]
        if best_i != default_i and gain_mape >= 0.02 and gain_rmse > 0:
            chosen[param] = values[best_i]
            rep.p(f"→ switch to **{tab['value'][best_i]}** (MAPE −{gain_mape:.3f} pts, RMSE −${gain_rmse:.2f}).")
        else:
            rep.p(f"→ keep the EDA value **{tab['value'][default_i].replace(' ◀', '')}** (best alternative gains {gain_mape:.3f} MAPE pts, ${gain_rmse:.2f} RMSE).")

    # 4. chosen model ---------------------------------------------------------------------------------------------
    rep.h2("4. The chosen structural model")
    rep.p("Settings: " + ", ".join(f"{k} = {v}" for k, v in chosen.items()))
    pf, preds = fd.evaluate(lambda: StructuralModel(coords=coords, **chosen), "chosen structural")
    rep.table(pf.drop(columns="model").set_index("fold")[["MAPE (%)", "RMSE ($)", "MAE ($)", "MAPE seen cities (%)", "MAPE unseen cities (%)", "MAPE month 1 (%)", "MAPE month 2 (%)", "bias (%)", "RMSE all rows ($)"]], digits=2)

    allte = pd.concat([f["test"].assign(pred=p, genuine=f["genuine"], fold=f["name"]) for f, p in zip(fd.folds, preds)])
    g = allte[allte.genuine].copy()
    g["ape"] = 100 * np.abs(g.pred / g.posted_rate - 1)
    g["err"] = 100 * (g.pred / g.posted_rate - 1)
    br = pd.concat({
        "distance": g.groupby(pd.cut(g.distance, [69, 300, 600, 1000, 1500, 2000, 3500]), observed=True).ape.agg(["mean", "size"]),
        "equipment": g.groupby("equipment").ape.agg(["mean", "size"]),
    })
    br.columns = ["MAPE (%)", "test rows"]
    br.index = [f"{a}: {b}" for a, b in br.index]
    rep.p("Where the errors are (all folds pooled):")
    rep.table(br, digits=2)

    fig, ax = plt.subplots(figsize=(12, 3.6))
    for (name, d), color in zip(g.groupby("fold", sort=False), (BLUE, ORANGE, AQUA, YELLOW)):
        s = d.groupby("date").err.median()
        ax.plot(s.index, s.rolling(3, center=True, min_periods=1).mean(), color=color, linewidth=1.3, label=name)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set(title="Daily median error of the chosen model on each fold's test months (3-day smoothed)", ylabel="error (%, + = over-predicting)")
    ax.legend(loc="upper left", ncol=2)
    rep.figure(fig, "bias_over_time", "Systematic drifts away from 0 would show the time terms mis-extrapolating. Overlapping folds share months, so lines should roughly agree.")
    rep.save()
    pd.to_pickle({"chosen": chosen, "ladder": lad, "sweeps": sweep_results}, ROOT / "outputs" / "structural_choice.pkl")


if __name__ == "__main__":
    main()
