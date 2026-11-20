"""Modelling step 3: compare the finalists, look at their December behaviour, sanity-check Nov–Dec predictions,
and do a PREVIEW run of the official scorer (files go to outputs/preview/, nothing final is written).

Selection rule: lowest future-month MAPE, but if a simpler model is within 0.02 MAPE points (fold noise) and
not worse on RMSE by more than $1, the simpler model wins: easier to explain, fewer ways to break.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from freight.data import (  # noqa: E402
    city_coordinates, clean_inputs, daily_market_index, flag_corrupted, load_december, load_train, load_validation, weight_fill_values,
)
from freight.models import GBMModel, HybridModel, StructuralModel  # noqa: E402
from freight.reporting import AQUA, BLUE, INK_2, ORANGE, Report, set_style  # noqa: E402

PREVIEW = ROOT / "outputs" / "preview"


def main() -> None:
    set_style()
    train, val, dec = load_train(), load_validation(), load_december()
    coords = city_coordinates(train, val)
    s_choice = pd.read_pickle(ROOT / "outputs" / "structural_choice.pkl")
    g_choice = pd.read_pickle(ROOT / "outputs" / "gbm_choice.pkl")
    chosen = g_choice["structural_v2"]
    rep = Report("03_compare", "Modelling 03 · Finalists, December behaviour and a preview run of the scorer")

    # 1. scoreboard ----------------------------------------------------------------------------------------------
    rep.h2("1. Scoreboard (future months, 4 folds, hidden cities)")
    board = pd.concat([s_choice["ladder"].iloc[[0]], g_choice["summary"]])
    cols = ["MAPE (%)", "RMSE ($)", "MAE ($)", "MAPE seen cities (%)", "MAPE unseen cities (%)", "MAPE month 2 (%)", "bias (%)", "worst-fold MAPE (%)"]
    rep.table(board[cols], digits=3)
    s, h = g_choice["summary"].loc["structural v2 (+ ramp per equipment)"], g_choice["summary"].loc["hybrid on v2"]
    simpler_wins = (h["MAPE (%)"] > s["MAPE (%)"] - 0.02) and (h["RMSE ($)"] > s["RMSE ($)"] - 1.0)
    winner = "structural" if simpler_wins else "hybrid"
    rep.p(f"Selection rule: the simpler model wins unless the more complex one is better by ≥ 0.02 MAPE points and ≥ $1 RMSE. "
          f"Hybrid on v2: {h['MAPE (%)']:.3f}% / ${h['RMSE ($)']:.1f}; structural v2: {s['MAPE (%)']:.3f}% / ${s['RMSE ($)']:.1f}. " + (
              "Within noise, so **the simpler structural v2 is the front-runner**." if simpler_wins else
              "The gain is clear, so **the hybrid on v2 is the front-runner**."))

    # 2. final fits on all clean training data -------------------------------------------------------------------------
    dmi = daily_market_index(train, val)
    wf = weight_fill_values(train)
    full = clean_inputs(train, dmi, wf)
    fit_df = full[~flag_corrupted(full)]
    v = clean_inputs(val, dmi, wf)
    d = clean_inputs(dec, dmi, wf, coords=coords)
    models = {
        "structural": StructuralModel(coords=coords, **chosen),
        "trees (tuned)": GBMModel(params=g_choice["gbm_params"], target=g_choice["gbm_target"], use_trend=g_choice["gbm_trend"]),
        "hybrid": HybridModel(StructuralModel(coords=coords, **chosen), params=g_choice["hybrid_params"]),
    }
    pv, pd_ = {}, {}
    for name, m in models.items():
        m.fit(fit_df)
        pv[name], pd_[name] = m.predict(v), m.predict(d)
    rep.p(f"Final fits use all {len(fit_df):,} clean training rows ({len(full) - len(fit_df)} corrupted labels removed), all 64 cities.")

    # 3. December ---------------------------------------------------------------------------------------------------
    rep.h2("2. December chart behaviour (Lexington → Fort Wayne, 360 mi, Dry Van, 32,000 lb)")
    rep.p("The December file has no market_index or coordinates. Coordinates come from the city lookup; market_index is each December "
          f"day's average over the validation file (≈{val[val.date.dt.month == 12].groupby('date').size().mean():.0f} loads per day).")
    fig, ax = plt.subplots(figsize=(12, 4))
    for (name, p), color in zip(pd_.items(), (BLUE, ORANGE, AQUA)):
        ax.plot(d.date, p, color=color, marker="o", markersize=3, label=name)
    ax.axhline(808, color=INK_2, linewidth=0.8)
    ax.annotate("lane's Dry Van median in train: $808", (d.date.iloc[0], 808), xytext=(0, -12), textcoords="offset points", fontsize=7.5, color=INK_2)
    ax.set(title="Predicted December price by model", ylabel="predicted rate ($)")
    ax.legend(loc="upper left")
    rep.figure(fig, "december_models", "What each finalist draws. Expected from the EDA: weekly zig-zag (Thu high, Sun low) rising into Dec 31 (quarter-end ramp).")
    dt = pd.DataFrame({"date": d.date.values, "dow": d.date.dt.day_name().str[:3].values, "market_index": d.mi.values, **{k: v_ for k, v_ in pd_.items()}}).set_index("date")
    rep.table(dt, digits=3)
    first, last = dt.iloc[:7], dt.iloc[-7:]
    rep.bullets([f"{k}: first week avg ${first[k].mean():.0f}, last week avg ${last[k].mean():.0f} ({100 * (last[k].mean() / first[k].mean() - 1):+.1f}%), "
                 f"weekly swing ≈ ${dt[k].groupby(dt.dow).mean().max() - dt[k].groupby(dt.dow).mean().min():.0f}" for k in pd_])

    # 4. validation sanity ---------------------------------------------------------------------------------------------
    rep.h2("3. Sanity checks on the 12,000 Nov–Dec predictions")
    rpm_v = {k: p / v.distance.values for k, p in pv.items()}
    sanity = pd.DataFrame({k: {"min $": p.min(), "median $": np.median(p), "max $": p.max(), "any ≤ 0": bool((p <= 0).any()),
                               "median $/mile Nov": np.median(rpm_v[k][v.date.dt.month.values == 11]),
                               "median $/mile Dec": np.median(rpm_v[k][v.date.dt.month.values == 12])} for k, p in pv.items()}).T
    rep.table(sanity, digits=3)
    tr_rpm = (fit_df.posted_rate / fit_df.distance).groupby(fit_df.date.dt.month).median()
    rep.p("Train median $/mile by month for comparison: " + ", ".join(f"{pd.Timestamp(2025, m, 1):%b} {r:.3f}" for m, r in tr_rpm.items()) + ".")
    agree = 100 * np.abs(pv["hybrid"] / pv["structural"] - 1)
    agree_t = 100 * np.abs(pv["trees (tuned)"] / pv["structural"] - 1)
    new_c = sorted((set(val.pickup) | set(val.delivery)) - set(train.pickup) - set(train.delivery))
    is_new = (val.pickup.isin(new_c) | val.delivery.isin(new_c)).values
    rep.bullets([
        f"Structural vs hybrid: median difference {np.median(agree):.2f}%, 99th percentile {np.percentile(agree, 99):.2f}%.",
        f"Structural vs trees: median difference {np.median(agree_t):.2f}%, 99th percentile {np.percentile(agree_t, 99):.2f}%; "
        f"on loads touching a new city {np.median(agree_t[is_new]):.2f}% vs {np.median(agree_t[~is_new]):.2f}% elsewhere.",
    ])

    # 5. preview scorer run --------------------------------------------------------------------------------------------------
    rep.h2(f"4. Preview: official scorer on the front-runner ({winner})")
    PREVIEW.mkdir(parents=True, exist_ok=True)
    tmpl = pd.read_csv(ROOT / "data" / "validation_predictions_template.csv")
    sub = tmpl[["load_id"]].merge(pd.DataFrame({"load_id": val.load_id, "predicted_rate": np.round(pv[winner], 2)}), on="load_id", how="left")
    sub.to_csv(PREVIEW / "validation_predictions.csv", index=False)
    dec_out = pd.read_csv(ROOT / "data" / "december_chart_inputs.csv")
    dec_out["predicted_rate"] = np.round(pd_[winner], 2)
    dec_out.to_csv(PREVIEW / "december_chart_inputs.csv", index=False)
    res = subprocess.run([sys.executable, str(ROOT / "score.py"), "--predictions", str(PREVIEW / "validation_predictions.csv"),
                          "--december-predictions", str(PREVIEW / "december_chart_inputs.csv"), "--output-dir", str(PREVIEW / "scorer_results")],
                         capture_output=True, text=True, cwd=ROOT)
    rep.p("Scorer output:")
    rep.lines += ["```", (res.stdout + res.stderr).strip(), "```", ""]
    png = PREVIEW / "scorer_results" / "candidate_december.png"
    if png.exists():
        shutil.copy(png, rep.fig_dir / "03_compare__scorer_preview.png")
        rep.lines += ["![Scorer preview](figures/03_compare__scorer_preview.png)", "", "*The chart exactly as score.py draws it (preview, not final).*", ""]
    rep.save()
    print(res.stdout, res.stderr)


if __name__ == "__main__":
    main()
