"""Modelling step 9: is "hybrid on v3" reliably better than structural v3 on data neither has seen?

Hybrid on v3 (trees fitted to v3's leftover errors) was ~1% better in every hidden-city draw of step 8, but below the
clear-gain bar. Before finalising we test it harder. The criteria were fixed BEFORE running; the hybrid is chosen only if
it meets all of them:
  C1  fresh rounds: two earlier rounds never used for any tuning (train Jan–Mar → test Apr–May, Jan–Apr → May–Jun):
      better on average in both MAPE and RMSE
  C2  consistency: better in ≥ 75% of the 24 (6 rounds × 4 hidden-city draws) paired comparisons, in MAPE and in RMSE
  C3  downside: never worse than v3 by more than 0.03 MAPE points in any round × draw
  C4  unseen cities: not worse on average for loads with a city never seen in training
  C5  Nov–Dec behaviour: fitted on all of Jan–Oct, the trees' typical adjustment in each month is within ±0.2%
      (i.e. they refine individual loads rather than bet on a level shift we cannot check)
Also reported: what the trees learn (residual trees restricted to one feature group at a time).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from freight.data import (  # noqa: E402
    city_coordinates, clean_inputs, daily_market_index, flag_corrupted, load_december, load_train, load_validation, weight_fill_values,
)
from freight.evaluation import FOLDS, FoldData  # noqa: E402
from freight.models import HybridModel, StructuralModel  # noqa: E402
from freight.reporting import Report, set_style  # noqa: E402

EARLY = [("train Jan–Mar → test Apr–May (fresh)", "2025-03-31", "2025-05-31"),
         ("train Jan–Apr → test May–Jun (fresh)", "2025-04-30", "2025-06-30")]
SEEDS = (0, 100, 200, 300)
GROUPS = {"quarter-end timing + equipment": ["days_to_quarter_end", "equipment_code"],
          "geography (4 coordinates)": ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"],
          "market index": ["mi"], "distance + weight": ["log_distance", "weight_abs"], "all 9 features": None}


def main() -> None:
    set_style()
    t0 = time.time()
    train, val, dec = load_train(), load_validation(), load_december()
    coords = city_coordinates(train, val)
    v3 = pd.read_pickle(ROOT / "outputs" / "feature_choice.pkl")["structural_v3"]
    hp = pd.read_pickle(ROOT / "outputs" / "gbm_choice.pkl")["hybrid_params"]

    def mk_v3():
        return StructuralModel(coords=coords, **v3)

    def mk_hyb(features=None):
        return lambda: HybridModel(StructuralModel(coords=coords, **v3), params=hp, features=features)

    rep = Report("09_hybrid_check", "Modelling 09 · Is the hybrid reliably better than structural v3?",
                 "Hybrid on v3 = structural v3 + gradient-boosted trees (400 trees, 31 leaves, ≥ 100 loads per leaf) fitted to v3's log "
                 "leftovers. Criteria fixed before running; the hybrid is chosen only if it meets all five.")
    rows = []
    for seed in SEEDS:
        fds = FoldData(train, val, folds=FOLDS + EARLY, seed=seed)
        for label, factory in (("v3", mk_v3), ("hybrid", mk_hyb())):
            pf, _ = fds.evaluate(factory, label)
            rows.append(pf.assign(draw=seed))
        print(f"  draw {seed} done ({time.time() - t0:.0f}s)", flush=True)
    allpf = pd.concat(rows)
    wide = allpf.pivot_table(index=["draw", "fold"], columns="model", values=["MAPE (%)", "RMSE ($)", "MAPE unseen cities (%)"])
    d = pd.DataFrame({m: wide[(m, "hybrid")] - wide[(m, "v3")] for m in ("MAPE (%)", "RMSE ($)", "MAPE unseen cities (%)")})
    d = d.reset_index()
    d["fresh"] = d["fold"].str.contains("fresh")

    rep.h2("1. Round by round (hybrid − v3, averaged over the 4 hidden-city draws; negative = hybrid better)")
    per_fold = d.groupby("fold", sort=False).agg(**{"Δ MAPE": ("MAPE (%)", "mean"), "Δ RMSE ($)": ("RMSE ($)", "mean"),
                                                   "Δ MAPE unseen cities": ("MAPE unseen cities (%)", "mean"),
                                                   "draws hybrid better (MAPE)": ("MAPE (%)", lambda s: int((s < 0).sum())),
                                                   "draws hybrid better (RMSE)": ("RMSE ($)", lambda s: int((s < 0).sum()))})
    lv = allpf.groupby(["model", "fold"], sort=False)[["MAPE (%)", "RMSE ($)"]].mean().unstack(0)
    per_fold.insert(0, "v3 MAPE", lv[("MAPE (%)", "v3")].reindex(per_fold.index))
    per_fold.insert(1, "hybrid MAPE", lv[("MAPE (%)", "hybrid")].reindex(per_fold.index))
    rep.table(per_fold, digits=3)

    c1_m, c1_r = d.loc[d.fresh, "MAPE (%)"].mean(), d.loc[d.fresh, "RMSE ($)"].mean()
    win_m, win_r = (d["MAPE (%)"] < 0).mean(), (d["RMSE ($)"] < 0).mean()
    worst = d["MAPE (%)"].max()
    unseen = d["MAPE unseen cities (%)"].mean()

    # Nov–Dec behaviour ----------------------------------------------------------------------------------------------
    dmi = daily_market_index(train, val)
    wf = weight_fill_values(train)
    full = clean_inputs(train, dmi, wf)
    c = full[~flag_corrupted(full)]
    v = clean_inputs(val, dmi, wf)
    dd = clean_inputs(dec, dmi, wf, coords=coords)
    s_model = mk_v3().fit(c)
    h_model = mk_hyb()().fit(c)
    adj = 100 * (np.exp(h_model.predict_log(v) - s_model.predict_log(v)) - 1)
    known = set(c.pickup) | set(c.delivery)
    new_city = ~(v.pickup.isin(known) & v.delivery.isin(known)).values
    nov = (v.date.dt.month == 11).values
    beh = pd.DataFrame([
        {"loads": k, "n": int(m.sum()), "median adjustment (%)": np.median(adj[m]), "mean (%)": adj[m].mean(),
         "5th pct (%)": np.percentile(adj[m], 5), "95th pct (%)": np.percentile(adj[m], 95)}
        for k, m in (("November", nov), ("December", ~nov), ("known cities", ~new_city), ("new city", new_city), ("all", np.ones(len(v), bool)))])
    dec_adj = 100 * (h_model.predict(dd) / s_model.predict(dd) - 1)
    c5 = beh.loc[beh["loads"].isin(["November", "December"]), "median adjustment (%)"].abs().max()

    rep.h2("2. Criteria")
    crit = pd.DataFrame([
        ("C1 fresh rounds: better on average (MAPE, RMSE)", f"{c1_m:+.3f} pts, {c1_r:+.2f} $", c1_m < 0 and c1_r < 0),
        ("C2 better in ≥ 75% of the 24 round × draw pairs", f"MAPE {100 * win_m:.0f}%, RMSE {100 * win_r:.0f}%", win_m >= 0.75 and win_r >= 0.75),
        ("C3 never worse by > 0.03 MAPE points", f"largest loss {worst:+.3f} pts", worst <= 0.03),
        ("C4 unseen-city loads not worse on average", f"{unseen:+.3f} pts", unseen <= 0),
        ("C5 Nov–Dec monthly median adjustment within ±0.2%", f"{c5:.2f}%", c5 <= 0.2),
    ], columns=["criterion", "result", "met"])
    rep.table(crit, index=False)
    choose_hybrid = bool(crit["met"].all())
    overall = d[["MAPE (%)", "RMSE ($)"]].mean()
    lvl = allpf.groupby("model")[["MAPE (%)", "RMSE ($)"]].mean()
    rep.p(f"Average over all 24 pairs: hybrid − v3 = {overall['MAPE (%)']:+.3f} MAPE points, {overall['RMSE ($)']:+.2f} $ RMSE "
          f"(v3 {lvl.loc['v3', 'MAPE (%)']:.3f}% / ${lvl.loc['v3', 'RMSE ($)']:.2f}; "
          f"hybrid {lvl.loc['hybrid', 'MAPE (%)']:.3f}% / ${lvl.loc['hybrid', 'RMSE ($)']:.2f}).")
    rep.p(f"**Decision: {'hybrid on v3' if choose_hybrid else 'structural v3'}** "
          f"({'all five criteria met' if choose_hybrid else 'not all criteria met'}).")

    rep.h2("3. What the trees change in Nov–Dec (both fitted on all of Jan–Oct)")
    rep.p("Adjustment = hybrid prediction ÷ v3 prediction − 1, on the 12,000 validation loads.")
    rep.table(beh, index=False, digits=3)
    rep.p(f"December chart (Lexington → Fort Wayne): the trees change the 31 daily prices by {dec_adj.min():+.2f}% to {dec_adj.max():+.2f}%.")

    rep.h2("4. What the trees learn")
    rep.p("Residual trees allowed to see one feature group only (4 standard rounds, first hidden-city draw).")
    base = allpf[(allpf.draw == 0) & (allpf.model == "v3") & ~allpf.fold.str.contains("fresh")]
    std = FoldData(train, val)
    g_rows = [{"trees see": "nothing (structural v3)", "MAPE (%)": base["MAPE (%)"].mean(), "RMSE ($)": base["RMSE ($)"].mean()}]
    for name, feats in GROUPS.items():
        pf, _ = std.evaluate(mk_hyb(feats), name)
        g_rows.append({"trees see": name, "MAPE (%)": pf["MAPE (%)"].mean(), "RMSE ($)": pf["RMSE ($)"].mean()})
        print(f"  group {name} done ({time.time() - t0:.0f}s)", flush=True)
    gt = pd.DataFrame(g_rows)
    gt["gain vs v3 (MAPE pts)"] = gt["MAPE (%)"].iloc[0] - gt["MAPE (%)"]
    rep.table(gt, index=False, digits=3)
    rep.save()
    pd.to_pickle({"pairs": d, "criteria": crit, "choose_hybrid": choose_hybrid, "behaviour": beh, "groups": gt},
                 ROOT / "outputs" / "hybrid_check.pkl")
    print(f"decision: {'hybrid' if choose_hybrid else 'v3'}  done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
