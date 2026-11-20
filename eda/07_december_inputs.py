"""07 · The December chart inputs — what will the model see, and what is missing?

The scorer's chart holds Lexington → Fort Wayne, 360 mi, Dry Van, 32,000 lb fixed and changes only the
date. The file has no coordinates, market_index or quote_signal, so we need to know:
  * what this lane looks like historically (level, spread, distance, weight)
  * which inputs can be recovered, and from where
  * what the calendar and market look like across December
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    BLUE,
    EQUIP_COLOR,
    EQUIPMENT,
    INK_2,
    ORANGE,
    SPLIT_COLOR,
    US_HOLIDAYS_2025,
    Report,
    add_calendar,
    add_rate_features,
    city_coords,
    daily_market_index,
    flag_outliers,
    load_raw,
    relative_rate,
    set_style,
)


def main() -> None:
    set_style()
    train, val, dec = load_raw()
    tr = add_rate_features(train)
    rel, _ = relative_rate(tr)
    tr = tr.join(rel)
    tr["is_outlier"] = flag_outliers(tr.rel_rate)
    P, D = dec.pickup.iloc[0], dec.delivery.iloc[0]
    rep = Report(
        "07_december_inputs",
        "07 · December chart inputs",
        f"Fixed inputs: {P} → {D}, {dec.distance.iloc[0]} mi, {dec.equipment.iloc[0]}, {dec.weight.iloc[0]:,} lb; only the date changes.",
    )

    # 1. what's missing ----------------------------------------------------------------------------
    rep.h2("1. Which model inputs are available?")
    cc = city_coords(train, val).drop_duplicates("city").set_index("city")
    dmi = daily_market_index(val)
    avail = pd.DataFrame(
        [
            {"input": "pickup, delivery, distance, equipment, weight, date", "in December file": "yes", "how to obtain": "given"},
            {"input": "pickup/delivery lat & lon", "in December file": "no",
             "how to obtain": f"city lookup from train ({P}: {cc.loc[P, 'lat']:.3f}, {cc.loc[P, 'lon']:.3f}; {D}: {cc.loc[D, 'lat']:.3f}, {cc.loc[D, 'lon']:.3f})"},
            {"input": "market_index", "in December file": "no",
             "how to obtain": f"daily mean from validation.csv: all {dmi.loc['2025-12'].shape[0]} December dates present, ~{val[val.date.dt.month == 12].groupby('date').size().mean():.0f} loads per date"},
            {"input": "quote_signal", "in December file": "no", "how to obtain": "not needed if the feature is dropped (step 06)"},
        ]
    )
    rep.table(avail, index=False)

    # 2. lane history ------------------------------------------------------------------------------
    rep.h2(f"2. History of {P} → {D}")
    lane = tr[(tr.pickup == P) & (tr.delivery == D)].copy()
    rev = tr[(tr.pickup == D) & (tr.delivery == P)]
    summ = lane.groupby("equipment").agg(loads=("rpm", "size"), median_rate=("posted_rate", "median"), min_rate=("posted_rate", "min"), max_rate=("posted_rate", "max"),
                                         median_rpm=("rpm", "median"), mean_distance=("distance", "mean"), median_weight=("weight", "median"), outliers=("is_outlier", "sum"))
    rep.table(summ.reindex([e for e in EQUIPMENT if e in summ.index]), digits=3)
    dv = lane[(lane.equipment == "Dry Van") & ~lane.is_outlier]
    rep.bullets(
        [
            f"Reverse lane {D} → {P}: {len(rev)} loads in train, median Dry Van rate ${rev[rev.equipment == 'Dry Van'].posted_rate.median():,.0f}.",
            f"Lane distance in train: mean {lane.distance.mean():.1f}, range {lane.distance.min():.1f}–{lane.distance.max():.1f} (chart fixes 360).",
            f"Dry Van weights in train: {dv.weight.abs().min():,.0f}–{dv.weight.abs().max():,.0f} lb (chart fixes 32,000).",
            f"Validation also has {((val.pickup == P) & (val.delivery == D)).sum()} loads on this lane (inputs only).",
        ]
    )
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
    ax = axes[0]
    for e in EQUIPMENT:
        g = lane[(lane.equipment == e) & ~lane.is_outlier]
        if len(g):
            ax.scatter(g.date, g.posted_rate, color=EQUIP_COLOR[e], s=30, label=e, edgecolors="white", linewidths=1, zorder=3)
    ax.axhline(dv.posted_rate.median(), color=INK_2, linewidth=0.9)
    ax.annotate(f"Dry Van median ${dv.posted_rate.median():,.0f}", (lane.date.min(), dv.posted_rate.median()), xytext=(0, 4), textcoords="offset points", fontsize=7.5, color=INK_2)
    ax.set(title=f"{P} → {D}: every load in train (corrupted labels removed)", ylabel="posted_rate ($)")
    ax.legend(loc="upper right")
    allm = daily_market_index(train, val)
    ax = axes[1]
    ax.plot(allm.loc[:"2025-10-31"].index, allm.loc[:"2025-10-31"].values, color=SPLIT_COLOR["train"], linewidth=0.9, label="train period")
    ax.plot(allm.loc["2025-11-01":].index, allm.loc["2025-11-01":].values, color=SPLIT_COLOR["validation"], linewidth=0.9, label="validation period (incl. December)")
    ax.set(title="Daily mean market_index", ylabel="index")
    ax.legend(loc="upper right")
    fig.tight_layout()
    rep.figure(fig, "lane_history", "The lane's Dry Van rate moves with the market. December's market sits at a level the lane has already seen (Jan, Sep–Oct).")

    # 3. December calendar & market ------------------------------------------------------------------
    rep.h2("3. December: calendar and market context")
    vd = add_calendar(val[val.date.dt.month == 12])
    cal = vd.groupby("date").agg(dow=("dow_name", "first"), days_to_quarter_end=("days_to_quarter_end", "first"), loads=("load_id", "size"),
                                 market_index=("market_index", "mean"), market_index_std=("market_index", "std"), quote_signal=("quote_signal", "mean"))
    hol = {pd.Timestamp(v): k for k, v in US_HOLIDAYS_2025.items()}
    cal["holiday"] = [hol.get(d, "") for d in cal.index]
    rep.table(cal, digits=3)
    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.plot(cal.index, cal.market_index, color=ORANGE, marker="o", linewidth=1.6)
    for d, r in cal.iterrows():
        if r.dow in ("Thu", "Sun"):
            ax.annotate(r.dow, (d, r.market_index), xytext=(0, 6 if r.dow == "Thu" else -12), textcoords="offset points", ha="center", fontsize=7, color=INK_2)
    ax.axvspan(pd.Timestamp("2025-12-08"), pd.Timestamp("2025-12-31 12:00"), color=BLUE, alpha=0.06, linewidth=0)
    ax.annotate("last ~24 days of Q4 (quarter-end ramp window seen in Q1–Q3)", (pd.Timestamp("2025-12-08"), cal.market_index.max()), xytext=(4, 2), textcoords="offset points", fontsize=7.5, color=INK_2)
    ax.set(title="December 2025: daily mean market_index from validation.csv", ylabel="market_index")
    rep.figure(fig, "december_market", "Weekly cycle (Thu peak, Sun trough) with a slight upward level through the month. The shaded window is where Q1–Q3 showed the quarter-end ramp.")
    rep.bullets(
        [
            f"December market_index ranges {cal.market_index.min():.3f}–{cal.market_index.max():.3f}; weekly swing ≈ {cal.groupby('dow').market_index.mean().max() - cal.groupby('dow').market_index.mean().min():.3f}.",
            f"Christmas (2025-12-25) falls on a {pd.Timestamp('2025-12-25').day_name()}; step 05 found no holiday effect in train.",
            "Implication: a model that uses market_index and a quarter-end feature should draw a weekly zig-zag that ramps up "
            "into Dec 31. A model with neither draws a flat line. A model trained on `month` can't say anything sensible about "
            "December, a month it never saw.",
        ]
    )
    _ = np
    rep.save()


if __name__ == "__main__":
    main()
