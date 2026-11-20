"""05 · Time — what changes with the date, and will it carry over to Nov–Dec?

Why this matters twice: validation is a *future* period (Nov–Dec) and the December chart varies
nothing but the date. So we need to know exactly which time effects exist and whether they are
expressed through a column (market_index) or only through the calendar.

Strategy (simple → complex):
  1. volume over time                       → is the sampling stable?
  2. market_index as a time series          → per-date level, weekly cycle, seasonal level
  3. lane-normalised rate over time         → how much of its day-to-day movement does market_index explain?
  4. residual after market_index            → what's left that only the calendar can explain:
                                              quarter-end ramps, day-of-week, holidays, slow drift
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    AQUA,
    BLUE,
    GRID,
    INK_2,
    ORANGE,
    SPLIT_COLOR,
    US_HOLIDAYS_2025,
    Report,
    add_calendar,
    add_rate_features,
    daily_market_index,
    flag_outliers,
    load_raw,
    ols,
    relative_rate,
    set_style,
)

YELLOW = "#eda100"
QUARTER_COLOR = {1: BLUE, 2: ORANGE, 3: AQUA, 4: YELLOW}
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def shade_quarter_ends(ax, start, end, days: int = 21) -> None:
    for qe in pd.date_range(start, end + pd.Timedelta(days=90), freq="QE"):
        lo, hi = max(qe - pd.Timedelta(days=days - 1), start), min(qe, end)
        if lo <= hi:
            ax.axvspan(lo, hi + pd.Timedelta(hours=12), color=GRID, alpha=0.6, linewidth=0)


def main() -> None:
    set_style()
    train, val, _ = load_raw()
    tr = add_rate_features(train)
    rel, _ = relative_rate(tr)
    tr = tr.join(rel)
    tr["is_outlier"] = flag_outliers(tr.rel_rate)
    dmi = daily_market_index(train, val)
    allr = add_calendar(pd.concat([train.assign(split="train"), val.assign(split="validation")], ignore_index=True))
    c = add_calendar(tr[~tr.is_outlier]).merge(dmi, left_on="date", right_index=True)
    c["weight_abs"] = c.weight.abs().fillna(c.weight.abs().median())
    rep = Report(
        "05_temporal",
        "05 · Time effects",
        "Validation is Nov–Dec (future) and the December chart changes only the date, so every time effect matters. "
        "Rate analyses use clean train rows and `rel_rate` (lane- and equipment-normalised).",
    )

    # 1. volume -------------------------------------------------------------------------------------
    rep.h2("1. Volume over time")
    per_day = allr.groupby(["date", "split"]).size().rename("loads").reset_index()
    fig, ax = plt.subplots(figsize=(12, 3))
    for s, g in per_day.groupby("split"):
        ax.plot(g.date, g.loads, color=SPLIT_COLOR[s], linewidth=0.9, label=s)
    ax.set(title="Loads per day", ylabel="loads")
    ax.legend(loc="upper left")
    rep.figure(fig, "volume", "Validation days carry ~25% more loads than train days; otherwise volume is flat with no gaps.")
    vol = allr.groupby(["split", "dow_name"]).size() / allr.groupby(["split", "dow_name"]).date.nunique()
    rep.table(vol.unstack("dow_name")[DOW], digits=1)

    # 2. market_index time series --------------------------------------------------------------------
    rep.h2("2. market_index as a time series")
    daily = allr.groupby("date").agg(mi=("market_index", "mean"), split=("split", "first"))
    daily["trend"] = daily.mi.rolling(7, center=True, min_periods=4).mean()
    daily["weekly"] = daily.mi - daily.trend
    daily["dow"] = daily.index.dayofweek
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 3.8), gridspec_kw={"width_ratios": [3, 1]})
    for s, g in daily.groupby("split"):
        ax.plot(g.index, g.mi, color=SPLIT_COLOR[s], linewidth=1, label=f"daily mean ({s})")
    ax.plot(daily.index, daily.trend, color=INK_2, linewidth=1.8, label="7-day centred mean")
    ax.set(title="Daily mean market_index, Jan–Dec 2025", ylabel="market_index")
    ax.legend(loc="upper right")
    for s, g in daily.groupby("split"):
        prof = g.groupby("dow").weekly.mean()
        ax2.plot(DOW, prof.values, color=SPLIT_COLOR[s], marker="o", label=s)
    ax2.axhline(0, color=INK_2, linewidth=0.8)
    ax2.set(title="Weekly component (daily − 7-day mean)", ylabel="deviation")
    ax2.legend()
    fig.tight_layout()
    rep.figure(fig, "market_index_series", "market_index = a slow seasonal level (peak May–Jun) plus a strong weekly cycle with the same phase in train and validation.")
    mtab = daily.groupby(daily.index.month).agg(
        mean=("mi", "mean"),
        weekly_amplitude=("weekly", lambda s: s.groupby(daily.loc[s.index, "dow"]).mean().pipe(lambda p: p.max() - p.min())),
        peak_day=("weekly", lambda s: DOW[int(s.groupby(daily.loc[s.index, "dow"]).mean().idxmax())]),
        trough_day=("weekly", lambda s: DOW[int(s.groupby(daily.loc[s.index, "dow"]).mean().idxmin())]),
    )
    mtab.index.name = "month"
    rep.table(mtab, digits=3)

    # 3. rel_rate vs market_index over time ----------------------------------------------------------
    rep.h2("3. Does market_index explain the rate's movement over time?")
    y = np.log(c.rel_rate.values)
    beta, fit = ols(np.column_stack([np.log(c.market_index_daily), np.log(c.weight_abs)]), y)
    c["resid"] = y - fit
    c["log_rel"] = y - y.mean()
    d_rel = c.groupby("date").log_rel.mean()
    d_res = c.groupby("date").resid.mean()
    share = 1 - d_res.var() / d_rel.var()
    rep.p(
        f"Model: log(rel_rate) = a + {beta[1]:.3f}·log(daily market_index) + {beta[2]:.3f}·log(weight). "
        f"Daily market_index explains **{100 * share:.0f}%** of the day-to-day variance of the daily average rate."
    )
    m = c[c.market_index.notna()]
    dev = np.log(m.market_index / m.market_index_daily)
    b2, _ = ols(np.column_stack([np.log(m.market_index_daily), dev, np.log(m.weight_abs)]), np.log(m.rel_rate.values))
    rep.p(
        f"Per-load vs daily: splitting market_index into the day's mean and the load's deviation from it gives coefficients "
        f"{b2[1]:.3f} (daily level) and {b2[2]:.3f} (load deviation). If the deviation's coefficient is close to the "
        "daily one, the per-load value carries real information; if near 0, it is noise around the daily level."
    )
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)
    start, end = c.date.min(), c.date.max()
    axes[0].plot(d_rel.index, 100 * d_rel, color=BLUE, linewidth=0.8)
    axes[0].plot(d_rel.index, 100 * d_rel.rolling(7, center=True).mean(), color=INK_2, linewidth=1.6)
    axes[0].set(title="Daily average lane-normalised rate (% vs overall mean)", ylabel="%")
    axes[1].plot(daily.loc[start:end].index, daily.loc[start:end].mi, color=BLUE, linewidth=0.9)
    axes[1].set(title="Daily mean market_index", ylabel="index")
    axes[2].plot(d_res.index, 100 * d_res, color=BLUE, linewidth=0.8)
    axes[2].plot(d_res.index, 100 * d_res.rolling(7, center=True).mean(), color=INK_2, linewidth=1.6)
    axes[2].axhline(0, color=INK_2, linewidth=0.6)
    axes[2].set(title="What market_index does NOT explain: daily residual (%); shaded = last 21 days of each quarter", ylabel="%")
    for ax in axes:
        shade_quarter_ends(ax, start, end)
    fig.tight_layout()
    rep.figure(fig, "rate_market_residual", "Top: the rate moves with the market. Bottom: after removing market_index, a repeating ramp into each quarter end and a slow upward drift remain.")

    # 4. quarter-end profile ---------------------------------------------------------------------------
    rep.h2("4. Quarter-end effect")
    dres = c.groupby("date").agg(resid=("resid", "mean"), dtq=("days_to_quarter_end", "first"), doq=("day_of_quarter", "first"), q=("quarter", "first"))
    fig, ax = plt.subplots(figsize=(11, 4))
    for q, g in dres.groupby("q"):
        g = g.sort_values("dtq")
        ax.plot(g.dtq, 100 * (g.resid.rolling(3, center=True, min_periods=1).mean()), color=QUARTER_COLOR[q], marker="o", markersize=2.5, linewidth=1.3, label=f"Q{q}" + (" (Oct only)" if q == 4 else ""))
    ax.invert_xaxis()
    ax.axhline(0, color=INK_2, linewidth=0.6)
    ax.set(title="Residual rate by days remaining in the quarter (3-day smoothed)", xlabel="days to quarter end (→ quarter end)", ylabel="residual (%)")
    ax.legend(loc="upper left")
    rep.figure(fig, "quarter_end_profile", "Each completed quarter shows the same shape: flat, then a ramp over roughly the last three weeks, then a reset when the new quarter starts.")
    qrows = []
    for q, g in dres.groupby("q"):
        last14 = g[g.dtq <= 13].resid.mean()
        mid = g[(g.dtq >= 21) & (g.dtq <= 50)].resid.mean()
        first14 = g[g.doq <= 13].resid.mean()
        qrows.append({"quarter": f"Q{q}", "first 14 days %": 100 * first14, "days 21–50 before end %": 100 * mid, "last 14 days %": 100 * last14, "ramp (last14 − mid) %": 100 * (last14 - mid)})
    rep.table(pd.DataFrame(qrows), index=False, digits=2)
    jumps = []
    for qe in pd.to_datetime(["2025-03-31", "2025-06-30", "2025-09-30"]):
        before = d_res.loc[qe - pd.Timedelta(days=6): qe].mean()
        after = d_res.loc[qe + pd.Timedelta(days=1): qe + pd.Timedelta(days=7)].mean()
        jumps.append({"quarter end": qe, "last 7 days %": 100 * before, "first 7 days of next quarter %": 100 * after, "drop %": 100 * (after - before)})
    rep.table(pd.DataFrame(jumps), index=False, digits=2)
    q123 = dres[dres.q <= 3]
    base = q123[(q123.dtq >= 30) & (q123.dtq <= 60)].resid.mean()
    prof = (q123[q123.dtq <= 35].groupby("dtq").resid.mean() - base) * 100
    start_ramp = int(prof[prof.rolling(3, center=True, min_periods=1).mean() > 1.0].index.max()) if (prof > 1.0).any() else None
    rep.p(f"Averaged over Q1–Q3, relative to days 30–60 before the end, the residual first exceeds +1% (3-day smoothed) about **{start_ramp} days** before quarter end and peaks at +{prof.max():.1f}%.")

    # 5. day of week --------------------------------------------------------------------------------
    rep.h2("5. Day of week")
    dw = pd.DataFrame({"raw rate (% vs mean)": 100 * c.groupby("dow_name").log_rel.mean(), "after market_index (%)": 100 * c.groupby("dow_name").resid.mean()}).reindex(DOW)
    dw.index.name = "day"
    rep.table(dw, digits=2)
    rep.p(f"Raw weekday spread {dw.iloc[:, 0].max() - dw.iloc[:, 0].min():.2f} pts; after market_index {dw.iloc[:, 1].max() - dw.iloc[:, 1].min():.2f} pts. "
          "If the second is ~0, the weekly price cycle is carried entirely by market_index.")

    # 6. holidays -------------------------------------------------------------------------------------
    rep.h2("6. Holidays")
    hrows = []
    for name, day in US_HOLIDAYS_2025.items():
        dday = pd.Timestamp(day)
        if dday > end:
            continue
        win = d_res.loc[dday - pd.Timedelta(days=2): dday + pd.Timedelta(days=2)]
        ctx = d_res.loc[dday - pd.Timedelta(days=14): dday + pd.Timedelta(days=14)].drop(win.index, errors="ignore")
        hrows.append({"holiday": name, "date": dday, "day %": 100 * d_res.get(dday, np.nan), "±2 days %": 100 * win.mean(), "surrounding 4 weeks %": 100 * ctx.mean(), "difference %": 100 * (win.mean() - ctx.mean())})
    rep.table(pd.DataFrame(hrows), index=False, digits=2)
    rep.p(f"Daily residual standard deviation for reference: {100 * d_res.std():.2f}%. Differences well inside that are noise.")

    # 7. slow drift ----------------------------------------------------------------------------------
    rep.h2("7. Slow drift")
    mon = c.groupby("month").agg(n=("rpm", "size"), median_rpm=("rpm", "median"), market_index=("market_index_daily", "mean"), raw_rate_pct=("log_rel", "mean"), residual_pct=("resid", "mean"))
    mon[["raw_rate_pct", "residual_pct"]] *= 100
    rep.table(mon, digits=3)
    core = dres[(dres.dtq > 21) & (dres.doq > 7)]
    t = (core.index - core.index.min()).days.values.astype(float)
    sl_core = ols(t[:, None], core.resid.values)[0][1] * 30 * 100
    t_all = (dres.index - dres.index.min()).days.values.astype(float)
    sl_all = ols(t_all[:, None], dres.resid.values)[0][1] * 30 * 100
    rep.p(f"Linear drift of the residual: {sl_all:+.2f}% per 30 days using all days, {sl_core:+.2f}% per 30 days excluding quarter-end ramps and quarter starts. "
          "A drift that survives the exclusion is a genuine trend, which a tree model cannot extrapolate into Nov–Dec on its own.")
    rep.save()


if __name__ == "__main__":
    main()
