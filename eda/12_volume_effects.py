"""12 · Deep-dive: does price depend on how many loads there are?

Freight intuition says busy lanes/cities/days should be pricier (more demand for the same trucks), and an
imbalance (many loads leaving a city, few arriving) should push outbound prices up. The data don't say this
explicitly, but we can count loads ourselves. Also relevant because validation has ~25% more loads per day
than train: if volume mattered, that would shift prices.

Strategy:
  1. build volume counts per load: day, equipment-day, city-day, city ±3 days, inbound/outbound imbalance,
     city-month, lane-month
  2. compare them with the price residual after the structural model (09) + time model (10), i.e. what is
     still unexplained. City-level counts vary within a day, so they are compared with the within-day
     residual; day-level counts with the daily residual.
  3. effect sizes, a joint test, and a random-noise baseline for what "no effect" looks like
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import (
    BLUE,
    INK_2,
    ORANGE,
    SPLIT_COLOR,
    Report,
    add_calendar,
    clean_train,
    daily_market_index,
    fit_structural,
    load_raw,
    ols,
    set_style,
)


def roll_city(df: pd.DataFrame, col: str) -> pd.Series:
    cnt = df.groupby([col, "date"]).size().unstack(col, fill_value=0)
    cnt = cnt.reindex(pd.date_range(df.date.min(), df.date.max()), fill_value=0)
    return cnt.rolling(7, center=True, min_periods=1).sum().stack()


def lookup(s: pd.Series, dates: pd.Series, cities: pd.Series) -> np.ndarray:
    return s.reindex(pd.MultiIndex.from_arrays([dates.values, cities.values])).fillna(0).values


def volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Counts computed over ALL loads of the split (including rows later dropped as corrupted labels)."""
    g = df.groupby
    out = pd.DataFrame(index=df.index)
    out["loads that day"] = g("date").load_id.transform("size")
    out["same-equipment loads that day"] = g(["date", "equipment"]).load_id.transform("size")
    out["equipment share of the day"] = out["same-equipment loads that day"] / out["loads that day"]
    out["pickup city: outbound loads that day"] = g(["date", "pickup"]).load_id.transform("size")
    out["delivery city: inbound loads that day"] = g(["date", "delivery"]).load_id.transform("size")
    out_r, in_r = roll_city(df, "pickup"), roll_city(df, "delivery")
    po, pi = lookup(out_r, df.date, df.pickup), lookup(in_r, df.date, df.pickup)
    di, do = lookup(in_r, df.date, df.delivery), lookup(out_r, df.date, df.delivery)
    out["pickup city: outbound, ±3 days"] = po
    out["pickup city: imbalance out − in, ±3 days"] = po - pi
    out["delivery city: inbound, ±3 days"] = di
    out["delivery city: imbalance in − out, ±3 days"] = di - do
    month = df.date.dt.month
    out["pickup city: outbound that month"] = df.groupby([month, df.pickup]).load_id.transform("size")
    out["lane: loads that month"] = df.groupby([month, df.pickup, df.delivery]).load_id.transform("size")
    return out


DAY_LEVEL = ["loads that day"]


def main() -> None:
    set_style()
    train, val, _ = load_raw()
    c, full = clean_train(train, val)
    vf_all = volume_features(train)
    vf = vf_all.loc[c.index]
    rep = Report(
        "12_volume_effects",
        "12 · Deep-dive: does load volume affect price?",
        "Volume counts are computed from all loads in the file. The price signal tested is the residual after the refined structural model "
        "(09) and the drift + quarter-end ramp (10): whatever distance, weight, market, equipment, city and time do not explain.",
    )

    # residual after structure + time model
    ref = fit_structural(c, "shared")
    cal = add_calendar(c)
    daily = c.assign(r=ref["resid"]).groupby("date").r.mean()
    t = (daily.index - pd.Timestamp("2025-01-01")).days.values / 30
    dtq = cal.groupby("date").days_to_quarter_end.first().reindex(daily.index).values
    _, fit_t = ols(np.column_stack([t, np.clip(1 - dtq / 28, 0, None)]), daily.values)
    resid = ref["resid"] - pd.Series(fit_t, index=daily.index).reindex(c.date).values
    within = resid - resid.groupby(c.date).transform("mean")
    daily_left = resid.groupby(c.date).mean()

    # 1. what the counts look like -------------------------------------------------------------------------
    rep.h2("1. The volume counts")
    vv = volume_features(val)
    desc = pd.DataFrame({"train median": vf_all.median(), "train p10–p90": vf_all.quantile(0.1).round(2).astype(str) + "–" + vf_all.quantile(0.9).round(2).astype(str),
                         "validation median": vv.median()})
    desc.index.name = "count per load"
    rep.table(desc, digits=2)
    rep.p("Validation has more loads per day (sampling), so raw counts are higher there. If any count mattered we would need a "
          "normalised version (e.g. a share of the day's loads). First: does any of them matter at all?")

    # 2. correlations and effect sizes ------------------------------------------------------------------------
    rep.h2("2. Is there any relationship with the unexplained price?")
    rng = np.random.default_rng(0)
    null = [abs(spearmanr(rng.permutation(within.values), vf.iloc[:, 3].values)[0]) for _ in range(200)]
    rows = []
    for col in vf.columns:
        if col in DAY_LEVEL:
            x = vf[col].groupby(c.date).first().reindex(daily_left.index)
            yv, n_ = daily_left, len(daily_left)
        else:
            x, yv, n_ = vf[col], within, len(within)
        rho, p = spearmanr(x.values, yv.values)
        b, _ = ols(((x - x.mean()) / x.std()).values[:, None], yv.values)
        rows.append({"volume measure": col, "compared with": "daily residual" if col in DAY_LEVEL else "within-day residual", "n": n_,
                     "Spearman ρ": rho, "p-value": p, "price change per +1 std of volume (%)": 100 * b[1]})
    tab = pd.DataFrame(rows)
    rep.table(tab, index=False, digits=4)
    rep.p(f"Baseline for 'no effect': correlating the residual with randomly shuffled values gives |ρ| up to {np.percentile(null, 95):.4f} "
          f"(95th percentile of 200 shuffles, n = {len(within):,}). Correlations of that size are noise.")

    X = vf.drop(columns=DAY_LEVEL)
    Xs = ((X - X.mean()) / X.std()).values
    _, f_ = ols(Xs, within.values)
    gain_bp = 1e4 * (within.std() - (within.values - f_).std())
    rep.p(f"Joint test: all load-level volume measures together explain {100 * (1 - np.var(within.values - f_) / np.var(within.values)):.3f}% of the "
          f"within-day residual variance (residual std improves by {gain_bp:.2f} basis points).")

    # 3. pictures ------------------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    picks = ["pickup city: outbound, ±3 days", "pickup city: imbalance out − in, ±3 days", "delivery city: inbound, ±3 days", "same-equipment loads that day"]
    for ax, col in zip(axes, picks):
        x = vf[col]
        bins = np.unique(np.quantile(x, np.linspace(0, 1, 11)))
        b = pd.DataFrame({"x": x.values, "y": 100 * within.values}).groupby(pd.cut(x.values, bins, include_lowest=True), observed=True).agg(x=("x", "median"), m=("y", "mean"), se=("y", lambda s: s.std() / np.sqrt(len(s))))
        ax.errorbar(b.x, b.m, yerr=1.96 * b.se, color=BLUE, marker="o", markersize=4, capsize=2, linewidth=1.4)
        ax.axhline(0, color=INK_2, linewidth=0.8)
        ax.set(title=col.replace("pickup city: ", "pickup: ").replace("delivery city: ", "delivery: "), xlabel="count")
        ax.title.set_fontsize(9)
    axes[0].set_ylabel("unexplained price (%, mean ± 95% CI)")
    fig.tight_layout()
    rep.figure(fig, "volume_bins", "Unexplained price by volume decile. Flat lines inside ±0.1% mean busier cities/days are not priced differently.")
    dl = pd.DataFrame({"loads": vf["loads that day"].groupby(c.date).first(), "resid": 100 * daily_left})
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.scatter(dl.loads, dl.resid, s=14, color=BLUE, alpha=0.7, linewidths=0)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set(title="Daily: total loads vs unexplained price", xlabel="loads that day", ylabel="daily residual (%)")
    rep.figure(fig, "daily_volume", "Each dot is one day (Jan–Oct).")

    # 4. does volume drive market_index? ------------------------------------------------------------------------
    rep.h2("3. Is market_index just volume in disguise?")
    dmi = daily_market_index(train)
    dvol = train.groupby("date").size()
    rho_mi = spearmanr(dmi.reindex(dvol.index).values, dvol.values)[0]
    vol_dow = train.groupby(train.date.dt.dayofweek).size() / train.groupby(train.date.dt.dayofweek).date.nunique()
    rep.p(f"Daily load count vs daily market_index: Spearman ρ = {rho_mi:.3f}. Loads per weekday: "
          + ", ".join(f"{d} {v:.0f}" for d, v in zip(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], vol_dow.values))
          + ". Volume is flat across the week while market_index swings Thu-high / Sun-low, so the market index is not a volume count.")

    rep.h2("Conclusion")
    big = tab.loc[tab["price change per +1 std of volume (%)"].abs().idxmax()]
    rep.bullets(
        [
            f"The largest effect of any volume measure is {big['price change per +1 std of volume (%)']:+.3f}% per standard deviation "
            f"({big['volume measure']}). Next to ~1.8% per-load noise that is negligible.",
            "No sign that busy days, busy cities, trailer-type demand, directional imbalance or lane popularity move prices.",
            "**Decision: no volume features.** This also means validation's higher daily load count is a sampling difference, not a market signal.",
        ]
    )
    _ = (ORANGE, SPLIT_COLOR)
    rep.save()


if __name__ == "__main__":
    main()
