"""02 · Univariate distributions (train) — what does each column look like on its own?

Strategy: look at the target on linear and log scales (is the error structure additive or
multiplicative?), normalise it by distance (rate per mile) to see the price level, and look at
every input for skew, spikes, impossible values and caps. This decides the target transform and
which columns need cleaning.
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
    WEIGHT_CAP,
    Report,
    add_rate_features,
    load_raw,
    set_style,
)

QS = [0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1]


def logbins(s: pd.Series, n: int = 120) -> np.ndarray:
    return np.logspace(np.log10(s.min()), np.log10(s.max()), n)


def main() -> None:
    set_style()
    train, _, _ = load_raw()
    tr = add_rate_features(train)
    rep = Report(
        "02_univariate",
        "02 · Univariate distributions (train)",
        "Goal: understand each column on its own before relating them. Train set only; "
        "train-vs-validation comparisons live in step 06.",
    )

    # target ------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    axes[0].hist(tr.posted_rate, bins=150, color=BLUE)
    axes[0].set(title="posted_rate", xlabel="$ per load", ylabel="loads")
    axes[1].hist(tr.posted_rate, bins=logbins(tr.posted_rate), color=BLUE)
    axes[1].set(title="posted_rate (log x)", xscale="log", xlabel="$ per load")
    axes[2].hist(tr.rpm, bins=logbins(tr.rpm), color=BLUE)
    axes[2].set(title="rate per mile (log x, log y)", xscale="log", yscale="log", xlabel="$ per mile")
    rep.h2("1. Target")
    rep.figure(
        fig,
        "target",
        "Left: raw rate is right-skewed with a long tail. Middle: on a log axis it is close to "
        "symmetric → multiplicative structure. Right: rate per mile is tightly concentrated, with "
        "isolated tails far from the bulk (log y makes the few hundred tail rows visible).",
    )
    qt = pd.DataFrame({"posted_rate ($)": tr.posted_rate.quantile(QS), "rate per mile ($)": tr.rpm.quantile(QS)})
    qt.index = [f"p{q * 100:g}" for q in QS]
    qt.index.name = "quantile"
    rep.table(qt, digits=3)
    rep.bullets(
        [
            f"Skewness: posted_rate {tr.posted_rate.skew():.2f}, log(posted_rate) {tr.log_rate.skew():.2f}, "
            f"rate per mile {tr.rpm.skew():.2f}.",
            f"Rate per mile p1–p99 is {tr.rpm.quantile(0.01):.2f}–{tr.rpm.quantile(0.99):.2f}, but min/max are "
            f"{tr.rpm.min():.2f}/{tr.rpm.max():.2f}: far tails that look like corrupted labels, not market moves.",
        ]
    )

    # inputs -------------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7))
    ax = axes[0, 0]
    ax.hist(tr.distance, bins=100, color=BLUE)
    ax.set(title="distance", xlabel="miles", ylabel="loads")

    ax = axes[0, 1]
    bins = np.linspace(-WEIGHT_CAP, WEIGHT_CAP, 191)
    w = tr.weight.dropna()
    ax.hist(w[w >= 0], bins=bins, color=BLUE, label="weight ≥ 0")
    ax.hist(w[w < 0], bins=bins, color=ORANGE, label="weight < 0")
    ax.set(title="weight (raw)", xlabel="lb", yscale="log")
    ax.annotate(
        f"{(tr.weight == WEIGHT_CAP).sum():,} rows exactly at {WEIGHT_CAP:,.0f}",
        xy=(WEIGHT_CAP, (tr.weight == WEIGHT_CAP).sum()),
        xytext=(-2_000, 3_000),
        fontsize=7.5,
        color=INK_2,
        ha="right",
        arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 0.6},
    )
    ax.legend(loc="upper left")

    ax = axes[0, 2]
    ax.hist(tr.market_index.dropna(), bins=80, color=BLUE)
    ax.set(title="market_index", xlabel="index")

    ax = axes[1, 0]
    ax.hist(tr.quote_signal, bins=80, color=BLUE)
    ax.set(title="quote_signal", xlabel="value", ylabel="loads")

    ax = axes[1, 1]
    share = tr.equipment.value_counts(normalize=True).reindex(EQUIPMENT) * 100
    ax.barh(share.index[::-1], share.values[::-1], color=[EQUIP_COLOR[e] for e in share.index[::-1]], height=0.55)
    for i, (e, v) in enumerate(share[::-1].items()):
        ax.text(v + 0.8, i, f"{v:.1f}%", va="center", fontsize=8, color=INK_2)
    ax.set(title="equipment share", xlabel="% of loads", xlim=(0, 70))
    ax.grid(axis="y", visible=False)

    ax = axes[1, 2]
    per_day = tr.groupby("date").size()
    ax.plot(per_day.index, per_day.values, color=BLUE, linewidth=0.9)
    ax.plot(per_day.index, per_day.rolling(7, center=True).mean(), color=INK_2, linewidth=1.6, label="7-day mean")
    ax.set(title="loads per day", ylabel="loads")
    ax.legend(loc="lower right")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    rep.h2("2. Inputs")
    rep.figure(
        fig,
        "inputs",
        "Distance is spread from 70 to 3,400 miles. Weight has a negative mirror image (orange) and "
        "a spike at the 47,500 lb cap. market_index is multi-modal (a time-driven series, see step 05). "
        "quote_signal looks like a rate-per-mile number. Equipment is imbalanced toward Dry Van.",
    )
    summary = {}
    for c in ["distance", "weight", "market_index", "quote_signal", "posted_rate", "rpm"]:
        s = tr[c]
        summary[c] = {
            "missing": s.isna().sum(),
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "p1": s.quantile(0.01),
            "median": s.median(),
            "p99": s.quantile(0.99),
            "max": s.max(),
            "skew": s.skew(),
        }
    rep.table(pd.DataFrame(summary).T, digits=3)

    # cities --------------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 6.5), sharey=True)
    for ax, role in zip(axes, ["pickup", "delivery"]):
        vc = tr[role].value_counts()
        ax.bar(vc.index, vc.values, color=BLUE, width=0.7)
        ax.axhline(vc.mean(), color=INK_2, linewidth=0.8)
        ax.set(title=f"loads per {role} city", ylabel="loads")
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.grid(axis="x", visible=False)
        ax.margins(x=0.005)
    fig.tight_layout()
    rep.h2("3. Cities")
    rep.figure(fig, "cities", "Loads per city (line = mean). Volume is fairly even across the 64 train cities.")
    for role in ("pickup", "delivery"):
        vc = tr[role].value_counts()
        rep.p(f"{role}: min {vc.min()} ({vc.idxmin()}), max {vc.max()} ({vc.idxmax()}), coefficient of variation {vc.std() / vc.mean():.2f}.")
    rep.save()


if __name__ == "__main__":
    main()
