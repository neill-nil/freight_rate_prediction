"""06 · Train vs validation — will what we learn carry over? Are any features too good to be true?

Strategy:
  1. covariate shift per input (KS distance, ranges)       → does validation sit inside the training range?
  2. categorical shift: cities, lanes                        → how much of validation is genuinely new?
  3. feature–target stability month by month                 → a feature whose relationship flips sign over
                                                                time cannot be trusted for a future period
  4. quote_signal deep-dive                                  → leakage test, and a target-free proxy to see
                                                                what its relationship looks like in validation
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from common import (
    BLUE,
    DIV_CMAP,
    INK,
    INK_2,
    ORANGE,
    RED,
    SPLIT_COLOR,
    Report,
    add_rate_features,
    flag_outliers,
    load_raw,
    relative_rate,
    set_style,
)


def main() -> None:
    set_style()
    train, val, _ = load_raw()
    tr = add_rate_features(train)
    rel, _ = relative_rate(tr)
    tr = tr.join(rel)
    tr["is_outlier"] = flag_outliers(tr.rel_rate)
    c = tr[~tr.is_outlier].copy()
    rep = Report(
        "06_drift_and_leakage",
        "06 · Train vs validation: drift and leakage",
        "Validation (Nov–Dec) is a future period. A model only transfers if (a) validation inputs sit inside what it saw, "
        "and (b) each input's relationship with the rate is stable over time.",
    )

    # 1. numeric drift ---------------------------------------------------------------------------
    rep.h2("1. Numeric inputs: train vs validation")
    feats = {
        "distance": lambda d: d.distance,
        "weight (abs)": lambda d: d.weight.abs(),
        "market_index": lambda d: d.market_index,
        "quote_signal": lambda d: d.quote_signal,
        "pickup_lat": lambda d: d.pickup_lat,
        "pickup_lon": lambda d: d.pickup_lon,
    }
    rows = {}
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.5))
    for ax, (name, f) in zip(axes.flat, feats.items()):
        a, b = f(train).dropna(), f(val).dropna()
        rows[name] = {
            "train mean": a.mean(), "val mean": b.mean(), "train std": a.std(), "val std": b.std(),
            "train p1–p99": f"{a.quantile(0.01):.3g}–{a.quantile(0.99):.3g}", "val p1–p99": f"{b.quantile(0.01):.3g}–{b.quantile(0.99):.3g}",
            "val share inside train min–max %": 100 * b.between(a.min(), a.max()).mean(),
            "KS distance": ks_2samp(a, b).statistic,
        }
        bins = np.histogram_bin_edges(pd.concat([a, b]), 60)
        ax.hist(a, bins=bins, density=True, color=SPLIT_COLOR["train"], alpha=0.5, label="train")
        ax.hist(b, bins=bins, density=True, histtype="step", linewidth=1.8, color=SPLIT_COLOR["validation"], label="validation")
        ax.set(title=name, yticks=[])
    axes[0, 0].legend()
    fig.tight_layout()
    rep.figure(fig, "numeric_drift", "Distance, weight and geography are stable. market_index and quote_signal are narrower in validation (Nov–Dec is a quiet, single-season slice).")
    rep.table(pd.DataFrame(rows).T, digits=3)
    rep.p("KS distance: 0 = identical distributions, 1 = no overlap. Values above ~0.1 indicate a shift worth checking.")
    mi_month = pd.concat([train, val]).groupby(pd.concat([train, val]).date.dt.month).market_index.agg(["mean", "min", "max"])
    mi_month.index.name = "month"
    rep.p("market_index by month: validation's level is low but matches Jan, Aug–Oct, which are in training.")
    rep.table(mi_month, digits=3)

    # 2. categorical drift ------------------------------------------------------------------------
    rep.h2("2. Cities and lanes")
    tr_c = set(train.pickup) | set(train.delivery)
    new_c = sorted((set(val.pickup) | set(val.delivery)) - tr_c)
    lane_n = train.groupby(["pickup", "delivery"]).size()
    vlane = pd.MultiIndex.from_frame(val[["pickup", "delivery"]])
    seen_n = lane_n.reindex(vlane).fillna(0).values
    touch_new = val.pickup.isin(new_c) | val.delivery.isin(new_c)
    cov = pd.DataFrame(
        {
            "share of validation rows %": [
                100 * touch_new.mean(),
                100 * (seen_n == 0).mean(),
                100 * ((seen_n == 0) & ~touch_new).mean(),
                100 * ((seen_n >= 1) & (seen_n < 5)).mean(),
                100 * (seen_n >= 5).mean(),
            ]
        },
        index=pd.Index(
            ["touch a city never seen in train", "lane never seen in train", "  …of which both cities are known", "lane seen 1–4 times", "lane seen ≥ 5 times"],
            name="validation coverage",
        ),
    )
    rep.table(cov, digits=1)
    nc = pd.DataFrame({"as pickup": val.pickup.value_counts().reindex(new_c), "as delivery": val.delivery.value_counts().reindex(new_c)})
    nc.index.name = "new city"
    rep.table(nc)
    eq = pd.DataFrame({n: d.equipment.value_counts(normalize=True) * 100 for n, d in (("train", train), ("validation", val))})
    rep.p(f"Equipment mix is unchanged: {', '.join(f'{e} {r.train:.1f}% → {r.validation:.1f}%' for e, r in eq.iterrows())}.")

    # 3. feature-target stability by month ------------------------------------------------------------
    rep.h2("3. Is each feature's relationship with the rate stable over time?")
    cands = {
        "log distance": np.log(c.distance),
        "weight (abs)": c.weight.abs(),
        "market_index": c.market_index,
        "quote_signal": c.quote_signal,
        "pickup_lat": c.pickup_lat,
        "pickup_lon": c.pickup_lon,
    }
    month = c.date.dt.month
    heat = {}
    for target in ("rpm", "rel_rate"):
        heat[target] = pd.DataFrame(
            {m: {n: s[month == m].corr(c[target][month == m], method="spearman") for n, s in cands.items()} for m in sorted(month.unique())}
        )
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.6), sharey=True)
    for ax, (target, h) in zip(axes, heat.items()):
        im = ax.imshow(h.values, cmap=DIV_CMAP.reversed(), vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(h.shape[1]), [pd.Timestamp(2025, m, 1).strftime("%b") for m in h.columns])
        ax.set_yticks(range(h.shape[0]), h.index)
        ax.tick_params(axis="y", labelleft=ax is axes[0])
        ax.grid(False)
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                v = h.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if abs(v) > 0.55 else INK)
        ax.set_title(f"Spearman correlation with {'rate per mile' if target == 'rpm' else 'lane-normalised rate'}, by month")
    fig.colorbar(im, ax=axes, shrink=0.9, label="correlation")
    rep.figure(fig, "stability_heatmap", "Every feature keeps its sign month after month, except quote_signal, which swings between strongly positive and strongly negative.")
    sd = pd.DataFrame({t: h.std(axis=1) for t, h in heat.items()})
    sd["sign flips (rpm)"] = heat["rpm"].apply(lambda r: int((np.sign(r).diff().fillna(0) != 0).sum()), axis=1)
    sd.columns = ["std of monthly corr (rpm)", "std of monthly corr (rel_rate)", "sign flips (rpm)"]
    rep.table(sd, digits=3)

    # 4. quote_signal deep-dive ------------------------------------------------------------------------
    rep.h2("4. quote_signal deep-dive")
    wk = c.groupby(c.date.dt.to_period("W")).apply(lambda d: d.quote_signal.corr(d.rpm))
    fig, ax = plt.subplots(figsize=(12, 3.4))
    x = wk.index.start_time
    ax.bar(x, wk.values, width=5, color=[BLUE if v >= 0 else RED for v in wk.values])
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set(title="Weekly correlation between quote_signal and rate per mile (blue positive, red negative)", ylabel="Pearson r", ylim=(-1.05, 1.05))
    rep.figure(fig, "quote_signal_weekly", "Within a regime quote_signal is almost a copy of the target (|r| up to 1.0); the sign flips at regime boundaries and it carries nothing in August.")

    months = [3, 4, 6, 7, 8, 9]
    fig, axes = plt.subplots(1, len(months), figsize=(15, 2.9), sharey=True)
    rng = np.random.default_rng(0)
    for ax, m in zip(axes, months):
        d = c[month == m]
        d = d.iloc[rng.choice(len(d), min(1500, len(d)), replace=False)]
        ax.scatter(d.quote_signal, d.rpm, s=3, alpha=0.35, color=BLUE, linewidths=0)
        ax.set(title=pd.Timestamp(2025, m, 1).strftime("%B"), xlabel="quote_signal")
    axes[0].set_ylabel("rate per mile")
    fig.tight_layout()
    rep.figure(fig, "quote_signal_scatter", "quote_signal vs rate per mile in different months: rising, falling, rising, falling, flat, rising.")

    rep.h3("What does quote_signal do in validation? A target-free proxy")
    rep.p("We cannot correlate with the target in validation, but in train, wherever quote_signal tracks the rate it must also track "
          "log distance (the dominant driver of rate per mile). So corr(quote_signal, log distance) is a proxy for its "
          "relationship with the target that we *can* compute in validation.")
    prox = pd.concat(
        [
            pd.DataFrame({"corr(quote_signal, log distance)": train.groupby(train.date.dt.month).apply(lambda d: d.quote_signal.corr(np.log(d.distance))),
                          "corr(quote_signal, rate per mile)": tr.groupby(tr.date.dt.month).apply(lambda d: d[~d.is_outlier].quote_signal.corr(d[~d.is_outlier].rpm)), "split": "train"}),
            pd.DataFrame({"corr(quote_signal, log distance)": val.groupby(val.date.dt.month).apply(lambda d: d.quote_signal.corr(np.log(d.distance))),
                          "corr(quote_signal, rate per mile)": np.nan, "split": "validation"}),
        ]
    )
    prox.index.name = "month"
    rep.table(prox, digits=3)
    rep.p("In train the proxy mirrors the target correlation month by month (with the opposite sign, because rate per mile falls with distance). "
          "In validation the proxy is ≈ 0, like August: **quote_signal carries no information about Nov–Dec rates.** "
          "A model that leans on it would score well on a random split of train and then fail on validation.")
    q = c.groupby(c.date.dt.month).quote_signal.agg(["mean", "std"])
    qv = val.groupby(val.date.dt.month).quote_signal.agg(["mean", "std"])
    rep.p(f"Its own distribution barely moves (train monthly std {q['std'].min():.2f}–{q['std'].max():.2f}, validation {qv['std'].min():.2f}–{qv['std'].max():.2f}), "
          "so a simple drift check would *not* catch it. Only the relationship changes.")
    _ = ORANGE
    rep.save()


if __name__ == "__main__":
    main()
