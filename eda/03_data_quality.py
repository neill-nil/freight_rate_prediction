"""03 · Data quality — for every anomaly: how big is it, what is it, and what is the evidence?

Strategy: an anomaly is only "fixed" once we know its mechanism. For each issue we test a
hypothesis with data rather than assume:
  * negative weights   → sign flip?        (|neg| distribution vs positives, label behaviour)
  * weight == 47,500   → censoring cap?    (spike + label behaviour of capped rows)
  * missing values     → random?           (label behaviour, time pattern)
  * market_index       → a per-date value? (within-date vs between-date variance → imputation)
  * coordinates        → trustworthy?      (clipping, distance vs straight-line consistency)
  * label outliers     → corrupted labels? (gap in relative-rate distribution, concentration tests)
The reference for "is this label normal?" is rel_rate: rate per mile divided by the typical rate
per mile for the same lane and equipment (see common.relative_rate).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from scipy.stats import norm

from common import (
    BLUE,
    INK_2,
    ORANGE,
    OUTLIER_HI,
    OUTLIER_LO,
    SEQ_CMAP,
    WEIGHT_CAP,
    Report,
    add_rate_features,
    city_coords,
    error_summary,
    flag_outliers,
    haversine_miles,
    load_raw,
    relative_rate,
    set_style,
)


def weight_status(w: pd.Series) -> pd.Series:
    return pd.Series(
        np.select([w.isna(), w < 0, w == WEIGHT_CAP], ["missing", "negative", "capped"], "normal"),
        index=w.index,
    )


def label_behaviour(df: pd.DataFrame, by: pd.Series) -> pd.DataFrame:
    g = df.groupby(by)
    return pd.DataFrame(
        {
            "rows": g.size(),
            "median rel_rate": g.rel_rate.median(),
            "rel_rate IQR": g.rel_rate.quantile(0.75) - g.rel_rate.quantile(0.25),
            "outlier %": 100 * g.is_outlier.mean(),
        }
    )


def main() -> None:
    set_style()
    train, val, _ = load_raw()
    tr = add_rate_features(train)
    rel, premium = relative_rate(tr)
    tr = tr.join(rel)
    tr["is_outlier"] = flag_outliers(tr.rel_rate)
    rep = Report(
        "03_data_quality",
        "03 · Data quality deep-dive",
        "Each issue gets a size, a mechanism hypothesis and evidence. `rel_rate` = rate per mile ÷ "
        f"typical rate per mile for the same lane and equipment (equipment premiums used: "
        f"{', '.join(f'{k} {v:.3f}' for k, v in premium.items())}). rel_rate ≈ 1 means a normal label.",
    )
    summary = []

    # ===================================================================== A. weight
    rep.h2("A. Weight")
    counts = pd.DataFrame(
        {
            n: {
                "rows": len(d),
                "missing": d.weight.isna().sum(),
                "negative": (d.weight < 0).sum(),
                f"exactly {WEIGHT_CAP:,.0f}": (d.weight == WEIGHT_CAP).sum(),
                f"exactly -{WEIGHT_CAP:,.0f}": (d.weight == -WEIGHT_CAP).sum(),
            }
            for n, d in (("train", tr), ("validation", val))
        }
    )
    rep.table(counts)

    rep.h3("A1. Negative weights: sign flip?")
    neg = tr.weight[tr.weight < 0].abs()
    pos = tr.weight[tr.weight > 0]
    ks = ks_2samp(neg, pos)
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    qtab = pd.DataFrame({"|negative weights|": neg.quantile(qs), "positive weights": pos.quantile(qs)})
    qtab.index = [f"p{int(q * 100)}" for q in qs]
    rep.table(qtab)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bins = np.linspace(0, WEIGHT_CAP, 60)
    ax.hist(pos, bins=bins, density=True, color=BLUE, alpha=0.55, label=f"positive (n={len(pos):,})")
    ax.hist(neg, bins=bins, density=True, histtype="step", color=ORANGE, linewidth=2, label=f"|negative| (n={len(neg):,})")
    ax.set(title="Absolute value of negative weights vs positive weights", xlabel="lb", ylabel="density")
    ax.legend()
    rep.figure(fig, "weight_signflip", "If negatives are sign flips, |negative| should have the same shape as positives.")
    rep.p(f"KS test |negative| vs positive: D = {ks.statistic:.3f}, p = {ks.pvalue:.2f} (no evidence the distributions differ).")

    rep.h3("A2. Label behaviour by weight status")
    rep.p("If the weight field is broken but the load is normal, its label should look like any other load's.")
    rep.table(label_behaviour(tr, weight_status(tr.weight)))

    rep.h3("A3. Pile-ups at both ends: is weight clipped?")
    w = tr.weight.abs()
    floor_ = w.min()
    inner = w[(w > floor_) & (w < WEIGHT_CAP)]
    med, sd = inner.median(), (inner.quantile(0.75) - inner.quantile(0.25)) / 1.349  # robust normal fit
    clip_tab = pd.DataFrame(
        {
            "rows exactly at value": [(w == floor_).sum(), (w == WEIGHT_CAP).sum()],
            "observed share %": [100 * (w == floor_).mean(), 100 * (w == WEIGHT_CAP).mean()],
            "normal-fit tail beyond value %": [100 * norm.cdf(floor_, med, sd), 100 * norm.sf(WEIGHT_CAP, med, sd)],
            "next distinct value": [np.sort(w.dropna().unique())[1], np.sort(w.dropna().unique())[-2]],
        },
        index=pd.Index([f"floor {floor_:,.0f}", f"cap {WEIGHT_CAP:,.0f}"], name="end"),
    )
    rep.table(clip_tab, digits=2)
    rep.p(
        f"A robust normal fit to the unclipped weights (median {med:,.0f}, sd {sd:,.0f}) predicts roughly the tail mass "
        "that piles up at each end, so weight was **clipped to [5,000, 47,500]**: a capped row's true weight is "
        "'at least 47,500'. Do capped rows price like the heaviest loads?"
    )
    near = pd.cut(w, [30_000, 35_000, 40_000, 42_500, 45_000, WEIGHT_CAP - 1, WEIGHT_CAP])
    keep = w.between(30_000, WEIGHT_CAP)
    rep.table(label_behaviour(tr[keep], near[keep].astype(str)))
    rep.p("The weight effect flattens above ~42,500 lb, so the cap costs little information.")

    rep.h3("A4. Missing weight: random?")
    miss_month = tr.groupby(tr.date.dt.month).weight.apply(lambda s: 100 * s.isna().mean())
    rep.p(
        f"Missing-weight share by month ranges {miss_month.min():.2f}%–{miss_month.max():.2f}%; "
        "label behaviour of missing rows is in the A2 table."
    )
    summary += [
        {"issue": "negative weight", "train": (tr.weight < 0).sum(), "validation": (val.weight < 0).sum(),
         "evidence": f"|neg| ≈ positives (KS D={ks.statistic:.3f}); labels normal", "candidate handling": "take abs()"},
        {"issue": "weight clipped to [5,000, 47,500]", "train": tr.weight.abs().isin([floor_, WEIGHT_CAP]).sum(),
         "validation": val.weight.abs().isin([floor_, WEIGHT_CAP]).sum(), "evidence": "pile-ups ≈ normal-fit tail mass",
         "candidate handling": "keep value (effect flat at the top); optional is_capped flag"},
        {"issue": "missing weight", "train": tr.weight.isna().sum(), "validation": val.weight.isna().sum(),
         "evidence": "labels normal, no time pattern", "candidate handling": "impute (median per equipment) + flag"},
    ]

    # ===================================================================== B. market_index
    rep.h2("B. market_index: a per-date value?")
    rows = {}
    for n, d in (("train", tr), ("validation", val)):
        g = d.groupby("date").market_index
        total_var = d.market_index.var()
        within_var = (d.market_index - g.transform("mean")).var()
        loo = (g.transform("sum") - d.market_index) / (g.transform("count") - 1)
        rows[n] = {
            "mean within-date std": g.std().mean(),
            "std of daily means": g.mean().std(),
            "share of variance explained by date": 1 - within_var / total_var,
            "leave-one-out daily-mean imputation MAE": (loo - d.market_index).abs().mean(),
            "missing": d.market_index.isna().sum(),
        }
    rep.table(pd.DataFrame(rows), digits=4)
    may = tr[tr.date.dt.month == 5]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    jitter = np.random.default_rng(0).uniform(-0.3, 0.3, len(may))
    ax.scatter(may.date + pd.to_timedelta(jitter, unit="D"), may.market_index, s=3, color=BLUE, alpha=0.35, linewidths=0, label="individual loads")
    dm = may.groupby("date").market_index.mean()
    ax.plot(dm.index, dm.values, color=ORANGE, linewidth=2, label="daily mean")
    ax.set(title="market_index for each load, May 2025", ylabel="market_index")
    ax.legend(loc="lower left")
    rep.figure(fig, "market_index_per_date", "Loads on the same day share almost the same market_index; the daily mean carries the signal.")
    rep.table(label_behaviour(tr, tr.market_index.isna().map({True: "missing", False: "present"})))
    summary.append(
        {"issue": "missing market_index", "train": tr.market_index.isna().sum(), "validation": val.market_index.isna().sum(),
         "evidence": f"date explains {rows['train']['share of variance explained by date']:.0%} of its variance",
         "candidate handling": "impute with same-date mean"}
    )

    # ===================================================================== C. coordinates & distance
    rep.h2("C. Coordinates and distance")
    cc = city_coords(tr, val).drop_duplicates("city").set_index("city")
    train_cities = set(tr.pickup) | set(tr.delivery)
    clipped = cc[(cc.lon == cc.lon.max()) | (cc.lat == cc.lat.min())]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    is_new = ~cc.index.isin(train_cities)
    ax.scatter(cc.lon[~is_new], cc.lat[~is_new], s=28, color=BLUE, label="city in train", zorder=3, edgecolors="white", linewidths=1)
    ax.scatter(cc.lon[is_new], cc.lat[is_new], s=34, color=ORANGE, label="city only in validation", zorder=3, edgecolors="white", linewidths=1)
    for city, r in cc.iterrows():
        ax.annotate(city, (r.lon, r.lat), xytext=(3, 3), textcoords="offset points", fontsize=6, color=INK_2)
    for city, r in clipped.iterrows():
        ax.scatter(r.lon, r.lat, s=140, facecolors="none", edgecolors=INK_2, linewidths=1, zorder=2)
    ax.set(title="City coordinates as given in the data (rings = sitting on a bounding-box edge)", xlabel="longitude", ylabel="latitude")
    ax.legend(loc="lower left")
    rep.figure(fig, "city_map", "Coordinates are only roughly geographic (e.g. Boston/Providence pinned at lon −69.5, Laredo at lat 25.5).")
    rep.p("Cities sitting exactly on an extreme coordinate:")
    rep.table(clipped, digits=5)

    allr = pd.concat([tr.assign(split="train"), val.assign(split="validation")], ignore_index=True)
    allr["straight_line"] = haversine_miles(allr.pickup_lat, allr.pickup_lon, allr.delivery_lat, allr.delivery_lon)
    allr["circuity"] = allr.distance / allr.straight_line
    rep.h3("C1. Road distance vs straight-line distance")
    rep.table(allr.circuity.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_frame("distance / straight-line"))
    lane_c = allr.groupby(["pickup", "delivery"]).agg(circuity=("circuity", "median"), distance=("distance", "mean"), straight_line=("straight_line", "first"), n=("load_id", "size"))
    rep.p("Lanes with the most extreme ratios. The top ones sit on the 70-mile distance floor (C3) while their "
          "coordinates are almost on top of each other; the lowest ratios are still ≥ 1.06, physically plausible.")
    rep.table(pd.concat([lane_c.nlargest(6, "circuity"), lane_c.nsmallest(4, "circuity")]), digits=2)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    hb = ax.hexbin(allr.straight_line, allr.distance, gridsize=70, bins="log", cmap=SEQ_CMAP, mincnt=1)
    x = np.linspace(0, allr.straight_line.max(), 10)
    ax.plot(x, x * allr.circuity.median(), color=ORANGE, linewidth=1.2, label=f"distance = {allr.circuity.median():.2f} × straight-line")
    ax.set(title="Road distance vs straight-line distance", xlabel="straight-line miles (from given coordinates)", ylabel="distance column (miles)")
    ax.legend(loc="upper left")
    fig.colorbar(hb, ax=ax, label="loads (log)")
    rep.figure(fig, "distance_vs_haversine", "The distance column is consistent with a ~1.18 road-circuity factor; deviations come from distorted coordinates.")

    rep.h3("C2. Distance noise within a lane")
    lane_d = allr.groupby(["pickup", "delivery"]).distance.agg(["mean", "std", "min", "max", "count"])
    lane_d["cv %"] = 100 * lane_d["std"] / lane_d["mean"]
    rep.p(
        f"Within a lane, distance varies with median CV {lane_d['cv %'].median():.1f}% (p95 {lane_d['cv %'].quantile(0.95):.1f}%): "
        "per-load route noise, not a different lane."
    )
    rep.table(lane_d[lane_d["max"] / lane_d["min"] > 1.4].round(1))
    sym = lane_d["mean"].rename("ab").reset_index()
    sym = sym.merge(sym.rename(columns={"pickup": "delivery", "delivery": "pickup", "ab": "ba"}), on=["pickup", "delivery"])
    rep.p(f"Symmetry: median |d(A→B) − d(B→A)| / d = {100 * ((sym.ab - sym.ba).abs() / sym.ab).median():.2f}% across {len(sym):,} lane pairs.")

    rep.h3("C3. A floor on distance")
    floor_tab = pd.DataFrame(
        {
            n: {
                "rows with distance == 70.0": (d.distance == 70).sum(),
                "next most common lowest value count": d.distance[d.distance > 70].value_counts().sort_index().iloc[0],
                "lanes with at least one row at 70.0": d[d.distance == 70].groupby(["pickup", "delivery"]).ngroups,
            }
            for n, d in (("train", tr), ("validation", val))
        }
    )
    rep.table(floor_tab)
    rep.p("70.0 appears dozens of times while every other value appears a handful of times: distance is **floored at 70 miles**. "
          "For those short lanes the true distance is 'at most 70', so rate per mile there is inflated.")
    summary += [
        {"issue": "distance floored at 70 mi", "train": (tr.distance == 70).sum(), "validation": (val.distance == 70).sum(),
         "evidence": "pile-up at exactly 70.0", "candidate handling": "keep; short-haul fixed-cost structure absorbs it"},
    ]
    summary += [
        {"issue": "clipped coordinates", "train": tr.pickup.isin(clipped.index).sum() + tr.delivery.isin(clipped.index).sum(),
         "validation": val.pickup.isin(clipped.index).sum() + val.delivery.isin(clipped.index).sum(),
         "evidence": "cities pinned at round extreme lat/lon", "candidate handling": "trust `distance`; use coords only as coarse region features"},
        {"issue": "distance noise within lane", "train": "", "validation": "",
         "evidence": f"median CV {lane_d['cv %'].median():.1f}%", "candidate handling": "use distance as-is (it is what the load ran)"},
    ]

    # ===================================================================== D. label outliers
    rep.h2("D. Label outliers")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.hist(np.log10(tr.rel_rate), bins=300, color=BLUE)
    for t in (OUTLIER_LO, OUTLIER_HI):
        ax.axvline(np.log10(t), color=ORANGE, linewidth=1.2)
    ax.set(title="rel_rate (log scale): rate per mile ÷ typical for lane & equipment", yscale="log", ylabel="loads (log)")
    ticks = [0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5]
    ax.set_xticks(np.log10(ticks), [f"{t:g}×" for t in ticks])
    rep.figure(
        fig,
        "rel_rate_gap",
        f"A dense core around 1× and two separated clusters. Orange lines = thresholds {OUTLIER_LO}× / {OUTLIER_HI}×, "
        "placed in the empty gaps, so the exact choice barely matters.",
    )
    hi, lo = tr.rel_rate > OUTLIER_HI, tr.rel_rate < OUTLIER_LO
    core = tr.rel_rate[~tr.is_outlier]
    gap_hi = ((tr.rel_rate > 1.25) & (tr.rel_rate <= OUTLIER_HI)).sum()
    gap_lo = ((tr.rel_rate >= OUTLIER_LO) & (tr.rel_rate < 0.8)).sum()
    rep.bullets(
        [
            f"Inflated labels (> {OUTLIER_HI}×): {hi.sum()} ({100 * hi.mean():.2f}%), range {tr.rel_rate[hi].min():.2f}×–{tr.rel_rate[hi].max():.2f}×.",
            f"Deflated labels (< {OUTLIER_LO}×): {lo.sum()} ({100 * lo.mean():.2f}%), range {tr.rel_rate[lo].min():.2f}×–{tr.rel_rate[lo].max():.2f}× "
            f"(i.e. ÷{1 / tr.rel_rate[lo].max():.1f} to ÷{1 / tr.rel_rate[lo].min():.1f}).",
            f"Core (non-outlier) rel_rate: p0.1–p99.9 = {core.quantile(0.001):.3f}–{core.quantile(0.999):.3f}, "
            f"median absolute deviation from 1 = {100 * (core - 1).abs().median():.1f}% (this includes real effects such as "
            "weight and market, so pure label noise is smaller).",
            f"Rows in the gap zones (1.25×, 2×] and [0.5×, 0.8×): {gap_hi} and {gap_lo}: the clusters are genuinely separate.",
        ]
    )
    fig, ax = plt.subplots(figsize=(7, 3.2))
    bins = np.linspace(1.8, 6, 50)
    ax.hist(tr.rel_rate[hi], bins=bins, color=BLUE, alpha=0.6, label="inflated: rel_rate")
    ax.hist(1 / tr.rel_rate[lo], bins=bins, histtype="step", color=ORANGE, linewidth=2, label="deflated: 1 / rel_rate")
    ax.set(title="Size of the corruption factor", xlabel="multiplier", ylabel="loads")
    ax.legend()
    rep.figure(fig, "corruption_factor", "Inflated and deflated labels share the same factor range (~2–5.5×): looks like random ×k or ÷k corruption.")

    rep.h3("D1. Are outliers concentrated anywhere? (chi-square test of independence)")
    factors = {
        "equipment": tr.equipment,
        "month": tr.date.dt.month,
        "day of week": tr.date.dt.dayofweek,
        "distance quintile": pd.qcut(tr.distance, 5, labels=False),
        "weight status": weight_status(tr.weight),
        "market_index missing": tr.market_index.isna(),
        "pickup city": tr.pickup,
    }
    conc = []
    for name, f in factors.items():
        tab = pd.crosstab(f, tr.is_outlier)
        chi2, p, *_ = chi2_contingency(tab)
        rate = tr.is_outlier.groupby(f).mean() * 100
        conc.append({"factor": name, "levels": len(tab), "outlier % min": rate.min(), "outlier % max": rate.max(), "chi-square p": p})
    rep.table(pd.DataFrame(conc), index=False, digits=3)
    rep.p("High p-values mean the outliers are scattered at random rather than tied to a segment.")

    rep.h3("D2. How much do 1.3% of rows matter?")
    pred = tr.expected_rpm * tr.distance
    err = (pred - tr.posted_rate)
    rep.table(
        pd.DataFrame(
            {
                "share of rows": [tr.is_outlier.mean()],
                "share of total absolute error": [err.abs()[tr.is_outlier].sum() / err.abs().sum()],
                "share of total squared error": [(err**2)[tr.is_outlier].sum() / (err**2).sum()],
            },
            index=["outlier rows"],
        ),
        digits=3,
    )
    es = pd.DataFrame({"all rows": error_summary(tr.posted_rate, pred), "non-outlier rows": error_summary(tr.posted_rate[~tr.is_outlier], pred[~tr.is_outlier])}).T
    rep.p("Error of the in-sample 'lane typical rate' reference itself (not a model, just a yardstick):")
    rep.table(es, digits=2)
    rep.p("A squared-error loss would spend most of its effort on these few corrupted rows → drop them from training and/or use a robust loss.")
    summary.append(
        {"issue": "corrupted labels (×/÷ 2–5.5)", "train": tr.is_outlier.sum(), "validation": "unknown",
         "evidence": "isolated clusters, random across segments", "candidate handling": "drop from training; robust loss; check metric choice"}
    )

    # ===================================================================== summary
    rep.h2("Summary of issues")
    rep.table(pd.DataFrame(summary), index=False)
    rep.p("Validation shows the same issues at similar or higher rates (tables A and B), so every fix must be a "
          "function applied at prediction time, not a one-off edit of the training file.")
    rep.save()


if __name__ == "__main__":
    main()
