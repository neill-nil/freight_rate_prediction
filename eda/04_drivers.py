"""04 · What drives the rate — simplest explanation first, then one factor at a time.

Strategy (simple → complex):
  1. distance alone (one straight line)            → how far does the obvious driver get us?
  2. functional form of distance per equipment     → fixed cost + per-mile, or a power law?
  3. equipment premium, measured *within* a lane   → multiplicative or additive?
  4. weight / market_index / quote_signal          → effect shape and size on the lane-normalised rate
  5. geography: what's left at lane level          → real lane effects or noise? origin/destination
                                                     structure? spatially smooth (matters for unseen cities)?
Corrupted labels (step 03) are excluded so they don't distort shapes.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    BLUE,
    DIV_CMAP,
    EQUIP_COLOR,
    EQUIPMENT,
    INK_2,
    ORANGE,
    SEQ_CMAP,
    WEIGHT_CAP,
    Report,
    add_rate_features,
    binned,
    city_coords,
    error_summary,
    flag_outliers,
    load_raw,
    ols,
    plot_binned,
    relative_rate,
    set_style,
)

DIST_BINS = [69, 100, 150, 200, 300, 400, 500, 650, 800, 1000, 1250, 1500, 1800, 2100, 2500, 3000, 3500]


def slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(ols(x[:, None], y)[0][1])


def main() -> None:
    set_style()
    train, _, _ = load_raw()
    tr = add_rate_features(train)
    rel, premium = relative_rate(tr)
    tr = tr.join(rel)
    tr["is_outlier"] = flag_outliers(tr.rel_rate)
    c = tr[~tr.is_outlier].copy()
    c["weight_abs"] = c.weight.abs()
    rep = Report(
        "04_drivers",
        "04 · What drives the rate",
        f"Train rows without corrupted labels (n = {len(c):,}). We start with the most obvious driver and add one "
        "factor at a time, measuring each on the lane-normalised rate (`rel_rate`) so factors don't get confused "
        "with lane mix.",
    )
    effects = []

    # 1. distance alone ------------------------------------------------------------------------
    rep.h2("1. Distance alone")
    b_lin, f_lin = ols(c.distance.values[:, None], c.posted_rate.values)
    b_log, f_log = ols(np.log(c.distance.values)[:, None], np.log(c.posted_rate.values))
    tab = pd.DataFrame(
        {
            f"rate = {b_lin[0]:.0f} + {b_lin[1]:.3f}·distance": error_summary(c.posted_rate, f_lin),
            f"rate = {np.exp(b_log[0]):.2f}·distance^{b_log[1]:.3f}": error_summary(c.posted_rate, np.exp(f_log)),
        }
    ).T
    tab.index.name = "in-sample fit (descriptive only)"
    rep.table(tab, digits=2)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    hb = ax.hexbin(c.distance, c.posted_rate, gridsize=80, bins="log", cmap=SEQ_CMAP, mincnt=1)
    x = np.linspace(70, c.distance.max(), 50)
    ax.plot(x, b_lin[0] + b_lin[1] * x, color=ORANGE, linewidth=1.6, label="straight-line fit")
    ax.set(title="Rate vs distance", xlabel="distance (miles)", ylabel="posted_rate ($)")
    ax.legend(loc="upper left")
    fig.colorbar(hb, ax=ax, label="loads (log)")
    rep.figure(
        fig,
        "rate_vs_distance",
        f"Distance alone gives R² = {tab['R2'].iloc[0]:.3f}, yet MAPE is still {tab['MAPE (%)'].iloc[0]:.1f}%: "
        "the vertical spread at a given distance is what everything else has to explain.",
    )

    # 2. functional form per equipment -------------------------------------------------------------
    rep.h2("2. Shape of the distance effect, per equipment")
    rows, fits, pfits = [], {}, {}
    pred_fp, pred_pl = pd.Series(index=c.index, dtype=float), pd.Series(index=c.index, dtype=float)
    for e in EQUIPMENT:
        d = c[c.equipment == e]
        bf, ff = ols(d.distance.values[:, None], d.posted_rate.values)
        bp, fp = ols(np.log(d.distance.values)[:, None], np.log(d.posted_rate.values))
        pred_fp[d.index], pred_pl[d.index] = ff, np.exp(fp)
        fits[e], pfits[e] = bf, bp
        rows.append(
            {
                "equipment": e,
                "fixed $/load (a)": bf[0],
                "$/mile (b)": bf[1],
                "MAPE % fixed+per-mile": error_summary(d.posted_rate, ff)["MAPE (%)"],
                "power-law exponent": bp[1],
                "MAPE % power law": error_summary(d.posted_rate, np.exp(fp))["MAPE (%)"],
            }
        )
    rep.table(pd.DataFrame(rows), index=False, digits=3)
    m_fp, m_pl = error_summary(c.posted_rate, pred_fp)["MAPE (%)"], error_summary(c.posted_rate, pred_pl)["MAPE (%)"]
    exps = [pfits[e][1] for e in EQUIPMENT]
    rep.p(
        f"All equipment together: fixed+per-mile MAPE {m_fp:.2f}%, power law MAPE {m_pl:.2f}%. "
        f"Power-law exponents per equipment: {', '.join(f'{x:.3f}' for x in exps)}. Both forms say rate per mile falls "
        "with distance (a fixed cost spread over more miles); the residual plot shows which form has the right shape."
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
    xs = np.geomspace(70, 3400, 100)
    ax = axes[0]
    for e in EQUIPMENT:
        d = c[c.equipment == e]
        b = binned(d.distance, d.rpm, DIST_BINS)
        ax.scatter(b.x_mid, b["median"], color=EQUIP_COLOR[e], s=26, zorder=3, label=e, edgecolors="white", linewidths=1)
        ax.plot(xs, np.exp(pfits[e][0]) * xs ** (pfits[e][1] - 1), color=EQUIP_COLOR[e], linewidth=1.4)
    ax.set(xscale="log", title="Rate per mile vs distance (dots: binned medians, lines: power-law fit)", xlabel="distance (miles, log)", ylabel="$ per mile")
    ax.legend()
    ax = axes[1]
    for pred, label, color in ((pred_fp, "fixed + per-mile", ORANGE), (pred_pl, "power law", BLUE)):
        b = binned(c.distance, 100 * (pred / c.posted_rate - 1), DIST_BINS)
        ax.plot(b.x_mid, b["median"], color=color, marker="o", label=label)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set(xscale="log", title="Which form has the right shape? Median error by distance", xlabel="distance (miles, log)", ylabel="median error (%)")
    ax.legend()
    fig.tight_layout()
    rep.figure(
        fig,
        "rpm_taper",
        "Left: rate per mile falls with distance in the same shape for every equipment. Right: a correct functional form "
        "leaves a flat error line; a systematic curve means the form is wrong and a model would have to learn the correction.",
    )
    rpm_fit = lambda dist: np.exp(pfits["Dry Van"][0]) * dist ** (pfits["Dry Van"][1] - 1)  # noqa: E731
    p5, p95 = c.distance.quantile([0.05, 0.95])
    effects.append({"driver": "distance (p5→p95, rate per mile)", "effect size": f"{100 * (rpm_fit(p95) / rpm_fit(p5) - 1):+.0f}% per mile ({p5:.0f}→{p95:.0f} mi)"})

    # 3. equipment premium within lane -----------------------------------------------------------
    rep.h2("3. Equipment premium, measured within the same lane")
    lane_eq = c.groupby(["pickup", "delivery", "equipment"]).rpm.median().unstack()
    lane_dist = c.groupby(["pickup", "delivery"]).distance.mean()
    prem_rows, prem_by_dist = [], {}
    for e in ("Reefer", "Flatbed"):
        r = (lane_eq[e] / lane_eq["Dry Van"]).dropna()
        prem_rows.append({"equipment vs Dry Van": e, "lanes": len(r), "median ratio": r.median(), "p25": r.quantile(0.25), "p75": r.quantile(0.75)})
        prem_by_dist[e] = r.groupby(pd.cut(lane_dist.reindex(r.index), [69, 300, 600, 1000, 1500, 2000, 3500]), observed=True).median()
    rep.table(pd.DataFrame(prem_rows), index=False)
    pbd = pd.DataFrame(prem_by_dist)
    pbd.index = pbd.index.astype(str)
    pbd.index.name = "lane distance (mi)"
    rep.p("Premium by lane distance. A constant ratio means the premium is **multiplicative** (a % of the rate); an additive $/mile premium would make the ratio drift with distance.")
    rep.table(pbd, digits=3)
    effects += [{"driver": f"equipment: {e}", "effect size": f"{100 * (premium[e] - 1):+.1f}% vs Dry Van"} for e in ("Reefer", "Flatbed")]

    # 4. weight, market_index, quote_signal on rel_rate -------------------------------------------
    rep.h2("4. Weight, market_index and quote_signal (on the lane-normalised rate)")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    specs = [
        ("weight_abs", "weight (lb, abs)", np.arange(5_000, WEIGHT_CAP + 2_501, 2_500)),
        ("market_index", "market_index", np.arange(0.65, 1.50, 0.05)),
        ("quote_signal", "quote_signal", np.arange(0.7, 3.65, 0.15)),
    ]
    for ax, (col, label, bins) in zip(axes, specs):
        for e in EQUIPMENT:
            d = c[(c.equipment == e) & c[col].notna()]
            b = binned(d[col], d.rel_rate, bins)
            b = b[b.n >= 30]
            plot_binned(ax, b, EQUIP_COLOR[e], e, band=(e == "Dry Van"))
        ax.axhline(1, color=INK_2, linewidth=0.8)
        ax.set(title=f"rel_rate vs {label}", xlabel=label)
    axes[0].set_ylabel("rel_rate (median; band = IQR for Dry Van)")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    rep.figure(fig, "weight_market_quote", "Heavier loads and stronger markets cost more, in the same way for every equipment type. quote_signal shows almost no relationship overall (but see step 06).")

    elastic = []
    w_ok = c[c.weight_abs.between(5_001, WEIGHT_CAP - 1)]
    m_ok = c[c.market_index.notna()]
    for name, frame, xcol, logx in (("weight (log-log elasticity)", w_ok, "weight_abs", True), ("market_index (log-log elasticity)", m_ok, "market_index", True), ("quote_signal (slope of log rel_rate)", c, "quote_signal", False)):
        row = {"factor": name}
        for e in ["all", *EQUIPMENT]:
            d = frame if e == "all" else frame[frame.equipment == e]
            xv = np.log(d[xcol].values) if logx else d[xcol].values
            row[e] = slope(xv, np.log(d.rel_rate.values))
        row["Spearman with rel_rate"] = frame[xcol].corr(frame.rel_rate, method="spearman")
        elastic.append(row)
    rep.table(pd.DataFrame(elastic), index=False, digits=3)
    w5, w95 = w_ok.weight_abs.quantile([0.05, 0.95])
    m5, m95 = m_ok.market_index.quantile([0.05, 0.95])
    effects += [
        {"driver": "weight (p5→p95)", "effect size": f"{100 * ((w95 / w5) ** elastic[0]['all'] - 1):+.1f}% ({w5:,.0f}→{w95:,.0f} lb)"},
        {"driver": "market_index (p5→p95)", "effect size": f"{100 * ((m95 / m5) ** elastic[1]['all'] - 1):+.1f}% ({m5:.2f}→{m95:.2f})"},
    ]

    # 5. geography ------------------------------------------------------------------------------
    rep.h2("5. Geography: what is left at lane level?")
    lanes = c.groupby(["pickup", "delivery"]).agg(base=("lane_base_rpm", "first"), dist=("distance", "mean"), n=("rpm", "size"), within_sd=("rel_rate", "std"))
    lanes = lanes[lanes.n >= 3].copy()
    # flexible distance curve (cubic in log distance) so curve misfit can't masquerade as a lane/city effect
    ld = np.log(lanes.dist.values)
    _, curve = ols(np.column_stack([ld, ld**2, ld**3]), np.log(lanes.base.values))
    lanes["premium"] = np.log(lanes.base) - curve
    corr_dist = np.corrcoef(lanes.premium, ld)[0, 1]
    sigma_w = np.sqrt(np.nanmean(lanes.within_sd**2))
    noise_sd = np.sqrt(np.mean((1.2533 * sigma_w / np.sqrt(lanes.n)) ** 2))
    rep.p(
        "Lane premium = log(lane's typical Dry-Van-equivalent rate per mile) − log(what a flexible distance curve predicts). "
        f"Lanes with ≥ 3 clean loads: {len(lanes):,}. Correlation of premium with log distance: {corr_dist:.3f} "
        "(≈ 0 confirms the curve removed distance)."
    )

    cities = sorted(set(lanes.index.get_level_values(0)) | set(lanes.index.get_level_values(1)))
    idx = {city: i for i, city in enumerate(cities)}
    k = len(cities)
    X = np.zeros((len(lanes), 2 * k))
    X[np.arange(len(lanes)), [idx[o] for o in lanes.index.get_level_values(0)]] = 1
    X[np.arange(len(lanes)), [k + idx[d] for d in lanes.index.get_level_values(1)]] = 1
    wts = np.sqrt(lanes.n.values)
    beta, *_ = np.linalg.lstsq(X * wts[:, None], lanes.premium.values * wts, rcond=None)
    fitted = X @ beta
    origin = pd.Series(beta[:k], index=cities)
    dest = pd.Series(beta[k:], index=cities)
    origin, dest = origin - origin.mean(), dest - dest.mean()  # only differences are identifiable
    resid = lanes.premium.values - fitted
    r2 = 1 - np.average(resid**2, weights=lanes.n) / np.average((lanes.premium - np.average(lanes.premium, weights=lanes.n)) ** 2, weights=lanes.n)
    var_tab = pd.DataFrame(
        {
            "std (%)": [100 * lanes.premium.std(), 100 * noise_sd, 100 * np.std(resid)],
        },
        index=pd.Index(
            ["observed lane premium", "expected from sampling noise alone", "left after origin + destination effects"],
            name="lane-level variation",
        ),
    )
    rep.table(var_tab, digits=2)
    rep.p(
        f"An additive origin + destination model explains {100 * r2:.0f}% of lane-premium variance. The closer the leftover is to "
        "the sampling-noise line, the less a model needs lane identity, which matters because 12% of validation rows are "
        "on lanes never seen in training."
    )

    eff = pd.DataFrame({"origin effect %": 100 * origin, "destination effect %": 100 * dest})
    rep.p(f"Correlation between a city's origin effect and its destination effect: {eff.corr().iloc[0, 1]:.2f}. "
          "Strongly negative would mean headhaul/backhaul imbalance; strongly positive means a regional price level that "
          "applies whichever way the truck goes.")
    top = pd.concat([eff.nlargest(6, "origin effect %"), eff.nsmallest(6, "origin effect %")])
    rep.p("Most and least expensive origins:")
    rep.table(top, digits=2)

    cc = city_coords(train).drop_duplicates("city").set_index("city").reindex(cities)
    coords = cc[["lat", "lon"]].values
    dmat = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(dmat, np.inf)
    nn = np.argsort(dmat, axis=1)[:, :3]
    smooth = {}
    for name, s in (("origin", origin), ("destination", dest)):
        neigh = s.values[nn].mean(axis=1)
        smooth[name] = {
            "corr with 3 nearest cities' mean": np.corrcoef(s.values, neigh)[0, 1],
            "corr with lon": np.corrcoef(s.values, cc.lon)[0, 1],
            "corr with lat": np.corrcoef(s.values, cc.lat)[0, 1],
            "std (%)": 100 * s.std(),
        }
    rep.p("Are city effects spatially smooth? If neighbours look alike, unseen cities can borrow from nearby ones.")
    rep.table(pd.DataFrame(smooth).T, digits=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    lim = float(np.nanmax(np.abs(eff.values)))
    for ax, col in zip(axes, eff.columns):
        sc = ax.scatter(cc.lon, cc.lat, c=eff[col], cmap=DIV_CMAP, vmin=-lim, vmax=lim, s=70, edgecolors="white", linewidths=1, zorder=3)
        for city, r in cc.iterrows():
            ax.annotate(city, (r.lon, r.lat), xytext=(3, 3), textcoords="offset points", fontsize=5.5, color=INK_2)
        ax.set(title=col.replace(" %", " (% vs average)"), xlabel="longitude", ylabel="latitude")
    fig.colorbar(sc, ax=axes, label="% (red = more expensive)", shrink=0.85)
    rep.figure(fig, "city_effects_map", "City effects on rate after distance and equipment. Spatial clustering means geography (lat/lon) can stand in for city identity.")

    pairs = lanes.premium.rename("ab").reset_index()
    pairs = pairs.merge(pairs.rename(columns={"pickup": "delivery", "delivery": "pickup", "ab": "ba"}), on=["pickup", "delivery"])
    rep.p(f"Direction matters? Correlation of premium(A→B) with premium(B→A) over {len(pairs):,} directed lanes: {pairs.ab.corr(pairs.ba):.2f}.")
    effects += [
        {"driver": "lane premium (std across lanes)", "effect size": f"±{100 * lanes.premium.std():.1f}% (noise alone ±{100 * noise_sd:.1f}%)"},
        {"driver": "origin city (std)", "effect size": f"±{100 * origin.std():.1f}%"},
        {"driver": "destination city (std)", "effect size": f"±{100 * dest.std():.1f}%"},
    ]

    # summary ------------------------------------------------------------------------------------
    rep.h2("Summary: effect sizes, ranked by how much they move the rate")
    rep.table(pd.DataFrame(effects), index=False)
    rep.save()


if __name__ == "__main__":
    main()
