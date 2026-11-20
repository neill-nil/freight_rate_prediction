"""11 · Deep-dive: can we price cities the model has never seen?

12% of validation loads touch one of 8 cities absent from training. Step 04/09 found that each city has
one price level (same as pickup or delivery) and that neighbouring cities look alike. Here we test that
directly by simulating the situation: hide each training city in turn and predict its effect from the others.

Strategy (simple → complex):
  0. ignore it (treat as an average city)
  1. average of the k nearest known cities (k = 1…8), plain or distance-weighted
  2. a smooth surface over latitude/longitude (linear, quadratic)
Then translate the best method into price error for affected loads, and apply it to the 8 new cities.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    DIV_CMAP,
    INK,
    INK_2,
    Report,
    add_calendar,
    city_coords,
    clean_train,
    fit_structural,
    haversine_miles,
    load_raw,
    ols,
    set_style,
)


def knn_predict(D: np.ndarray, eff: np.ndarray, i: int, k: int, weighted: bool, exclude_self: bool = True) -> float:
    d = D[i].copy()
    if exclude_self:
        d[i] = np.inf
    nn = np.argsort(d)[:k]
    if not weighted:
        return float(eff[nn].mean())
    w = 1 / np.maximum(d[nn], 1.0)
    return float(np.sum(w * eff[nn]) / w.sum())


def surface_loo(lat: np.ndarray, lon: np.ndarray, eff: np.ndarray, degree: int) -> np.ndarray:
    cols = [lat, lon] if degree == 1 else [lat, lon, lat**2, lon**2, lat * lon]
    X = np.column_stack(cols)
    out = np.empty(len(eff))
    for i in range(len(eff)):
        m = np.arange(len(eff)) != i
        b, _ = ols(X[m], eff[m])
        out[i] = b[0] + X[i] @ b[1:]
    return out


def main() -> None:
    set_style()
    train, val, _ = load_raw()
    c, _ = clean_train(train, val)
    ref = fit_structural(c, "shared")
    eff = ref["city"]
    cities = list(eff.index)
    cc = city_coords(train, val).drop_duplicates("city").set_index("city")
    known = cc.loc[cities]
    lat, lon = known.lat.values, known.lon.values
    D = haversine_miles(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    e = eff.values
    rep = Report(
        "11_unseen_cities",
        "11 · Deep-dive: pricing cities never seen in training",
        f"City effects come from the refined structural model (09), one effect per city. They range {100 * e.min():+.1f}% to "
        f"{100 * e.max():+.1f}% (std {100 * e.std():.2f}%). Distances between cities use the dataset's own coordinates.",
    )

    # 1. leave-one-city-out on the effects -------------------------------------------------------------------
    rep.h2("1. Hide one city, predict its effect from the others")
    rows = [{"method": "0  ignore (average city = 0)", "RMSE of effect (%)": 100 * np.sqrt(np.mean(e**2)), "share of city variation explained": 0.0}]
    preds = {"0  ignore (average city = 0)": np.zeros_like(e)}
    for k in (1, 2, 3, 5, 8):
        for weighted in (False, True):
            p = np.array([knn_predict(D, e, i, k, weighted) for i in range(len(e))])
            name = f"1  {k} nearest {'city' if k == 1 else 'cities'}, {'distance-weighted' if weighted else 'plain average'}"
            preds[name] = p
            rows.append({"method": name, "RMSE of effect (%)": 100 * np.sqrt(np.mean((e - p) ** 2)), "share of city variation explained": 1 - np.mean((e - p) ** 2) / np.mean(e**2)})
    for deg, label in ((1, "2  linear surface over lat/lon"), (2, "2  quadratic surface over lat/lon")):
        p = surface_loo(lat, lon, e, deg)
        preds[label] = p
        rows.append({"method": label, "RMSE of effect (%)": 100 * np.sqrt(np.mean((e - p) ** 2)), "share of city variation explained": 1 - np.mean((e - p) ** 2) / np.mean(e**2)})
    loo = pd.DataFrame(rows)
    rep.table(loo, index=False, digits=3)
    best = loo.iloc[1:].sort_values("RMSE of effect (%)").iloc[0]
    best_name = best["method"]
    rep.p(f"Best: **{best_name.split('  ', 1)[1]}**: error {best['RMSE of effect (%)']:.2f}% vs {loo.iloc[0]['RMSE of effect (%)']:.2f}% when ignoring the city, "
          f"i.e. it recovers **{100 * best['share of city variation explained']:.0f}%** of a hidden city's price level from its neighbours alone.")
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    p = preds[best_name]
    ax.scatter(100 * e, 100 * p, s=30, color="#2a78d6", edgecolors="white", linewidths=1, zorder=3)
    lim = 100 * max(np.abs(e).max(), np.abs(p).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], color=INK_2, linewidth=0.8)
    for i, city in enumerate(cities):
        if abs(e[i] - p[i]) > 0.012:
            ax.annotate(city, (100 * e[i], 100 * p[i]), xytext=(3, 3), textcoords="offset points", fontsize=6.5, color=INK_2)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), title="Hidden city: true vs predicted effect", xlabel="true city effect (%)", ylabel="predicted from neighbours (%)")
    rep.figure(fig, "loo_scatter", "Each dot is one city predicted only from the others. Dots near the diagonal mean neighbours are a good stand-in.")

    # 2. what it means for price error --------------------------------------------------------------------------
    rep.h2("2. What that means for the price error of affected loads")
    cal = add_calendar(c)
    daily = c.assign(r=ref["resid"]).groupby("date").r.mean()
    t = ((daily.index - pd.Timestamp("2025-01-01")).days.values / 30)
    dtq = cal.groupby("date").days_to_quarter_end.first().reindex(daily.index).values
    b, fit_t = ols(np.column_stack([t, np.clip(1 - dtq / 28, 0, None)]), daily.values)
    r = ref["resid"].values - pd.Series(fit_t, index=daily.index).reindex(c.date).values  # residual after time model
    idx = {city: i for i, city in enumerate(cities)}
    pi, di = c.pickup.map(idx).values, c.delivery.map(idx).values
    erows = []
    for name in ("0  ignore (average city = 0)", best_name):
        shift = preds[name] - e  # predicted − true effect
        ape = []
        for which in (pi, di):  # scenario: this end of the load is the unseen city
            ape.append(np.abs(np.exp(-(r - shift[which])) - 1))
        ape = np.concatenate(ape)
        erows.append({"city effect used for the unseen city": name.split("  ", 1)[1], "MAPE of affected loads (%)": 100 * ape.mean(), "median APE (%)": 100 * np.median(ape)})
    known_ape = np.abs(np.exp(-r) - 1)
    erows.insert(0, {"city effect used for the unseen city": "city known (its own fitted effect)", "MAPE of affected loads (%)": 100 * known_ape.mean(), "median APE (%)": 100 * np.median(known_ape)})
    rep.table(pd.DataFrame(erows), index=False, digits=3)
    rep.p("Structural model + time model (drift + ramp), in-sample. Each load is scored twice: once pretending its pickup city is new, once its "
          "delivery city. The gap between rows is the cost of not having seen a city, and how much of it the neighbours win back.")

    # 3. the 8 new cities -----------------------------------------------------------------------------------------
    rep.h2("3. The 8 validation-only cities")
    new = sorted(set(cc.index) - set(cities))
    nn_known = np.sort(np.where(np.eye(len(cities), dtype=bool), np.inf, D), axis=1)[:, 0]
    k_best = int(best_name.split("  ", 1)[1].split(" ")[0]) if best_name.startswith("1") else 3
    weighted = "weighted" in best_name
    nrows = []
    for city in new:
        la, lo = cc.loc[city, "lat"], cc.loc[city, "lon"]
        d = haversine_miles(la, lo, lat, lon)
        order = np.argsort(d)
        w_ = 1 / np.maximum(d[order[:k_best]], 1.0) if weighted else np.ones(k_best)
        pred = float(np.sum(w_ * e[order[:k_best]]) / w_.sum())
        nrows.append({"new city": city, "nearest known cities (distance, mi)": ", ".join(f"{cities[j]} ({d[j]:.0f})" for j in order[:3]),
                      "predicted effect (%)": 100 * pred, "val loads touching it": int(((val.pickup == city) | (val.delivery == city)).sum())})
    ntab = pd.DataFrame(nrows)
    rep.table(ntab, index=False, digits=2)
    new_nn = [haversine_miles(cc.loc[cty, "lat"], cc.loc[cty, "lon"], lat, lon).min() for cty in new]
    rep.p(f"Distance to the nearest known city: new cities median {np.median(new_nn):.0f} mi (max {np.max(new_nn):.0f}); known cities among "
          f"themselves median {np.median(nn_known):.0f} mi. New cities are about as close to their neighbours as known cities are, so the "
          "hide-one-city test above is a fair simulation of their situation.")

    fig, ax = plt.subplots(figsize=(11, 5.6))
    lim = 100 * max(np.abs(e).max(), np.abs(ntab["predicted effect (%)"].values / 100).max())
    sc = ax.scatter(lon, lat, c=100 * e, cmap=DIV_CMAP, vmin=-lim, vmax=lim, s=55, edgecolors="white", linewidths=1, zorder=3)
    for city in new:
        row = ntab.set_index("new city").loc[city]
        ax.scatter(cc.loc[city, "lon"], cc.loc[city, "lat"], c=[row["predicted effect (%)"]], cmap=DIV_CMAP, vmin=-lim, vmax=lim, s=150, marker="D", edgecolors=INK, linewidths=1.4, zorder=4)
        ax.annotate(city, (cc.loc[city, "lon"], cc.loc[city, "lat"]), xytext=(6, -10), textcoords="offset points", fontsize=8, fontweight="bold", color=INK)
    for i, city in enumerate(cities):
        ax.annotate(city, (lon[i], lat[i]), xytext=(3, 3), textcoords="offset points", fontsize=5.5, color=INK_2)
    fig.colorbar(sc, ax=ax, label="city effect (%, red = more expensive)", shrink=0.85)
    ax.set(title="Known city effects (circles) and predicted effects for the 8 new cities (diamonds)", xlabel="longitude", ylabel="latitude")
    rep.figure(fig, "new_cities_map", "New cities get the price level of their neighbourhood.")
    rep.bullets(
        [
            "A model should describe location by coordinates (or neighbour-based city effects), not city names: names give no information about a new city.",
            "Tree models with lat/lon features learn this neighbourhood structure naturally. A linear model needs it supplied as a feature "
            "(e.g. the k-nearest-neighbour city effect).",
        ]
    )
    rep.save()


if __name__ == "__main__":
    main()
