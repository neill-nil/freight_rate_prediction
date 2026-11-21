"""Modelling step 4: feature-engineering audit. Have we used everything the data offers?

1. Inventory: raw inputs → what each model uses, and what was dropped and why
2. Screen of candidate features not yet tested, each added to structural v2 and judged by 5-fold
   out-of-fold error (so features with many parameters can't win by memorising)
3. The one that matters (lane effect) confirmed on the real test: future months with hidden cities
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

from freight.data import (  # noqa: E402
    city_coordinates, clean_inputs, daily_market_index, flag_corrupted, load_train, load_validation, weight_fill_values,
)
from freight.evaluation import FoldData, summarise  # noqa: E402
from freight.features import quarter_ramp  # noqa: E402
from freight.models import HybridModel, StructuralModel  # noqa: E402
from freight.reporting import Report, set_style  # noqa: E402


def main() -> None:
    set_style()
    train, val = load_train(), load_validation()
    coords = city_coordinates(train, val)
    dmi = daily_market_index(train, val)
    full = clean_inputs(train, dmi, weight_fill_values(train))
    c = full[~flag_corrupted(full)].reset_index(drop=True)
    g = pd.read_pickle(ROOT / "outputs" / "gbm_choice.pkl")
    v2, hp = g["structural_v2"], g["hybrid_params"]
    rep = Report("04_feature_screen", "Modelling 04 · Feature-engineering audit",
                 "Have we used everything the data offers before trying more model types? Inventory, a screen of untested candidates, "
                 "and a time-forward confirmation of the one that mattered.")

    # 1. inventory --------------------------------------------------------------------------------------------
    rep.h2("1. Inventory: from 12 raw inputs to model features")
    inv = pd.DataFrame([
        ("pickup, delivery (city names)", "linear: one shared effect per city (63 columns); unseen city → 5-nearest-neighbour effect; lane effect (pickup→delivery)", "trees: replaced by coordinates", "eda/04, 09, 11; modelling 04"),
        ("pickup/delivery lat & lon", "kNN distances for unseen cities", "trees: 4 features", "eda/03, 11"),
        ("distance", "log distance, its square and cube (curved distance effect)", "trees: log distance; target = log price per mile", "eda/04, 09; modelling 02"),
        ("equipment", "Reefer, Flatbed indicators; × quarter-end ramp", "trees: categorical", "eda/04; modelling 02"),
        ("weight", "abs() for sign flips, equipment-median for blanks, clipped to 20k–40k", "trees: cleaned weight", "eda/03, 09"),
        ("date", "quarter-end ramp (28 days) and its × equipment; slow drift (damped beyond training)", "trees: days to quarter end, time index", "eda/05, 10; modelling 01–02"),
        ("market_index", "linear term; blanks → same-date mean", "trees: same", "eda/03, 10"),
        ("quote_signal", "DROPPED: leakage trap (meaning flips by month, none in Nov–Dec)", "DROPPED", "eda/06, 08"),
    ], columns=["raw input", "structural (linear) model", "tree models", "evidence"])
    rep.table(inv, index=False)
    m = StructuralModel(coords=coords, **v2).fit(c)
    X0 = m._design(c, m._time_index(c))
    rep.p(f"Structural v2 design: {X0.shape[1]} columns ({X0.shape[1] - len(m.cities_) + 1} engineered terms + {len(m.cities_) - 1} city effects) "
          "+ intercept. Lane effects (below) add one shrunk level per directed lane seen in training.")
    dropped = pd.DataFrame([
        ("calendar", "day of week, month, 11 US holidays, quarter step levels, curved ramp, other ramp lengths", "no gain after market_index + ramp + drift; month can't generalise to unseen Nov–Dec", "eda/05, 10; modelling 01, 04"),
        ("market", "log vs linear, seasonal level / weekly / per-load split, trailing 14/28/42-day means, momentum, volatility, × equipment, × distance", "≤ 1 basis point", "eda/09, 10; modelling 04"),
        ("weight & distance shapes", "log weight, linear weight, other bands, kink, cap flag, 70-mile floor flag, log(d+c), degree 1/2/4, × equipment", "worse or ≤ 0.5 bp", "eda/09; modelling 01, 04"),
        ("missing-value flags", "weight was missing, market_index was missing", "0 bp: values are missing at random", "eda/03, 09"),
        ("geography", "separate pickup/delivery effects, lat/lon surfaces, region-pair effects, travel direction (Δlat/Δlon, bearing), city × equipment", "≤ 0.6 bp, or worse", "eda/04, 09, 11; modelling 04"),
        ("volume", "11 load counts (day, equipment-day, city in/out, imbalance, city-month, lane-month)", "≤ 0.03% per std", "eda/12"),
        ("drift", "drift × equipment", "0 gain", "modelling 02"),
    ], columns=["family", "engineered & tested", "why dropped", "evidence"])
    rep.p("Engineered, tested, and dropped:")
    rep.table(dropped, index=False)

    # 2. OOF screen ---------------------------------------------------------------------------------------------
    rep.h2("2. Screen of candidates not tested before")
    y = np.log(c.posted_rate.values)
    kf = list(KFold(5, shuffle=True, random_state=0).split(c))

    def oof(X):
        r = np.empty(len(y))
        for a, b in kf:
            beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(a)), X[a]]), y[a], rcond=None)
            r[b] = y[b] - np.column_stack([np.ones(len(b)), X[b]]) @ beta
        return r

    mape = lambda r: 100 * np.mean(np.abs(np.exp(-r) - 1))  # noqa: E731
    r0 = oof(X0)
    sd0, m0 = r0.std(), mape(r0)
    rows = []

    def test(name, cols):
        r = oof(np.column_stack([X0, cols]))
        rows.append({"candidate": name, "columns": 1 if cols.ndim == 1 else cols.shape[1], "residual-std gain (basis points)": 1e4 * (sd0 - r.std()), "MAPE gain (pts)": m0 - mape(r)})

    eq = c.equipment.values
    rf, fb = (eq == "Reefer").astype(float), (eq == "Flatbed").astype(float)
    dlat, dlon = (c.delivery_lat - c.pickup_lat).values, (c.delivery_lon - c.pickup_lon).values
    test("travel direction: Δlat, Δlon", np.column_stack([dlat, dlon]))
    ang = np.arctan2(dlat, dlon)
    test("travel direction: bearing (sin, cos)", np.column_stack([np.sin(ang), np.cos(ang)]))
    for k in (4, 6, 8):
        lab = pd.Series(KMeans(k, n_init=10, random_state=0).fit_predict(coords[["lat", "lon"]].values), index=coords.index)
        D = pd.get_dummies(c.pickup.map(lab) * k + c.delivery.map(lab)).values[:, 1:].astype(float)
        test(f"region-pair effects ({k} regions)", D)
    idx = {cc: i for i, cc in enumerate(m.cities_)}
    M = np.zeros((len(c), len(m.cities_)))
    for role in ("pickup", "delivery"):
        M[np.arange(len(c)), c[role].map(idx).values] += 1
    test("city × equipment", np.column_stack([M[:, 1:] * rf[:, None], M[:, 1:] * fb[:, None]]))
    dow = pd.get_dummies(c.date.dt.dayofweek).values[:, 1:].astype(float)
    test("day of week", dow)
    test("day of week × equipment", np.column_stack([dow * rf[:, None], dow * fb[:, None]]))
    lvl = dmi.rolling(7, center=True, min_periods=4).mean()
    for lag in (7, 14, 28):
        test(f"market momentum (level change over {lag} days)", np.nan_to_num((lvl - lvl.shift(lag)).reindex(c.date).values))
    vol = dmi.rolling(14, min_periods=7).std().reindex(c.date).values
    test("market volatility (14-day std)", np.nan_to_num(vol, nan=np.nanmean(vol)))
    for R in (14, 42):
        rr = quarter_ramp(c.date, R)
        test(f"second ramp, length {R} days, × equipment", np.column_stack([rr, rr * rf, rr * fb]))
    z = np.log(c.distance.values) - 7
    test("distance curve per equipment", np.column_stack([z**2 * rf, z**3 * rf, z**2 * fb, z**3 * fb]))
    lane = (c.pickup + "→" + c.delivery).values
    pair = np.where(c.pickup < c.delivery, c.pickup + "–" + c.delivery, c.delivery + "–" + c.pickup)
    for key_name, key in (("lane effect, directed (A→B ≠ B→A)", lane), ("lane effect, either direction", pair)):
        for lam in (3, 10):
            r = np.empty(len(y))
            for a, b in kf:
                A = np.column_stack([np.ones(len(a)), X0[a]])
                beta, *_ = np.linalg.lstsq(A, y[a], rcond=None)
                gg = pd.DataFrame({"k": key[a], "r": y[a] - A @ beta}).groupby("k").r.agg(["sum", "count"])
                enc = (gg["sum"] / (gg["count"] + lam)).reindex(key[b]).fillna(0).values
                r[b] = y[b] - (np.column_stack([np.ones(len(b)), X0[b]]) @ beta + enc)
            rows.append({"candidate": f"{key_name}, shrink λ={lam}", "columns": "1 level per lane", "residual-std gain (basis points)": 1e4 * (sd0 - r.std()), "MAPE gain (pts)": m0 - mape(r)})
    rep.p(f"Base: structural v2, out-of-fold residual std {100 * sd0:.3f}%, MAPE {m0:.3f}%. Each candidate is added on its own. The lane effect "
          "is each route's mean leftover, shrunk toward 0: level = sum of residuals / (number of loads + λ).")
    rep.table(pd.DataFrame(rows).sort_values("MAPE gain (pts)", ascending=False), index=False, digits=3)
    rep.bullets([
        "**One clear winner: the directed lane effect.** Every other candidate is at noise level.",
        "Directed beats either-direction, so routes are not symmetric: A→B can be pricier than B→A beyond what the two cities explain.",
        "Correction to the EDA (04): it found lane-level variation of ±2.1% after city effects vs ±1.8% expected from noise, i.e. about ±1% "
        "genuine lane effect, and called it small. With ~10 loads per lane, a ±1% effect is learnable and turns out to be the largest one left.",
    ])

    # 3. time-forward confirmation ---------------------------------------------------------------------------------
    rep.h2("3. Confirmation on the real test (future months, hidden cities)")
    fd = FoldData(train, val)
    S = lambda **kw: StructuralModel(coords=coords, **{**v2, **kw})  # noqa: E731
    variants = {"structural v2": lambda: S(), **{f"v2 + lane effect λ={lam}": (lambda lam=lam: S(lane_shrink=lam)) for lam in (2, 3, 5, 10)},
                "hybrid on v2 (previous front-runner)": lambda: HybridModel(S(), hp),
                "hybrid on v2 + lane effect λ=3": lambda: HybridModel(S(lane_shrink=3), hp)}
    pfs = [fd.evaluate(mk, k)[0] for k, mk in variants.items()]
    s = summarise(pd.concat(pfs))
    cols = ["MAPE (%)", "RMSE ($)", "MAE ($)", "MAPE seen cities (%)", "MAPE unseen cities (%)", "MAPE month 2 (%)", "bias (%)", "worst-fold MAPE (%)"]
    rep.table(s[cols], digits=3)
    pf = pd.concat(pfs).pivot(index="fold", columns="model", values="MAPE (%)")[["structural v2", "v2 + lane effect λ=3", "hybrid on v2 (previous front-runner)", "hybrid on v2 + lane effect λ=3"]]
    rep.p("MAPE per fold:")
    rep.table(pf, digits=3)
    v3, h3 = s.loc["v2 + lane effect λ=3"], s.loc["hybrid on v2 + lane effect λ=3"]
    rep.bullets([
        f"**Structural v3 = v2 + lane effect (λ=3)**: {v3['MAPE (%)']:.3f}% MAPE, ${v3['RMSE ($)']:.1f} RMSE, better than the previous hybrid in every fold.",
        f"Trees on top add only {v3['MAPE (%)'] - h3['MAPE (%)']:.3f} MAPE points and ${v3['RMSE ($)'] - h3['RMSE ($)']:.2f} RMSE, below the 0.02 / $1 'clear gain' bar, "
        "so the simpler v3 becomes the front-runner. The trees had mostly been approximating the lane effect from coordinates.",
        f"Unseen-city loads don't benefit ({v3['MAPE unseen cities (%)']:.3f}%): their lanes have never been seen. That is the main place left to gain.",
        "λ = 2–3 is best; theory agrees (λ ≈ noise variance ÷ lane-effect variance ≈ 1.6² ÷ 1.0² ≈ 3).",
    ])
    rep.save()
    pd.to_pickle({"structural_v3": {**v2, "lane_shrink": 3}, "summary": s}, ROOT / "outputs" / "feature_choice.pkl")


if __name__ == "__main__":
    main()
