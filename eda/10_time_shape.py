"""10 · Deep-dive: the exact shape of the time effects, and what December should look like.

Step 05 found, after market_index: a ramp into each quarter end, a reset on day 1 of the next quarter,
and a slow upward drift. Here we pin down the *shape* with the refined structure from 09, because
(a) it decides which date features a model needs, and (b) the December chart is purely a time test.

Strategy (simple → complex), all fitted to the daily average residual of the structural model:
  A  nothing                       B  linear drift
  C  drift + linear quarter-end ramp (ramp length R chosen by grid)
  C' drift + curved ramp (power p)  D  per-quarter levels + ramp (steps instead of drift)
  E  drift + ramp with a separate height per quarter
Then: does the ramp learned on Q1–Q2 predict Q3? What's left after the best model? What does the
structure + time model imply for the December chart?
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    BLUE,
    GRID,
    INK_2,
    ORANGE,
    Report,
    add_calendar,
    clean_inputs,
    clean_train,
    daily_market_index,
    fit_structural,
    load_raw,
    ols,
    predict_structural,
    set_style,
    structural_matrix,
)

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
T0 = pd.Timestamp("2025-01-01")


def calendar_frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
    f = add_calendar(pd.DataFrame({"date": dates}))
    f["t"] = (f.date - T0).dt.days.astype(float)
    return f.set_index("date")


def ramp(dtq: np.ndarray, R: float, p: float = 1.0) -> np.ndarray:
    return np.clip(1 - dtq / R, 0, None) ** p


def design(f: pd.DataFrame, kind: str, R: float = 24, p: float = 1.0) -> np.ndarray:
    r = ramp(f.days_to_quarter_end.values, R, p)
    t = f.t.values / 30.0  # drift per 30 days
    q = f.quarter.values
    if kind == "B":
        return t[:, None]
    if kind in ("C", "C'"):
        return np.column_stack([t, r])
    if kind == "D":
        return np.column_stack([(q == 2), (q == 3), (q == 4), r]).astype(float)
    if kind == "E":
        return np.column_stack([t, r * (q == 1), r * (q == 2), r * (q == 3)])
    raise ValueError(kind)


def bic(rss: float, n: int, k: int) -> float:
    return n * np.log(rss / n) + k * np.log(n)


def main() -> None:
    set_style()
    train, val, dec = load_raw()
    c, _ = clean_train(train, val)
    ref = fit_structural(c, "shared")
    daily = c.assign(r=ref["resid"]).groupby("date").r.mean()
    f = calendar_frame(daily.index)
    y = daily.values
    n = len(y)
    rep = Report(
        "10_time_shape",
        "10 · Deep-dive: the shape of the time effects",
        "Target: the daily average residual of the refined structural model (09), i.e. what distance, weight, market_index, equipment "
        f"and city do not explain. {n} days, Jan 1 – Oct 31. Daily noise from sampling ~158 loads/day is about "
        f"{100 * ref['resid'].std() / np.sqrt(c.groupby('date').size().mean()):.2f}%, so a perfect time model would leave roughly that.",
    )

    # 1. compare candidate models ------------------------------------------------------------------------
    rep.h2("1. Which time model describes the residual best?")
    rows = []

    def add(name, X, k_extra=0):
        if X is None:
            rss = np.sum((y - y.mean()) ** 2)
            k = 1
        else:
            _, fit = ols(X, y)
            rss = np.sum((y - fit) ** 2)
            k = X.shape[1] + 1 + k_extra
        rows.append({"model": name, "parameters": k, "daily RMSE (%)": 100 * np.sqrt(rss / n), "BIC (lower = better)": bic(rss, n, k)})

    add("A  nothing (constant)", None)
    add("B  linear drift", design(f, "B"))
    gridR = np.arange(7, 61)
    rssR = [np.sum((y - ols(design(f, "C", R), y)[1]) ** 2) for R in gridR]
    R_best = int(gridR[int(np.argmin(rssR))])
    add(f"C  drift + linear ramp over last {R_best} days", design(f, "C", R_best), k_extra=1)
    best_p = min(((np.sum((y - ols(design(f, "C'", R, p), y)[1]) ** 2), R, p) for R in gridR for p in (0.5, 0.75, 1.0, 1.5, 2.0)))
    add(f"C' drift + curved ramp (R={best_p[1]}, power={best_p[2]})", design(f, "C'", best_p[1], best_p[2]), k_extra=2)
    add(f"D  quarter levels (steps) + ramp (R={R_best})", design(f, "D", R_best), k_extra=1)
    add(f"E  drift + ramp height per quarter (R={R_best})", design(f, "E", R_best), k_extra=1)
    tab = pd.DataFrame(rows)
    rep.table(tab, index=False, digits=3)
    bC, fitC = ols(design(f, "C", R_best), y)
    bE, _ = ols(design(f, "E", R_best), y)
    rep.bullets(
        [
            f"Best ramp length for a straight-line ramp: **{R_best} days** (grid 7–60).",
            f"Model C: drift **{100 * bC[1]:+.2f}% per 30 days**, ramp height **{100 * bC[2]:+.2f}%** at quarter end (rising linearly "
            f"from 0 at {R_best} days before), resetting on day 1 of the next quarter.",
            f"Ramp height fitted separately per quarter (model E): Q1 {100 * bE[2]:+.2f}%, Q2 {100 * bE[3]:+.2f}%, Q3 {100 * bE[4]:+.2f}%. "
            "Similar heights support one shared ramp.",
            "Lower BIC = better trade-off between fit and number of parameters. More complex variants only win if their BIC is clearly lower.",
        ]
    )
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(gridR, 100 * np.sqrt(np.array(rssR) / n), color=BLUE)
    ax.axvline(R_best, color=ORANGE, linewidth=1)
    ax.set(title="Model C: fit vs ramp length", xlabel="ramp length R (days before quarter end)", ylabel="daily RMSE (%)")
    rep.figure(fig, "ramp_length", f"The fit is clearly best around R ≈ {R_best} days, so the ramp length is well determined by the data.")

    # 2. out-of-sample check ---------------------------------------------------------------------------------
    rep.h2("2. Does the pattern generalise to a quarter it hasn't seen?")
    oos = []
    for cut, label in ((pd.Timestamp("2025-06-30"), "fit Jan–Jun → predict Jul–Oct (incl. the Q3 ramp)"), (pd.Timestamp("2025-08-31"), "fit Jan–Aug → predict Sep–Oct (Q3 ramp + Oct)")):
        tr_m, te_m = f.index <= cut, f.index > cut
        row = {"test": label, "constant (%)": 100 * np.sqrt(np.mean((y[te_m] - y[tr_m].mean()) ** 2))}
        for kind, name in (("B", "drift only (%)"), ("C", f"drift + ramp R={R_best} (%)")):
            X = design(f, kind, R_best)
            b, _ = ols(X[tr_m], y[tr_m])
            pred = np.column_stack([np.ones(te_m.sum()), X[te_m]]) @ b
            row[name] = 100 * np.sqrt(np.mean((y[te_m] - pred) ** 2))
        oos.append(row)
    rep.table(pd.DataFrame(oos), index=False, digits=3)
    rep.p("Daily RMSE on the held-out months. If drift + ramp, learned only from earlier quarters, predicts the later ones well, the "
          "pattern is a stable rule rather than a coincidence. (The structural model behind the residual used all months, but it has "
          "no time terms, so the time pattern itself is not fitted.)")

    # 3. ramp profile & leftovers ------------------------------------------------------------------------------
    rep.h2("3. Ramp shape up close, and what is left")
    no_ramp = y - (bC[0] + bC[1] * f.t.values / 30)
    prof = pd.Series(no_ramp).groupby(f.days_to_quarter_end.values).mean()
    prof = prof[prof.index <= 60]
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.8))
    ax = axes[0]
    ax.scatter(prof.index, 100 * prof.values, s=14, color=BLUE, label="average over Q1–Q3 (drift removed)", zorder=3)
    xs = np.arange(0, 61)
    ax.plot(xs, 100 * bC[2] * ramp(xs, R_best), color=ORANGE, linewidth=1.8, label=f"fitted linear ramp (R={R_best})")
    ax.invert_xaxis()
    ax.axhline(0, color=INK_2, linewidth=0.6)
    ax.set(title="Residual by days to quarter end", xlabel="days to quarter end", ylabel="residual (%)")
    ax.legend(loc="upper left")
    left = y - fitC
    ax = axes[1]
    ax.plot(f.index, 100 * y, color=BLUE, linewidth=0.8, label="daily residual")
    ax.plot(f.index, 100 * fitC, color=ORANGE, linewidth=1.6, label="model C")
    for qe in pd.to_datetime(["2025-03-31", "2025-06-30", "2025-09-30"]):
        ax.axvspan(qe - pd.Timedelta(days=R_best - 1), qe + pd.Timedelta(hours=12), color=GRID, alpha=0.6, linewidth=0)
    ax.set(title="Daily residual and model C (shaded = ramp windows)", ylabel="%")
    ax.legend(loc="upper left")
    fig.tight_layout()
    rep.figure(fig, "ramp_profile", "Left: the average ramp is close to a straight line rising into quarter end. Right: drift + ramp tracks the daily residual through all three quarters.")
    lf = pd.Series(left, index=f.index)
    dow = (100 * lf.groupby(f.dow.values).mean()).reindex(range(7))
    dow.index = DOW
    rep.bullets(
        [
            f"Leftover daily RMSE {100 * lf.std():.2f}% vs ~{100 * ref['resid'].std() / np.sqrt(c.groupby('date').size().mean()):.2f}% expected from sampling alone.",
            f"Day-to-day autocorrelation of what is left: {lf.autocorr(1):.2f} (≈ 0 means no slow pattern remains).",
            "Leftover by weekday (%): " + ", ".join(f"{k} {v:+.2f}" for k, v in dow.items()) + ".",
            f"Leftover by month (%): " + ", ".join(f"{pd.Timestamp(2025, m, 1):%b} {v:+.2f}" for m, v in (100 * lf.groupby(f.month.values).mean()).items()) + ".",
        ]
    )

    rep.h3("The low-market bend in step 09 was a time effect")
    r_time = ref["resid"] - pd.Series(fitC, index=f.index).reindex(c.date).values
    mbins = np.arange(0.70, 1.46, 0.05)
    cut = pd.cut(c.mi.values, mbins)
    mb = pd.DataFrame({"structure only (%)": 100 * ref["resid"].values, "after drift + ramp (%)": 100 * r_time.values}).groupby(cut, observed=True).median()
    mb["loads"] = pd.Series(1, index=range(len(c))).groupby(cut, observed=True).size().values
    mb.index = mb.index.astype(str)
    mb.index.name = "market_index"
    rep.table(mb, digits=2)
    low = c[c.mi < 0.8].date.dt.month.value_counts(normalize=True)
    rep.p(f"In step 09 the residual looked high at market_index < 0.8. {100 * low.get(9, 0):.0f}% of those loads are in September, inside the Q3 "
          "ramp. Once drift + ramp are removed, the line is flat: the market effect itself is straight.")

    rep.h3("Is the slow leftover a market effect we've mis-specified?")
    rep.p("The leftover drifts slowly (high autocorrelation, + in May–Jun, − in Aug–Sep), which lines up with the market's seasonal level. "
          "Test: refit the structure + drift + ramp, letting price respond separately to the market's seasonal level (7-day mean), its weekly "
          "wiggle, and each load's deviation, or to a lagged (trailing) market average.")
    dmi_all = daily_market_index(train, val)
    level7 = dmi_all.rolling(7, center=True, min_periods=4).mean()
    cc = add_calendar(c)
    mi_day, mi_lvl = cc.date.map(dmi_all).values, cc.date.map(level7).values
    Xs, nm = structural_matrix(cc, ref["cities"], "shared")
    Xb = np.delete(Xs, nm.index("market_index"), axis=1)
    tt = (cc.date - T0).dt.days.values / 30
    rr = ramp(cc.days_to_quarter_end.values, R_best)
    yy = np.log(cc.posted_rate.values)
    mrows = []
    for label, cols, names_ in (
        ("one market_index term", [cc.mi.values], ["market"]),
        ("separate: seasonal level / weekly wiggle / per-load deviation", [mi_lvl, mi_day - mi_lvl, cc.mi.values - mi_day], ["level", "weekly", "per-load"]),
        ("market_index + trailing 14-day average", [cc.mi.values, cc.date.map(dmi_all.rolling(14, min_periods=1).mean()).values], ["market", "trailing 14d"]),
        ("market_index + trailing 28-day average", [cc.mi.values, cc.date.map(dmi_all.rolling(28, min_periods=1).mean()).values], ["market", "trailing 28d"]),
    ):
        b, fit_ = ols(np.column_stack([Xb, tt, rr, *cols]), yy)
        dres = pd.Series(yy - fit_).groupby(cc.date.values).mean()
        mrows.append({"market terms": label, "coefficients": ", ".join(f"{n_} {v:.3f}" for n_, v in zip(names_, b[-len(cols):])),
                      "daily RMSE (%)": 100 * dres.std(), "leftover autocorrelation": dres.autocorr(1)})
    rep.table(pd.DataFrame(mrows), index=False, digits=3)
    rep.p("Price responds about equally to the seasonal level and the weekly wiggle, and lagged market averages add nothing, so **one "
          "market_index term is the right description**. The slow leftover (about ±0.5% month to month) is not a market effect. With only "
          "10 months it can't be pinned down; it is small next to the ~1.8% per-load noise.")

    # 4. projection & December expectation -------------------------------------------------------------------------
    rep.h2("4. What this implies for Nov–Dec and the December chart")
    allf = calendar_frame(pd.date_range("2025-01-01", "2025-12-31"))
    Xall = design(allf, "C", R_best)
    proj = np.column_stack([np.ones(len(allf)), Xall]) @ bC
    frozen = proj.copy()
    t_oct31 = (pd.Timestamp("2025-10-31") - T0).days / 30
    after = allf.index > "2025-10-31"
    frozen[after] = bC[0] + bC[1] * t_oct31 + bC[2] * ramp(allf.days_to_quarter_end.values[after], R_best)
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.plot(f.index, 100 * y, color=BLUE, linewidth=0.8, label="daily residual (train)")
    ax.plot(allf.index[~after], 100 * proj[~after], color=ORANGE, linewidth=1.6, label="model C (fitted)")
    ax.plot(allf.index[after], 100 * proj[after], color=ORANGE, linewidth=1.6, linestyle="--", label="projection: drift continues")
    ax.plot(allf.index[after], 100 * frozen[after], color=INK_2, linewidth=1.4, linestyle="--", label="projection: drift frozen at Oct level")
    ax.set(title="Time component: fitted Jan–Oct, projected Nov–Dec", ylabel="% vs average")
    ax.legend(loc="upper left", ncol=2)
    rep.figure(fig, "projection", "Projection = an assumption, not data. Both versions share the Q4 ramp in December; they differ only in whether the slow drift keeps going.")

    dmi = daily_market_index(train, val)
    weight_fill = train.assign(w=train.weight.abs()).groupby("equipment").w.median().to_dict()
    d = clean_inputs(dec.assign(market_index=dec.date.map(dmi)), dmi, weight_fill)
    base_log = predict_structural(ref, d)
    tcomp = pd.Series(proj, index=allf.index).reindex(d.date).values
    tfroz = pd.Series(frozen, index=allf.index).reindex(d.date).values
    exp_tab = pd.DataFrame(
        {
            "date": d.date.values,
            "dow": d.date.dt.day_name().str[:3].values,
            "market_index": d.mi.values,
            "structure only ($)": np.exp(base_log),
            "+ time, drift continues ($)": np.exp(base_log + tcomp),
            "+ time, drift frozen ($)": np.exp(base_log + tfroz),
        }
    )
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(exp_tab.date, exp_tab["structure only ($)"], color=BLUE, marker="o", markersize=3, label="structure + market only (no time effects)")
    ax.plot(exp_tab.date, exp_tab["+ time, drift continues ($)"], color=ORANGE, marker="o", markersize=3, label="+ drift + quarter-end ramp")
    ax.set(title="Expected December price: Lexington → Fort Wayne, 360 mi, Dry Van, 32,000 lb", ylabel="$")
    ax.legend(loc="upper left")
    rep.figure(fig, "december_expectation", "A sanity reference for the final model's chart, not a submission: weekly zig-zag from market_index, rising into Dec 31 from the Q4 ramp.")
    rep.table(exp_tab.set_index("date").iloc[::3], digits=3)
    rep.bullets(
        [
            f"Structure + market alone: ${exp_tab['structure only ($)'].min():.0f}–${exp_tab['structure only ($)'].max():.0f}, a weekly zig-zag (Thu high, Sun low).",
            f"Adding drift + ramp: ${exp_tab['+ time, drift continues ($)'].min():.0f}–${exp_tab['+ time, drift continues ($)'].max():.0f}; "
            f"the last week of December sits ~{100 * (np.exp(bC[2]) - 1):.1f}% above the first week's level, plus drift.",
            f"Train history for this lane (Dry Van, all weights): median $808 (step 07). Structure-only December mean ${exp_tab['structure only ($)'].mean():.0f}.",
            "Open modelling question: let the drift continue into Nov–Dec, or freeze it? Trees freeze it automatically (they can't extrapolate). "
            "A linear trend term continues it. Time-forward validation can tell us which works better.",
        ]
    )
    rep.save()


if __name__ == "__main__":
    main()
