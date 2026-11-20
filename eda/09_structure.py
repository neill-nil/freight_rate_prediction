"""09 · Deep-dive: does price follow a simple multiplicative structure?

Step 04 suggested that every factor moves price by a *percentage* and that factors don't interact.
If true, log(price) is a plain sum of effects:

    log(rate) = const + f(distance) + g(weight) + h(market_index) + equipment + city effects + time + noise

Strategy (simple → complex):
  1. ladder: add one factor at a time with the simplest shapes (logs) and measure what each explains
  2. interactions: does any factor's effect depend on another? (add each product term, measure gain)
  3. shapes: are logs the right shape for distance, weight, market? (residual bends + alternative forms)
  4. a second outlier pass with the refined structure
  5. noise: size, shape, constant in %? (decides how errors behave in MAPE vs RMSE)
Time effects are deliberately left in the residual here (deep-dive 10).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm

from common import (
    BLUE,
    EQUIP_COLOR,
    EQUIPMENT,
    INK_2,
    MODEL_OUTLIER_LOG,
    ORANGE,
    WEIGHT_CAP,
    WEIGHT_HI,
    WEIGHT_LO,
    Report,
    binned,
    clean_train,
    error_summary,
    fit_structural,
    load_raw,
    ols,
    plot_binned,
    set_style,
    structural_matrix,
)

DIST_BINS = [69, 100, 150, 200, 300, 400, 500, 650, 800, 1000, 1250, 1500, 1800, 2100, 2500, 3000, 3500]


def within_day_sd(resid: np.ndarray, dates: pd.Series) -> float:
    r = pd.Series(resid, index=dates.index)
    return float((r - r.groupby(dates).transform("mean")).std())


def ols_with_se(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    sigma2 = resid @ resid / (len(y) - A.shape[1])
    se = np.sqrt(np.diag(sigma2 * np.linalg.pinv(A.T @ A)))
    return beta, se, resid


def residual_panels(axes, c: pd.DataFrame, r: pd.Series, title_prefix: str) -> None:
    specs = [("distance", "distance (miles, log)", DIST_BINS, True), ("weight_abs", "weight (lb)", np.arange(5_000, WEIGHT_CAP + 2_501, 2_500), False), ("mi", "market_index", np.arange(0.65, 1.50, 0.05), False)]
    for ax, (col, label, bins, logx) in zip(axes, specs):
        for e in EQUIPMENT:
            m = c.equipment == e
            b = binned(c.loc[m, col], r[m], bins)
            plot_binned(ax, b[b.n >= 30], EQUIP_COLOR[e], e, band=(e == "Dry Van"))
        ax.axhline(0, color=INK_2, linewidth=0.8)
        ax.set(title=f"{title_prefix}: residual vs {label.split(' (')[0]}", xlabel=label, xscale="log" if logx else "linear")


def main() -> None:
    set_style()
    train, val, _ = load_raw()
    c, full = clean_train(train, val)
    cities = sorted(set(c.pickup) | set(c.delivery))
    y = np.log(c.posted_rate.values)
    rep = Report(
        "09_structure",
        "09 · Deep-dive: is price a simple product of independent effects?",
        f"Clean train rows (n = {len(c):,}). Fits are ordinary least squares on log(price), in-sample and descriptive: few parameters "
        "on 47k rows, so overfitting is negligible. Time effects are left in the residual on purpose (deep-dive 10). "
        "'Within-day noise' = residual after also removing each day's average, i.e. with time effects taken out.",
    )

    # 1. ladder ------------------------------------------------------------------------------------------
    rep.h2("1. Ladder: add one factor at a time (simplest shapes: logs)")
    base = {
        "log_distance": c.log_distance.values,
        "Reefer": (c.equipment == "Reefer").values.astype(float),
        "Flatbed": (c.equipment == "Flatbed").values.astype(float),
        "log_weight": c.log_weight.values,
        "log_mi": c.log_mi.values,
    }
    steps = [
        ("+ distance curve (power law)", ["log_distance"], "none"),
        ("+ equipment", ["log_distance", "Reefer", "Flatbed"], "none"),
        ("+ weight", ["log_distance", "Reefer", "Flatbed", "log_weight"], "none"),
        ("+ market_index", ["log_distance", "Reefer", "Flatbed", "log_weight", "log_mi"], "none"),
        ("+ one effect per city (shared by pickup & delivery)", list(base), "shared"),
        ("+ separate pickup and delivery effects", list(base), "separate"),
    ]
    rows = []
    pred0 = c.rpm.median() * c.distance
    rows.append({"step": "global median rate per mile × distance", "parameters": 1, "residual std (%)": 100 * np.std(np.log(pred0) - y), "within-day noise (%)": 100 * within_day_sd(np.log(pred0.values) - y, c.date), **error_summary(c.posted_rate, pred0)})
    for name, cols, mode in steps:
        X = np.column_stack([base[k] for k in cols])
        if mode != "none":
            Xc, nm = structural_matrix(c, cities, mode, "log")
            X = np.column_stack([X, Xc[:, len([n for n in nm if not n[:2] in ("o:", "d:", "c:")]):]])
        _, f = ols(X, y)
        rows.append({"step": name, "parameters": X.shape[1] + 1, "residual std (%)": 100 * np.std(y - f), "within-day noise (%)": 100 * within_day_sd(y - f, c.date), **error_summary(c.posted_rate, np.exp(f))})
    ref = fit_structural(c, "shared", form="refined")
    rows.append({"step": "→ refined shapes (section 3), shared cities", "parameters": len(ref["coef"]), "residual std (%)": 100 * ref["resid"].std(), "within-day noise (%)": 100 * within_day_sd(ref["resid"].values, c.date), **error_summary(c.posted_rate, np.exp(ref["fitted"]))})
    ladder = pd.DataFrame(rows).drop(columns=["R2"])
    rep.table(ladder, index=False, digits=2)
    rep.p("Each line adds one factor to the line above. A big drop means the factor matters; a flat line means it adds nothing. "
          "Separate pickup/delivery effects add nothing over one shared effect per city, so a city's price level applies whichever "
          "end of the trip it is on. The last line uses the better shapes found in section 3.")

    # 2. interactions -------------------------------------------------------------------------------------
    rep.h2("2. Interactions: does any factor's effect depend on another?")
    X0, names0 = structural_matrix(c, cities, "shared", "refined")
    _, f0 = ols(X0, y)
    sd0 = np.std(y - f0)
    ld, lw, mi = c.log_distance.values, c.log_weight.values, c.mi.values
    wc = np.clip(c.weight_abs.values, WEIGHT_LO, WEIGHT_HI) / 1e4
    rf, fb = base["Reefer"], base["Flatbed"]
    cands = {
        "distance × Reefer": ld * rf,
        "distance × Flatbed": ld * fb,
        "weight × Reefer": wc * rf,
        "weight × Flatbed": wc * fb,
        "market × Reefer": mi * rf,
        "market × Flatbed": mi * fb,
        "market × distance": mi * ld,
        "weight × distance": wc * ld,
        "weight × market": wc * mi,
        "weight was missing (imputed)": c.weight_missing.values.astype(float),
        "market_index was missing (imputed)": c.mi_missing.values.astype(float),
        "distance at the 70-mile floor": (c.distance.values == 70).astype(float),
    }
    irows = []
    for name, col in cands.items():
        b, se, resid = ols_with_se(np.column_stack([X0, col]), y)
        irows.append({"added term": name, "coefficient": b[-1], "t-stat": b[-1] / se[-1], "residual std (%)": 100 * np.std(resid), "improvement (basis points)": 1e4 * (sd0 - np.std(resid))})
    rep.p(f"Base: refined shapes + shared city effects (residual std {100 * sd0:.3f}%). Each row adds one term on its own. "
          "1 basis point = 0.01 percentage points of residual std.")
    rep.table(pd.DataFrame(irows).sort_values("improvement (basis points)", ascending=False), index=False, digits=4)
    rep.p("With 47k rows even tiny effects can be 'statistically significant', so judge by **improvement**. Anything below a "
          "few basis points is irrelevant next to ~2% noise: **the factors act independently**.")

    # 3. shapes --------------------------------------------------------------------------------------------
    rep.h2("3. Shapes: are logs the right curve for distance, weight and market?")
    logf = fit_structural(c, "shared", form="log")
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.6), sharey=True)
    residual_panels(axes[0], c, 100 * logf["resid"], "Logs")
    residual_panels(axes[1], c, 100 * ref["resid"], "Refined")
    for ax in axes[:, 0]:
        ax.set_ylabel("residual (%, median)")
    axes[0, 0].legend(loc="lower right")
    fig.tight_layout()
    rep.figure(fig, "residual_shapes", "Top: with log terms, short hauls (100–250 mi) are underpriced by ~2% and the weight line bends "
               "(light loads underpriced, heavy loads overpriced). Bottom: the refined shapes flatten the lines.")
    cityX = X0[:, len([n for n in names0 if not n.startswith("c:")]):]
    eq = np.column_stack([rf, fb])
    w = c.weight_abs.values
    def form_sd(*cols):
        _, f = ols(np.column_stack([*cols, eq, cityX]), y)
        return 100 * within_day_sd(y - f, c.date), 100 * np.std(y - f)
    forms = {
        "log d · log w · log mi  (step 04 guess)": (ld, lw, np.log(mi)),
        "log d · linear w · log mi": (ld, w / 1e4, np.log(mi)),
        "log d · linear w · linear mi": (ld, w / 1e4, mi),
        f"log d · clip(w, {WEIGHT_LO / 1e3:.0f}k–{WEIGHT_HI / 1e3:.0f}k) · linear mi": (ld, wc, mi),
        "log d + (log d)² · clip w · linear mi": (ld, ld**2, wc, mi),
        "log d + (log d)² + (log d)³ · clip w · linear mi  (refined)": (ld, ld**2, ld**3, wc, mi),
        "log(d + 50) · clip w · linear mi": (np.log(c.distance.values + 50), wc, mi),
    }
    ftab = pd.DataFrame({k: dict(zip(["within-day noise (%)", "residual std (%)"], form_sd(*v))) for k, v in forms.items()}).T
    ftab.index.name = "functional form"
    rep.table(ftab, digits=3)
    grid = sorted((form_sd(ld, ld**2, ld**3, np.clip(w, a, b) / 1e4, mi)[0], a, b) for a in (10_000, 15_000, 17_500, 20_000, 22_500) for b in (35_000, 37_500, 40_000, 42_500, 45_000))
    rep.p(f"Weight bounds were chosen by a grid search over lower {{10k–22.5k}} × upper {{35k–45k}}: best {grid[0][1]:,}–{grid[0][2]:,} lb "
          f"({grid[0][0]:.3f}% noise), next best {grid[1][1]:,}–{grid[1][2]:,} ({grid[1][0]:.3f}%). Round numbers 20k–40k are used.")
    _, f_now = ols(np.column_stack([ld, ld**2, ld**3, mi, eq, cityX]), y)
    wb = pd.Series(100 * (y - f_now)).groupby(pd.cut(w, [4_999, 10_000, 15_000, 20_000, 25_000, 30_000, 35_000, 40_000, 42_500, 47_499, 47_500]), observed=True).agg(["median", "size"])
    wb.index = wb.index.astype(str)
    wb.index.name = "weight (lb)"
    wb.columns = ["residual with NO weight term (%)", "loads"]
    rep.p("Why clipped: the residual of a model without any weight term, by weight. It is flat below ~20k, rises steadily to ~40k, then is flat again.")
    rep.table(wb, digits=2)
    k = ref["coef"]
    dgrid = np.array([100, 250, 500, 1000, 2000])
    ldg = np.log(dgrid)
    f_d = k["log_distance"] * ldg + k["log_distance^2"] * ldg**2 + k["log_distance^3"] * ldg**3
    meaning = pd.DataFrame(
        {
            "effect": [
                "price ratio when distance doubles: " + ", ".join(f"{a}→{2 * a} mi ×{np.exp(f_d[i + 1] - f_d[i]) if i + 1 < len(dgrid) and dgrid[i + 1] == 2 * a else np.exp(k['log_distance'] * np.log(2)):.2f}" for i, a in enumerate(dgrid[:-1]) if dgrid[i + 1] == 2 * a),
                f"+{100 * (np.exp(k['weight_clipped_10k']) - 1):.1f}% per +10,000 lb between 20k and 40k lb; no effect outside "
                f"(full band = +{100 * (np.exp(2 * k['weight_clipped_10k']) - 1):.1f}%)",
                f"+{100 * (np.exp(0.1 * k['market_index']) - 1):.2f}% per +0.1 market_index",
                f"Reefer {100 * (np.exp(k['Reefer']) - 1):+.1f}%, Flatbed {100 * (np.exp(k['Flatbed']) - 1):+.1f}% vs Dry Van",
                f"city effects span {100 * ref['city'].min():+.1f}% to {100 * ref['city'].max():+.1f}% (std {100 * ref['city'].std():.1f}%)",
            ]
        },
        index=pd.Index(["distance", "weight", "market_index", "equipment", "city"], name="refined structure"),
    )
    rep.table(meaning)

    # 4. second-pass outliers --------------------------------------------------------------------------------
    rep.h2("4. A second outlier pass")
    extra = full[full.is_outlier_model & ~full.is_outlier_lane]
    rep.p(f"The lane-based rule (step 03) can't catch a corrupted price on a thin lane where the corrupted value *is* the lane's median. "
          f"Checking every row against the structural model (|log residual| > {MODEL_OUTLIER_LOG}, i.e. off by more than ×1.5) catches "
          f"**{len(extra)} more**. The corruption factors are ≥ ×2.2, so the gap to genuine noise (~2%) is wide.")
    rep.table(extra[["load_id", "pickup", "delivery", "distance", "equipment", "posted_rate", "rpm", "rel_rate", "structural_resid"]].assign(**{"price vs model": lambda d: np.exp(d.structural_resid).map(lambda v: f"×{v:.2f}")}).drop(columns="structural_resid"), index=False, digits=3)
    rep.p(f"Total removed from training: {int(full.is_outlier.sum())} rows ({100 * full.is_outlier.mean():.2f}%).")

    # 5. noise ---------------------------------------------------------------------------------------------
    rep.h2("5. The noise: how big, what shape, constant in %?")
    rw = ref["resid"] - ref["resid"].groupby(c.date).transform("mean")
    mad_sd = 1.4826 * np.median(np.abs(rw - np.median(rw)))
    noise = pd.DataFrame(
        {"value": [100 * ref["resid"].std(), 100 * rw.std(), 100 * mad_sd, kurtosis(rw), 100 * rw.quantile(0.001), 100 * rw.quantile(0.999)]},
        index=pd.Index(["residual std (%)", "…after removing day-level effects (%)", "robust std (MAD) (%)", "excess kurtosis (0 = normal)", "p0.1 (%)", "p99.9 (%)"], name="statistic"),
    )
    rep.table(noise, digits=3)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    xs = np.linspace(-10, 10, 300)
    ax.hist(100 * rw, bins=160, range=(-10, 10), density=True, color=BLUE, alpha=0.6, label="residual (day effects removed)")
    ax.plot(xs, norm.pdf(xs, 0, 100 * rw.std()), color=ORANGE, linewidth=1.6, label="normal curve, same std")
    ax.set(title="Noise distribution in log space", xlabel="residual (%)", ylabel="density")
    ax.legend()
    rep.figure(fig, "noise_shape", "Noise left after the refined structure and day-level effects.")
    cut = pd.cut(c.distance, [69, 300, 600, 1000, 1500, 2000, 3500])
    by = pd.DataFrame({"noise std (%)": (100 * rw).groupby(cut, observed=True).std(), "loads": rw.groupby(cut, observed=True).size()})
    by.index = by.index.astype(str)
    by.index.name = "distance (mi)"
    rep.table(by, digits=3)
    eq_sd = (100 * rw).groupby(c.equipment).std()
    rep.p("Noise std by equipment: " + ", ".join(f"{e} {eq_sd[e]:.2f}%" for e in EQUIPMENT) + ".")

    pred = np.exp(ref["fitted"])
    err = pred - c.posted_rate.values
    long_ = c.distance.values >= np.quantile(c.distance.values, 0.8)
    rep.h3("What this means for the two accuracy measures")
    rep.bullets(
        [
            "Noise is about the same **percentage** at every distance and for every equipment type, so % errors are comparable across "
            "loads: **MAPE** treats a $300 short haul and a $6,000 cross-country load equally.",
            f"In dollars the same % is far bigger on long loads. The longest 20% of loads produce **{100 * np.sum(err[long_] ** 2) / np.sum(err ** 2):.0f}% "
            f"of the squared dollar error** but only {100 * np.sum(np.abs(err[long_] / c.posted_rate.values[long_])) / np.sum(np.abs(err / c.posted_rate.values)):.0f}% "
            "of the summed % error. **RMSE** is therefore mostly a measure of long-haul accuracy.",
            "Reporting both shows whether a model is good on typical loads (MAPE) and on the expensive ones (RMSE). Training on log "
            "price suits MAPE; any model choice will be checked on both.",
            f"With ~{100 * rw.std():.1f}% irreducible noise, a MAPE around {100 * np.mean(np.abs(rw)):.1f}% is roughly the floor. A test "
            "score far below that would signal leakage.",
        ]
    )
    rep.save()


if __name__ == "__main__":
    main()
