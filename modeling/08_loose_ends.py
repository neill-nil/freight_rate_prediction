"""Modelling step 8: loose ends before finalising. Ideas not tried yet, each judged on the same future-month folds.

1. fitting to the metric: weight expensive loads more (dollar errors) and metric-specific bias corrections
2. recency: weight recent months more, or train on recent months only
3. lane effects: fitted jointly with the other effects (backfitting) instead of in a second step; lane × equipment
4. neighbour lanes: routes never seen in training (incl. those with an unseen city) borrow from nearby routes
5. combination of whatever helped
6. robustness: other hidden-city draws, other corrupted-label cut-offs, Nov–Dec sensitivity to the drift assumption
Clear-gain bar to replace structural v3: ≥ 0.02 MAPE points AND ≥ $1 RMSE.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from freight.data import (  # noqa: E402
    city_coordinates, clean_inputs, daily_market_index, flag_corrupted, load_december, load_train, load_validation, weight_fill_values,
)
from freight.evaluation import FoldData, summarise  # noqa: E402
from freight.features import haversine_miles, months_since_start  # noqa: E402
from freight.models import HybridModel, StructuralModel  # noqa: E402
from freight.reporting import AQUA, BLUE, INK_2, ORANGE, Report, set_style  # noqa: E402

SHOW = ["MAPE (%)", "RMSE ($)", "MAPE seen cities (%)", "MAPE unseen cities (%)", "bias (%)", "worst-fold MAPE (%)",
        "Δ MAPE vs v3", "Δ RMSE vs v3", "folds better", "clear gain"]
V3 = "structural v3 (reference)"


def wlstsq(A: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    s = np.sqrt(w)
    return np.linalg.lstsq(A * s[:, None], y * s, rcond=None)[0]


class StructuralX(StructuralModel):
    """Structural v3 with experimental options. With all options at their defaults it is exactly v3.

    price_power : weight each load by (first-pass fitted price)^p; p=2 ≈ minimising squared dollar error
    half_life   : weight a load by 0.5^(age in months / half_life) (recency)
    window      : train on the last `window` months only
    joint_iters : backfitting rounds that re-fit the main effects net of lane effects (0 = two-step, as v3)
    lane_equip_shrink : extra shrunk level per lane × equipment
    prior_bw, prior_shrink, prior_scope : neighbour-lane prior. A lane's prior = shrunk, kernel-weighted mean residual of
                  routes whose pickup is near its pickup and delivery near its delivery (Gaussian kernel, bandwidth in miles,
                  the lane itself excluded). scope 'all': every lane shrinks toward its prior instead of 0; 'unseen': only
                  routes absent from training use it.
    adjust      : 'mean' (+σ²/2, the dollar-mean of a log-normal: RMSE-optimal) | 'mape' (−σ², MAPE-optimal)
    """

    def __init__(self, price_power: float = 0.0, half_life: float | None = None, window: float | None = None,
                 joint_iters: int = 0, lane_equip_shrink: float | None = None, prior_bw: float | None = None,
                 prior_shrink: float = 10.0, prior_scope: str = "all", adjust: str | None = None, **kw) -> None:
        super().__init__(**kw)
        self.price_power, self.half_life, self.window = price_power, half_life, window
        self.joint_iters, self.lane_equip_shrink = joint_iters, lane_equip_shrink
        self.prior_bw, self.prior_shrink, self.prior_scope, self.adjust = prior_bw, prior_shrink, prior_scope, adjust

    def fit(self, df: pd.DataFrame) -> "StructuralX":
        if self.window:
            df = df[df["date"] > df["date"].max() - np.timedelta64(int(30 * self.window), "D")]
        self.cities_ = sorted(set(df["pickup"]) | set(df["delivery"]))
        t = months_since_start(df["date"])
        self.t_max_ = float(t.max())
        A = np.column_stack([np.ones(len(df)), self._design(df, t)])
        y = np.log(df["posted_rate"].values)
        w = np.ones(len(df))
        if self.half_life:
            w = w * 0.5 ** ((self.t_max_ - t) / self.half_life)
        if self.price_power:
            f0 = A @ wlstsq(A, y, w)
            w = w * np.exp(self.price_power * (f0 - f0.mean()))
        w = w / w.mean()
        lanes = pd.Series(self._lane_key(df))
        off = np.zeros(len(df))
        for _ in range(1 + (self.joint_iters if self.lane_shrink is not None else 0)):
            self.coef_ = wlstsq(A, y - off, w)
            if self.lane_shrink is None:
                break
            self._fit_lanes(df, y - A @ self.coef_, w)
            off = lanes.map(self.lane_effects_).fillna(0.0).values
        n_city = len(self.cities_) - 1
        self.city_effects_ = pd.Series(np.r_[0.0, self.coef_[len(self.coef_) - n_city:]], index=self.cities_)
        resid = y - A @ self.coef_ - off
        if self.lane_equip_shrink is not None:
            key = lanes + "|" + df["equipment"].values
            g = pd.DataFrame({"k": key, "s": w * resid, "n": w}).groupby("k")[["s", "n"]].sum()
            self.lane_equip_ = g["s"] / (g["n"] + self.lane_equip_shrink)
            resid = resid - key.map(self.lane_equip_).values
        self.sigma2_ = float(np.mean(resid**2))
        return self

    def _kernel(self, cities) -> np.ndarray:
        c = self.coords
        d = haversine_miles(c.loc[cities, "lat"].values[:, None], c.loc[cities, "lon"].values[:, None],
                            c.loc[self.cities_, "lat"].values[None, :], c.loc[self.cities_, "lon"].values[None, :])
        return np.exp(-0.5 * (d / self.prior_bw) ** 2)

    def _fit_lanes(self, df: pd.DataFrame, resid: np.ndarray, w: np.ndarray) -> None:
        ci = {c: i for i, c in enumerate(self.cities_)}
        a, b = df["pickup"].map(ci).values, df["delivery"].map(ci).values
        K = len(self.cities_)
        S, N = np.zeros((K, K)), np.zeros((K, K))
        np.add.at(S, (a, b), w * resid)
        np.add.at(N, (a, b), w)
        lam = self.lane_shrink
        E = S / (N + lam)
        if self.prior_bw is not None:
            self._S, self._N, kc = S, N, self._kernel(self.cities_)
            if self.prior_scope == "all":
                prior = (kc @ S @ kc.T - S) / (kc @ N @ kc.T - N + self.prior_shrink)  # the lane itself excluded
                E = (S + lam * prior) / (N + lam)
        ii, jj = np.nonzero(N)
        self.lane_effects_ = pd.Series(E[ii, jj], index=[f"{self.cities_[i]}→{self.cities_[j]}" for i, j in zip(ii, jj)])

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        out = np.asarray(super().predict_log(df), float).copy()  # seen lanes get their effect, unseen lanes 0
        lanes = pd.Series(self._lane_key(df))
        if self.prior_bw is not None:
            new = ~lanes.isin(self.lane_effects_.index).values
            if new.any():
                sub = df[new]
                ka, kb = self._kernel(list(sub["pickup"])), self._kernel(list(sub["delivery"]))
                num = np.sum((ka @ self._S) * kb, axis=1)
                den = np.sum((ka @ self._N) * kb, axis=1)
                out[new] += num / (den + self.prior_shrink)
        if self.lane_equip_shrink is not None:
            out = out + (lanes + "|" + df["equipment"].values).map(self.lane_equip_).fillna(0.0).values
        if self.adjust == "mean":
            out = out + self.sigma2_ / 2
        elif self.adjust == "mape":
            out = out - self.sigma2_
        return out


def main() -> None:
    set_style()
    t0 = time.time()
    train, val, dec = load_train(), load_validation(), load_december()
    coords = city_coordinates(train, val)
    v3 = pd.read_pickle(ROOT / "outputs" / "feature_choice.pkl")["structural_v3"]
    hp = pd.read_pickle(ROOT / "outputs" / "gbm_choice.pkl")["hybrid_params"]
    fd = FoldData(train, val)
    print(f"folds ready ({time.time() - t0:.0f}s)", flush=True)
    rep = Report("08_loose_ends", "Modelling 08 · Loose ends before finalising",
                 "Ideas not tried before, each added to structural v3 and scored on the same future-month folds (4 folds, 2-month horizon, "
                 "4 hidden cities each, genuine test rows). Clear-gain bar to replace v3: ≥ 0.02 MAPE points **and** ≥ $1 RMSE. "
                 "'folds better' = folds (of 4) where the variant's MAPE beats v3's.")

    def make(**extra):
        return lambda: StructuralX(coords=coords, **{**v3, **extra})

    res: dict[str, tuple[pd.DataFrame, list[np.ndarray]]] = {}

    def run(label: str, factory, fdx: FoldData = fd) -> pd.DataFrame:
        pf, preds = fdx.evaluate(factory, label)
        res[label] = (pf, preds)
        print(f"  {label}: MAPE {pf['MAPE (%)'].mean():.3f}  RMSE {pf['RMSE ($)'].mean():.2f}  ({time.time() - t0:.0f}s)", flush=True)
        return pf

    ref_pf = run(V3, make())
    ref = summarise(ref_pf).iloc[0]
    refm = ref_pf.set_index("fold")["MAPE (%)"]

    def board(labels: list[str]) -> pd.DataFrame:
        pf = pd.concat([res[n][0] for n in labels])
        s = summarise(pf)
        s["Δ MAPE vs v3"] = s["MAPE (%)"] - ref["MAPE (%)"]
        s["Δ RMSE vs v3"] = s["RMSE ($)"] - ref["RMSE ($)"]
        s["folds better"] = [int((res[n][0].set_index("fold")["MAPE (%)"] < refm - 1e-9).sum()) for n in s.index]
        s["clear gain"] = (s["Δ MAPE vs v3"] <= -0.02) & (s["Δ RMSE vs v3"] <= -1.0)
        return s

    sections: dict[str, list[str]] = {}

    # 1. fitting to the metric ------------------------------------------------------------------------------------
    rep.h2("1. Fitting to the metric")
    rep.p("v3 is fitted on log price, which treats a 2% miss on a $300 load like a 2% miss on a $5,000 load (a MAPE-like target). "
          "If the hidden metric is RMSE in dollars, long expensive loads dominate it, so weighting them more could help RMSE at some "
          "cost in MAPE. Weights use a first-pass fitted price (not the observed price, which would bias the fit upward). "
          "The bias corrections shift every prediction by the amount that is optimal for each metric if errors are log-normal.")
    labels = [V3]
    for p in (1, 2):
        labels.append(run(f"weight ∝ fitted price^{p}", make(price_power=p))["model"].iloc[0])
    labels.append(run("bias correction for RMSE (+σ²/2)", make(adjust="mean"))["model"].iloc[0])
    labels.append(run("bias correction for MAPE (−σ²)", make(adjust="mape"))["model"].iloc[0])
    labels.append(run("weight ∝ price² + RMSE correction", make(price_power=2, adjust="mean"))["model"].iloc[0])
    sections["metric"] = labels[1:]
    rep.table(board(labels)[SHOW], digits=3)
    s2 = np.mean([StructuralX(coords=coords, **v3).fit(f["train"]).sigma2_ for f in fd.folds])
    rep.p(f"In-sample residual σ ≈ {100 * np.sqrt(s2):.2f}% of price, so the corrections move predictions by only "
          f"+{100 * s2 / 2:.3f}% (RMSE) and −{100 * s2:.3f}% (MAPE).")

    # 2. recency ----------------------------------------------------------------------------------------------------
    rep.h2("2. Recency: weight recent months more")
    rep.p("We predict ahead, so recent months may describe Nov–Dec better. Half-life = months after which a load counts half. "
          "Windows keep only the last N months (drift and city effects then rest on fewer loads).")
    labels = [V3]
    for h in (2, 4, 8):
        labels.append(run(f"recency half-life {h} months", make(half_life=h))["model"].iloc[0])
    for wdw in (3, 5):
        labels.append(run(f"train on last {wdw} months only", make(window=wdw))["model"].iloc[0])
    sections["recency"] = labels[1:]
    rep.table(board(labels)[SHOW], digits=3)

    # 3. lane estimation --------------------------------------------------------------------------------------------
    rep.h2("3. Lane effects: estimated jointly, and per equipment")
    rep.p("v3 fits the main effects first, then gives each lane its shrunk mean leftover (two steps). Fitting both together "
          "(backfitting until stable, the textbook mixed-model solution) stops city effects from soaking up lane effects. "
          "Lane × equipment adds a further shrunk level per lane and truck type.")
    labels = [V3]
    for lam in (2, 3, 5):
        labels.append(run(f"joint lane fit, λ={lam}", make(joint_iters=15, lane_shrink=lam))["model"].iloc[0])
    for le in (3, 10):
        labels.append(run(f"+ lane × equipment, λ={le}", make(lane_equip_shrink=le))["model"].iloc[0])
    sections["lanes"] = labels[1:]
    rep.table(board(labels)[SHOW], digits=3)

    # 4. neighbour-lane prior ---------------------------------------------------------------------------------------
    rep.h2("4. Neighbour lanes: borrowing from nearby routes")
    tr_lanes = set(train.pickup + "→" + train.delivery)
    tr_cities = set(train.pickup) | set(train.delivery)
    vl = val.pickup + "→" + val.delivery
    new_city = ~(val.pickup.isin(tr_cities) & val.delivery.isin(tr_cities))
    new_lane = ~vl.isin(tr_lanes)
    rep.p(f"In the real validation file, {100 * new_lane.mean():.1f}% of loads are on a route never seen in Jan–Oct: "
          f"{100 * (new_lane & new_city).mean():.1f}% because a city is new, {100 * (new_lane & ~new_city).mean():.1f}% between two known cities. "
          "v3 gives those routes a lane effect of 0. Here a route's prior is the kernel-weighted leftover of routes whose pickup is near its "
          "pickup and delivery near its delivery (itself excluded); 'all' shrinks every lane toward its prior instead of 0, "
          "'unseen' uses the prior only for routes with no training loads. This only helps if lane effects are smooth in geography.")
    labels = [V3]
    for bw in (50, 100, 200, 400):
        for ps in (5, 50):
            labels.append(run(f"prior bw {bw} mi, shrink {ps}, all lanes", make(prior_bw=bw, prior_shrink=ps))["model"].iloc[0])
    b4 = board(labels)
    best_prior = b4.drop(V3)["MAPE (%)"].idxmin()
    bw_best = int(best_prior.split("bw ")[1].split(" mi")[0])
    ps_best = int(best_prior.split("shrink ")[1].split(",")[0])
    labels.append(run(f"prior bw {bw_best} mi, shrink {ps_best}, unseen routes only",
                      make(prior_bw=bw_best, prior_shrink=ps_best, prior_scope="unseen"))["model"].iloc[0])
    sections["prior"] = labels[1:]
    rep.table(board(labels)[SHOW], digits=3)
    # score on the routes the prior is meant for
    rows = []
    for n in (V3, best_prior):
        for kind in ("seen route", "new route, known cities", "new route, unseen city"):
            ys, ps_ = [], []
            for f, p in zip(fd.folds, res[n][1]):
                te = f["test"]
                seen_route = (te.pickup + "→" + te.delivery).isin(set(f["train"].pickup + "→" + f["train"].delivery)).values
                m = f["genuine"] & {"seen route": seen_route, "new route, known cities": ~seen_route & ~f["unseen"],
                                    "new route, unseen city": ~seen_route & f["unseen"]}[kind]
                ys.append(te["posted_rate"].values[m])
                ps_.append(p[m])
            y_, p_ = np.concatenate(ys), np.concatenate(ps_)
            rows.append({"model": n, "routes": kind, "test loads": len(y_), "MAPE (%)": 100 * np.mean(np.abs(p_ - y_) / y_),
                         "RMSE ($)": float(np.sqrt(np.mean((p_ - y_) ** 2)))})
    rep.p("By route type (all folds pooled):")
    rep.table(pd.DataFrame(rows), index=False, digits=3)

    # 5. combination ---------------------------------------------------------------------------------------------------
    rep.h2("5. Combining what helped")
    everything = [n for sec in sections.values() for n in sec]
    full = board([V3] + everything)
    passed = full.index[full["clear gain"]].tolist()
    helped = {sec: board(labs).query("`Δ MAPE vs v3` < 0 and `Δ RMSE vs v3` < 0")["MAPE (%)"].idxmin()
              for sec, labs in sections.items() if len(board(labs).query("`Δ MAPE vs v3` < 0 and `Δ RMSE vs v3` < 0"))}
    rep.p("Clear-gain passes on their own: " + (", ".join(f"**{n}**" for n in passed) if passed else "**none**") + ". "
          "Best variant per section that improved both MAPE and RMSE at all: " + ("; ".join(helped.values()) if helped else "none") + ".")
    cfg = {}
    spec = {"weight ∝ fitted price^1": {"price_power": 1}, "weight ∝ fitted price^2": {"price_power": 2},
            "bias correction for RMSE (+σ²/2)": {"adjust": "mean"}, "bias correction for MAPE (−σ²)": {"adjust": "mape"},
            "weight ∝ price² + RMSE correction": {"price_power": 2, "adjust": "mean"}}
    for h in (2, 4, 8):
        spec[f"recency half-life {h} months"] = {"half_life": h}
    for wdw in (3, 5):
        spec[f"train on last {wdw} months only"] = {"window": wdw}
    for lam in (2, 3, 5):
        spec[f"joint lane fit, λ={lam}"] = {"joint_iters": 15, "lane_shrink": lam}
    for le in (3, 10):
        spec[f"+ lane × equipment, λ={le}"] = {"lane_equip_shrink": le}
    for n in sections["prior"]:
        bw = int(n.split("bw ")[1].split(" mi")[0])
        ps = int(n.split("shrink ")[1].split(",")[0])
        spec[n] = {"prior_bw": bw, "prior_shrink": ps, "prior_scope": "unseen" if "unseen" in n else "all"}
    for n in helped.values():
        cfg.update(spec[n])
    combo_label = None
    if len(helped) >= 2:
        combo_label = "combined: " + " + ".join(helped.values())
        run(combo_label, make(**cfg))
        cb = board([V3, combo_label])
        rep.table(cb[SHOW], digits=3)
    final_name = V3
    final_extra: dict = {}
    cand = board([V3] + everything + ([combo_label] if combo_label else []))
    winners = cand[cand["clear gain"]]
    if len(winners):
        final_name = winners["MAPE (%)"].idxmin()
        final_extra = cfg if final_name == combo_label else spec[final_name]
    rep.p(f"Decision: **{final_name}**" + (" stays." if final_name == V3 else " replaces v3."))
    best_other = cand.drop(V3)["MAPE (%)"].idxmin()

    # 6. robustness ------------------------------------------------------------------------------------------------------
    rep.h2("6. Robustness")
    rep.h3("6a. Other random choices of hidden cities")
    rep.p("Every fold hides 4 random cities from training. The comparison above used one draw; here are three more.")
    contenders = {"structural v2 (no lane effect)": lambda: StructuralX(coords=coords, **{**v3, "lane_shrink": None}),
                  V3: make(), "hybrid on v3 (trees on residuals)": lambda: HybridModel(StructuralModel(coords=coords, **v3), params=hp),
                  best_other: make(**(cfg if best_other == combo_label else spec[best_other]))}
    seed_rows = []
    for seed in (0, 100, 200, 300):
        fds = fd if seed == 0 else FoldData(train, val, seed=seed)
        for n, fac in contenders.items():
            pf = res[n][0] if seed == 0 and n in res else fds.evaluate(fac, n)[0]
            seed_rows.append({"hidden-city draw": seed, "model": n, "MAPE (%)": pf["MAPE (%)"].mean(), "RMSE ($)": pf["RMSE ($)"].mean(),
                              "MAPE unseen cities (%)": pf["MAPE unseen cities (%)"].mean()})
        print(f"  seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    sr = pd.DataFrame(seed_rows)
    for metric in ("MAPE (%)", "RMSE ($)", "MAPE unseen cities (%)"):
        tab = sr.pivot(index="model", columns="hidden-city draw", values=metric)[[0, 100, 200, 300]]
        tab["mean"], tab["spread (max−min)"] = tab.mean(axis=1), tab.max(axis=1) - tab.min(axis=1)
        rep.p(f"{metric} by hidden-city draw:")
        rep.table(tab.sort_values("mean"), digits=3)
    rank = sr.pivot(index="hidden-city draw", columns="model", values="MAPE (%)").rank(axis=1)
    rep.p("Rank of each model per draw (1 = best MAPE): " + "; ".join(f"{m}: {', '.join(str(int(r)) for r in rank[m])}" for m in rank.columns) + ".")

    rep.h3("6b. Other cut-offs for removing corrupted prices")
    rep.p("Only the training part of each fold changes; the scored test rows stay the same. 'Removed' counts rows flagged in all of Jan–Oct.")
    dmi_all = daily_market_index(train, val)
    full_clean = clean_inputs(train, dmi_all, weight_fill_values(train))
    variants = {
        "default: lane 0.5–2.0, model ±0.4": None,
        "looser lane 0.4–2.5, model ±0.4": lambda d: flag_corrupted(d, (0.4, 2.5), 0.4),
        "stricter lane 0.6–1.67, model ±0.4": lambda d: flag_corrupted(d, (0.6, 1 / 0.6), 0.4),
        "lane 0.5–2.0, stricter model ±0.3": lambda d: flag_corrupted(d, (0.5, 2.0), 0.3),
        "lane 0.5–2.0, looser model ±0.5": lambda d: flag_corrupted(d, (0.5, 2.0), 0.5),
        "pass 1 only (lane 0.5–2.0)": lambda d: flag_corrupted(d, (0.5, 2.0), None),
        "no cleaning": lambda d: pd.Series(False, index=d.index),
    }
    clean_rows = []
    for n, fn in variants.items():
        removed = int((flag_corrupted(full_clean) if fn is None else fn(full_clean)).sum())
        pf = ref_pf if fn is None else FoldData(train, val, flag_fn=fn).evaluate(make(), n)[0]
        s = summarise(pf.assign(model=n)).iloc[0]
        clean_rows.append({"rule": n, "removed": removed, "MAPE (%)": s["MAPE (%)"], "RMSE ($)": s["RMSE ($)"],
                           "worst-fold MAPE (%)": s["worst-fold MAPE (%)"], "bias (%)": s["bias (%)"]})
        print(f"  cleaning {n} done ({time.time() - t0:.0f}s)", flush=True)
    rep.table(pd.DataFrame(clean_rows), index=False, digits=3)

    rep.h3("6c. The drift assumption: what if prices stop rising, or keep rising?")
    rep.p("v3 continues the slow upward drift at half speed beyond the last training day ('damped'), a hedge chosen because folds split "
          "2–2 between stopping ('freeze') and continuing at full speed. Per fold, with v3:")
    tr_rows = []
    for tr in ("freeze", "damped", "continue"):
        pf = ref_pf if tr == v3["trend"] else fd.evaluate(make(trend=tr), f"trend {tr}")[0]
        tr_rows.append(pf.assign(model=f"trend {tr}"))
    tp = pd.concat(tr_rows)
    tt = tp.pivot(index="fold", columns="model", values="MAPE (%)")
    tt.loc["mean"] = tt.mean()
    rep.table(tt, digits=3)
    rr = tp.pivot(index="fold", columns="model", values="RMSE ($)")
    rr.loc["mean"] = rr.mean()
    rep.p("RMSE ($) per fold:")
    rep.table(rr, digits=2)

    # Nov–Dec sensitivity with the final model fitted on all of Jan–Oct
    wf = weight_fill_values(train)
    c = full_clean[~flag_corrupted(full_clean)]
    m = StructuralX(coords=coords, **{**v3, **final_extra}).fit(c)
    v = clean_inputs(val, dmi_all, wf)
    d = clean_inputs(dec, dmi_all, wf, coords=coords)
    preds = {}
    for tr in ("freeze", "damped", "continue"):
        m.trend = tr
        preds[tr] = (m.predict(v), m.predict(d))
    m.trend = v3["trend"]
    sens = []
    for tr in ("freeze", "continue"):
        r = preds[tr][0] / preds["damped"][0] - 1
        nov = (v.date.dt.month == 11).values
        sens.append({"assumption": tr, "Nov: median vs damped (%)": 100 * np.median(r[nov]),
                     "Dec: median vs damped (%)": 100 * np.median(r[~nov]),
                     "Dec 31 chart price vs damped (%)": 100 * (preds[tr][1][-1] / preds["damped"][1][-1] - 1)})
    rep.p("Nov–Dec predictions (model fitted on all of Jan–Oct) under each assumption, relative to the chosen 'damped':")
    rep.table(pd.DataFrame(sens), index=False, digits=2)
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    for tr, col in zip(("freeze", "damped", "continue"), (BLUE, ORANGE, AQUA)):
        ax.plot(d.date, preds[tr][1], color=col, marker="o", ms=3, label=f"{tr}{' (chosen)' if tr == v3['trend'] else ''}")
        ax.annotate(f"{tr} ${preds[tr][1][-1]:.0f}", (d.date.iloc[-1], preds[tr][1][-1]), xytext=(6, 0), textcoords="offset points",
                    fontsize=7.5, color=INK_2, va="center")
    ax.set(title="December chart price under each drift assumption (Lexington → Fort Wayne, Dry Van)", ylabel="predicted rate ($)")
    ax.legend(loc="upper left", ncol=3)
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 8, 15, 22, 29]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_xlim(d.date.min() - np.timedelta64(1, "D"), d.date.max() + np.timedelta64(5, "D"))
    gap0 = 100 * (preds["continue"][1][0] / preds["damped"][1][0] - 1)
    gap1 = 100 * (preds["continue"][1][-1] / preds["damped"][1][-1] - 1)
    rep.figure(fig, "december_drift", f"The assumptions differ by about ±{gap0:.1f}% on Dec 1, widening to ±{gap1:.1f}% by Dec 31. The weekly "
               "zig-zag and the quarter-end ramp are the same in all three.")

    # summary ----------------------------------------------------------------------------------------------------------
    rep.h2("7. Summary")
    top = cand.drop(V3).sort_values("MAPE (%)").head(6)
    rep.table(pd.concat([cand.loc[[V3]], top])[SHOW], digits=3)
    rep.save()
    pd.to_pickle({"board": cand, "final_name": final_name, "final_params": {**v3, **final_extra}, "seeds": sr,
                  "cleaning": pd.DataFrame(clean_rows), "trend_folds": tt, "sensitivity": pd.DataFrame(sens)},
                 ROOT / "outputs" / "loose_ends.pkl")
    print(f"decision: {final_name}  done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
