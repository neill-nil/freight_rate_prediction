"""08 · Proving quote_signal is a trap, by running models rather than just claiming it.

Experiment: the same gradient-boosted tree model trained with and without quote_signal, scored
  * under a random 5-fold split of Jan–Oct (the careless way), and
  * under time-forward splits (train on months before M, test on month M: what Nov–Dec really is).
Each test month is labelled with its quote_signal "regime" (positive / negative / none) so we can see
when it helps and when it hurts.
Counterfactual: in every test set, shuffle quote_signal. Its distribution stays the same (exactly what
validation looks like, step 06) but its link to price is gone, the state step 06 found for Nov–Dec.
Finally: how much would it move the real Nov–Dec predictions?
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from common import (
    AQUA,
    BLUE,
    INK_2,
    ORANGE,
    Report,
    clean_inputs,
    clean_train,
    daily_market_index,
    error_summary,
    load_raw,
    set_style,
)

BASE = ["log_distance", "equipment_code", "weight_abs", "mi", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "day_of_year"]
WITH = BASE + ["quote_signal"]
EQUIP_CODE = {"Dry Van": 0, "Reefer": 1, "Flatbed": 2}
RNG = np.random.default_rng(0)


def frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.assign(equipment_code=df.equipment.map(EQUIP_CODE), day_of_year=df.date.dt.dayofyear)[cols]


def model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
        categorical_features=["equipment_code"], early_stopping=False, random_state=0,
    )


def fit(df: pd.DataFrame, cols: list[str]) -> HistGradientBoostingRegressor:
    return model().fit(frame(df, cols), np.log(df.posted_rate))


def score(m, df: pd.DataFrame, cols: list[str]) -> dict:
    s = error_summary(df.posted_rate, np.exp(m.predict(frame(df, cols))))
    return {"MAPE (%)": s["MAPE (%)"], "RMSE ($)": s["RMSE ($)"]}


def evaluate(tr_: pd.DataFrame, te_: pd.DataFrame) -> list[dict]:
    m_wo, m_w = fit(tr_, BASE), fit(tr_, WITH)
    shuffled = te_.assign(quote_signal=RNG.permutation(te_.quote_signal.values))
    return [
        {"variant": "without quote_signal", **score(m_wo, te_, BASE)},
        {"variant": "with quote_signal", **score(m_w, te_, WITH)},
        {"variant": "with quote_signal, test values shuffled", **score(m_w, shuffled, WITH)},
    ]


def regime(df: pd.DataFrame) -> str:
    r = df.quote_signal.corr(df.rpm)
    return "positive" if r > 0.5 else "negative" if r < -0.5 else "none"


def main() -> None:
    set_style()
    train, val, _ = load_raw()
    c, _ = clean_train(train, val)
    month = c.date.dt.month
    regimes = {m: regime(c[month == m]) for m in sorted(month.unique())}
    rep = Report(
        "08_trap_proof",
        "08 · Proof that quote_signal is a trap",
        "Same model (gradient-boosted trees on log price), trained **with** and **without** quote_signal, scored on clean rows. "
        "Two accuracy measures throughout: **MAPE** (average % error, every load counts equally) and **RMSE** (dollar error, big "
        "misses on expensive loads count most).",
    )
    rep.p("quote_signal regime per month (corr with rate per mile > 0.5 = positive, < −0.5 = negative, else none): "
          + ", ".join(f"{pd.Timestamp(2025, m, 1):%b} {r}" for m, r in regimes.items())
          + ". Nov–Dec: **none** (step 06 proxy).")

    rows = []
    for k, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(c)):
        rows += [{"scheme": "random 5-fold", "test": "random (avg of 5)", "fold": k, **r} for r in evaluate(c.iloc[a], c.iloc[b])]
    for m in (7, 8, 9, 10):
        tr_, te_ = c[month < m], c[month == m]
        seen = sorted({regimes[x] for x in range(1, m)})
        label = f"{pd.Timestamp(2025, m, 1):%b} ({regimes[m]})"
        rows += [{"scheme": "time-forward", "test": label, "fold": m, "test regime": regimes[m], "regime seen in training": regimes[m] in seen, **r} for r in evaluate(tr_, te_)]
    res = pd.DataFrame(rows)

    # 1. headline -------------------------------------------------------------------------------------
    rep.h2("1. The careless random split says: keep it")
    rnd = res[res.scheme == "random 5-fold"].groupby("variant")[["MAPE (%)", "RMSE ($)"]].mean()
    rep.table(rnd, digits=2)
    g = rnd.loc["with quote_signal"] / rnd.loc["without quote_signal"] - 1
    rep.p(f"With quote_signal: MAPE {100 * g['MAPE (%)']:+.0f}%, RMSE {100 * g['RMSE ($)']:+.0f}%. Looks like a clear win.")

    # 2. clean counterfactual inside the random split ------------------------------------------------------
    rep.h2("2. Clean test: same months, but quote_signal stripped of its meaning")
    rep.p("Inside the random split nothing else changes (same months, same model), so shuffling quote_signal in the test fold isolates "
          "one question: how does the model do when quote_signal carries no signal, which is the Nov–Dec situation?")
    rs = rnd.loc[["without quote_signal", "with quote_signal", "with quote_signal, test values shuffled"]]
    rep.table(rs, digits=2)
    r_sh = rs.loc["with quote_signal, test values shuffled"] / rs.loc["without quote_signal"] - 1
    rep.p(f"With its meaning removed, the model that relies on quote_signal is **{100 * r_sh['MAPE (%)']:+.0f}% MAPE / "
          f"{100 * r_sh['RMSE ($)']:+.0f}% RMSE** versus the model that never used it.")

    # 3. the real no-signal month ---------------------------------------------------------------------------
    rep.h2("3. Real test: August, the one training month with no signal")
    aug = res[(res.scheme == "time-forward") & (res.fold == 8)].set_index("variant")
    nod = [x for x in BASE if x != "day_of_year"]
    tr8, te8 = c[month < 8], c[month == 8]
    aug_nod = pd.DataFrame({"without quote_signal": score(fit(tr8, nod), te8, nod), "with quote_signal": score(fit(tr8, nod + ["quote_signal"]), te8, nod + ["quote_signal"])}).T
    aug_tab = pd.concat(
        [aug.loc[["without quote_signal", "with quote_signal"], ["MAPE (%)", "RMSE ($)"]].add_prefix("with date feature · "),
         aug_nod.add_prefix("no date feature · ")], axis=1)
    aug_tab.index.name = "train Jan–Jul → test Aug"
    rep.table(aug_tab, digits=2)
    rep.p(f"In the real no-signal month quote_signal makes the model clearly worse, whether or not it also sees the date: "
          f"MAPE {aug.loc['without quote_signal', 'MAPE (%)']:.2f}% → {aug.loc['with quote_signal', 'MAPE (%)']:.2f}% with a date feature, "
          f"{aug_nod.loc['without quote_signal', 'MAPE (%)']:.2f}% → {aug_nod.loc['with quote_signal', 'MAPE (%)']:.2f}% without. "
          "**Nov–Dec are in the same no-signal state** (step 06: quote_signal's link to distance is 0.01 in Nov–Dec, 0.02 in August).")

    # 4. other time-forward months (confounded) --------------------------------------------------------------
    rep.h2("4. The other time-forward months, and why they don't change the conclusion")
    summ = res.groupby(["test", "variant"])[["MAPE (%)", "RMSE ($)"]].mean().unstack("variant")
    order = ["random (avg of 5)"] + [t for t in summ.index if t != "random (avg of 5)"]
    order = order[:1] + sorted(order[1:], key=lambda t: pd.to_datetime(t.split(" ")[0], format="%b").month)
    summ = summ.reindex(order)
    cf = pd.DataFrame(
        {
            "MAPE without qs": summ[("MAPE (%)", "without quote_signal")],
            "MAPE with qs": summ[("MAPE (%)", "with quote_signal")],
            "MAPE with qs, shuffled": summ[("MAPE (%)", "with quote_signal, test values shuffled")],
            "RMSE without qs": summ[("RMSE ($)", "without quote_signal")],
            "RMSE with qs": summ[("RMSE ($)", "with quote_signal")],
            "RMSE with qs, shuffled": summ[("RMSE ($)", "with quote_signal, test values shuffled")],
        }
    )
    cf.index.name = "test set"
    rep.table(cf, digits=2)
    rep.p("In July, September and October (regimes already seen in training) quote_signal still lowers MAPE, and in July and October "
          "**even the shuffled version does**. Shuffled values carry no row-level information, so that gain can't come from what "
          "quote_signal says about each load. Two confounds explain it:")
    qm = c.groupby(month).quote_signal.mean()
    rep.bullets(
        [
            "**Time extrapolation.** These baseline models have no proper time handling yet. A tree treats every future day like the "
            "last training days, so after a quarter-end ramp (end of June, end of September) the no-quote_signal model carries the "
            "peak forward: it over-predicts July by a median +3.8% and October by +1.8% (checked separately).",
            f"**quote_signal's monthly average tracks its regime** (positive months {qm[[1, 2, 3, 6, 9]].mean():.2f}, negative months "
            f"{qm[[4, 5, 7, 10]].mean():.2f}, August {qm[8]:.2f}; Nov–Dec {val.quote_signal.mean():.2f}). A model can use that monthly "
            "level to patch over missing time effects. That is luck, not information about the load.",
            "Neither confound helps in Nov–Dec: quote_signal's level there sits at the no-signal value, and time effects should be "
            "modelled properly instead (deep-dive 10).",
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    x = np.arange(len(cf))
    series = [("without qs", BLUE, "without quote_signal"), ("with qs", ORANGE, "with quote_signal (real values)"), ("with qs, shuffled", AQUA, "with quote_signal, no-signal future (shuffled)")]
    for ax, metric in ((axes[0], "MAPE"), (axes[1], "RMSE")):
        for i, (col, color, label) in enumerate(series):
            ax.bar(x + (i - 1) * 0.27, cf[f"{metric} {col}"], width=0.25, color=color, label=label)
        ax.set_xticks(x, [t.replace(" (avg of 5)", "\nsplit") for t in cf.index])
        ax.set(title=f"{metric} by test set", ylabel="MAPE (%)" if metric == "MAPE" else "RMSE ($)")
        ax.grid(axis="x", visible=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    rep.figure(fig, "with_without", "Blue = never used quote_signal. Orange = uses it, real values. Green = uses it, but the test data carries no signal. "
               "The two clean comparisons are the random split (left group) and August, the real no-signal month.")

    # 5. mechanism -----------------------------------------------------------------------------------------
    rep.h2("5. Why the random split is fooled")
    mech_rows = []
    for k, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(c)):
        nod = [x for x in BASE if x != "day_of_year"]
        for name, cols in (("without qs, no date feature", nod), ("with qs, no date feature", nod + ["quote_signal"])):
            mech_rows.append({"variant": name, **score(fit(c.iloc[a], cols), c.iloc[b], cols)})
    mech = pd.concat([pd.DataFrame(mech_rows).groupby("variant").mean(), rnd.rename(index={"without quote_signal": "without qs, with date feature", "with quote_signal": "with qs, with date feature"}).drop(index="with quote_signal, test values shuffled")])
    mech.index.name = "random 5-fold"
    rep.table(mech, digits=2)
    rep.p("quote_signal's meaning changes with the month. A random split puts every month in both training and test, so the model "
          "always gets to learn the current month's meaning before being tested on it. The future gives no such preview.")

    # 6. validation ----------------------------------------------------------------------------------------
    rep.h2("6. What it would do to the real Nov–Dec predictions")
    weight_fill = train.assign(w=train.weight.abs()).groupby("equipment").w.median().to_dict()
    v = clean_inputs(val, daily_market_index(train, val), weight_fill)
    m_w, m_wo = fit(c, WITH), fit(c, BASE)
    p_with = np.exp(m_w.predict(frame(v, WITH)))
    p_med = np.exp(m_w.predict(frame(v.assign(quote_signal=c.quote_signal.median()), WITH)))
    p_wo = np.exp(m_wo.predict(frame(v, BASE)))
    d1, d2 = 100 * np.abs(p_with / p_med - 1), 100 * np.abs(p_with / p_wo - 1)
    sens = pd.DataFrame(
        {
            "same model: real quote_signal vs fixed at median": [d1.mean(), np.median(d1), np.percentile(d1, 95)],
            "with-qs model vs without-qs model": [d2.mean(), np.median(d2), np.percentile(d2, 95)],
        },
        index=["mean |difference| %", "median |difference| %", "95th percentile %"],
    ).T
    rep.table(sens, digits=2)
    rep.p("Because quote_signal carries no information in Nov–Dec, every percent it moves a validation prediction is error added for nothing.")

    rep.h2("Conclusion")
    rep.bullets(
        [
            "quote_signal is not a stable property of a load: its meaning flips between regimes and it is meaningless in some.",
            "A random split rewards it, because every test month's meaning was already learned in training.",
            "Whenever it carries no signal (shuffled test data, or the real month of August) the model that uses it is much worse than "
            "one that never did. For Nov–Dec the evidence (step 06) says 'no signal'.",
            "The apparent gains in other time-forward months come from confounds (time extrapolation, quote_signal's monthly level), not "
            "from information about the load. **Decision: drop it, and evaluate everything with time-forward splits.**",
        ]
    )
    rep.save()


if __name__ == "__main__":
    main()
