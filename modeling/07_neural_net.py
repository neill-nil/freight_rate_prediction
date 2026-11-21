"""Modelling step 7: a neural network (multi-layer perceptron), standalone and stacked on v3.

Same leak-free features and folds as steps 5–6. scikit-learn MLPRegressor: inputs standardised, target
(log price per mile) standardised, Adam optimiser, early stopping on a 10% slice of the training rows.
  * standalone MLP on all 20 features (can it learn the structure on its own?)
  * residual MLP: learns only what structural v3 leaves over (like the residual stacks in step 6)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import TransformedTargetRegressor  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from freight.data import load_train, load_validation  # noqa: E402
from freight.evaluation import FoldData, summarise  # noqa: E402
from freight.reporting import Report, set_style  # noqa: E402
from freight.zoo import BASE_FEATURES, LEVEL_FEATURES, STRUCT_FEATURE, evaluate_estimator  # noqa: E402

COLS = ["MAPE (%)", "RMSE ($)", "MAE ($)", "MAPE seen cities (%)", "MAPE unseen cities (%)", "MAPE month 2 (%)", "bias (%)", "worst-fold MAPE (%)"]
REF = "Structural v3 (reference)"
ALL_COLS = BASE_FEATURES + LEVEL_FEATURES


def mlp(hidden: tuple, alpha: float, lr: float):
    net = MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha, learning_rate_init=lr, batch_size=256, max_iter=200,
                       early_stopping=True, validation_fraction=0.1, n_iter_no_change=12, random_state=0)
    return TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), net), transformer=StandardScaler())


STANDALONE = [((64, 32), 1e-4, 1e-3), ((128, 64), 1e-4, 1e-3), ((128, 64), 1e-3, 1e-3), ((256, 128), 1e-4, 5e-4), ((128, 64, 32), 1e-4, 1e-3)]
RESIDUAL = [((32, 16), 1e-3, 1e-3), ((64, 32), 1e-3, 1e-3), ((64, 32), 1e-2, 1e-3)]


def main() -> None:
    set_style()
    t0 = time.time()
    train, val = load_train(), load_validation()
    fd = FoldData(train, val)
    mats = pd.read_pickle(ROOT / "outputs" / "fold_matrices.pkl")
    ens = pd.read_pickle(ROOT / "outputs" / "ensemble_preds.pkl")
    rep = Report("07_neural_net", "Modelling 07 · Neural network (multi-layer perceptron)",
                 "Same features, folds and scoring as Modelling 05–06. scikit-learn MLPRegressor with standardised inputs and target, Adam, "
                 "early stopping on 10% of the training rows.")
    rows = [fd.score_predictions(ens["preds"][REF], REF)]
    tables = {}
    preds = {}
    for kind, configs, cols, residual in (("standalone MLP", STANDALONE, ALL_COLS, False), ("v3 + MLP on residuals", RESIDUAL, ALL_COLS + STRUCT_FEATURE, True)):
        tab, best = [], (np.inf, None, None, None)
        for hidden, alpha, lr in configs:
            pf, pr = evaluate_estimator(fd, mats, lambda h=hidden, a=alpha, l=lr: mlp(h, a, l), kind, cols, residual=residual)
            s = summarise(pf).iloc[0]
            tab.append({"hidden layers": str(hidden), "L2 (alpha)": alpha, "learning rate": lr, "MAPE (%)": s["MAPE (%)"], "RMSE ($)": s["RMSE ($)"], "worst-fold MAPE (%)": s["worst-fold MAPE (%)"]})
            if s["MAPE (%)"] < best[0]:
                best = (s["MAPE (%)"], (hidden, alpha, lr), pf, pr)
            print(f"  {kind} {hidden} a={alpha} lr={lr}: {s['MAPE (%)']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        tables[kind] = pd.DataFrame(tab).sort_values("MAPE (%)")
        rows.append(best[2])
        preds[kind] = best[3]

    rep.h2("1. Result")
    s = summarise(pd.concat(rows)).sort_values("MAPE (%)")
    rep.table(s[COLS], digits=3)
    ref = s.loc[REF]
    beats = [n for n in s.index if n != REF and s.loc[n, "MAPE (%)"] <= ref["MAPE (%)"] - 0.02 and s.loc[n, "RMSE ($)"] <= ref["RMSE ($)"] - 1.0]
    rep.p("Beats structural v3 by the clear-gain bar: " + (", ".join(beats) if beats else "**no**") + ".")
    rep.p("MAPE per fold:")
    rep.table(pd.concat(rows).pivot(index="fold", columns="model", values="MAPE (%)"), digits=3)
    rep.h2("2. Tuning")
    for kind, tab in tables.items():
        rep.h3(kind)
        rep.table(tab, index=False, digits=4)
    rep.save()
    pd.to_pickle({"preds": preds, "summary": s}, ROOT / "outputs" / "nn_preds.pkl")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
