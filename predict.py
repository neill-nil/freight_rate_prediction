"""Final pipeline: train the chosen model on all of Jan–Oct and write the two deliverable files.

    python predict.py              # writes validation_predictions.csv, fills data/december_chart_inputs.csv, runs score.py
    python predict.py --no-score   # skip the scorer run

Steps
1. load data/train_test.csv, data/validation.csv, data/december_chart_inputs.csv
2. clean inputs the same way for every file (weight sign flips and blanks, blank market_index → that day's average,
   December coordinates from the city lookup); market_index for December = each day's average in validation.csv
3. remove corrupted training prices (two-pass rule)
4. fit the final model (src/freight/config.py) on the cleaned Jan–Oct loads
5. predict the 12,000 validation loads → validation_predictions.csv (template order)
6. predict the 31 December rows → data/december_chart_inputs.csv (only the predicted_rate column is filled; inputs untouched)
7. run score.py → scorer_results/candidate_december.png
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from freight.config import FINAL_MODEL, HYBRID_TREES, STRUCTURAL_V3  # noqa: E402
from freight.data import (  # noqa: E402
    DATA_DIR, city_coordinates, clean_inputs, daily_market_index, flag_corrupted, load_december, load_train, load_validation,
    weight_fill_values,
)
from freight.models import HybridModel, StructuralModel  # noqa: E402

PREDICTIONS = ROOT / "validation_predictions.csv"
TEMPLATE = DATA_DIR / "validation_predictions_template.csv"
DECEMBER = DATA_DIR / "december_chart_inputs.csv"


def build_model(coords: pd.DataFrame):
    structural = StructuralModel(coords=coords, **STRUCTURAL_V3)
    if FINAL_MODEL == "structural_v3":
        return structural
    if FINAL_MODEL == "hybrid_v3":
        return HybridModel(structural, params=HYBRID_TREES)
    raise ValueError(f"unknown FINAL_MODEL {FINAL_MODEL!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-score", action="store_true", help="do not run score.py at the end")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # keep our lines in order with the scorer's output

    # 1–2. load and clean
    train, val, dec = load_train(), load_validation(), load_december()
    coords = city_coordinates(train, val)
    daily_mi = daily_market_index(train, val)
    weight_fill = weight_fill_values(train)
    tr = clean_inputs(train, daily_mi, weight_fill)
    va = clean_inputs(val, daily_mi, weight_fill)
    de = clean_inputs(dec, daily_mi, weight_fill, coords=coords)

    # 3. corrupted training prices
    bad = flag_corrupted(tr)
    tr = tr[~bad]
    print(f"training loads: {len(train):,}; corrupted prices removed: {int(bad.sum())}; used: {len(tr):,}")

    # 4. fit
    model = build_model(coords).fit(tr)
    print(f"model: {FINAL_MODEL}")

    # 5. validation predictions, in the template's order
    pred = pd.Series(model.predict(va), index=val["load_id"].values)
    template = pd.read_csv(TEMPLATE)
    out = pd.DataFrame({"load_id": template["load_id"], "predicted_rate": pred.reindex(template["load_id"]).values.round(2)})
    assert len(out) == 12_000 and out["load_id"].is_unique, "unexpected validation rows"
    assert np.isfinite(out["predicted_rate"]).all() and (out["predicted_rate"] > 0).all(), "non-positive or missing predictions"
    out.to_csv(PREDICTIONS, index=False)
    rpm = out["predicted_rate"].values / val.set_index("load_id").loc[out["load_id"], "distance"].values
    print(f"wrote {PREDICTIONS.name}: {len(out):,} rows, ${out.predicted_rate.min():,.0f}–${out.predicted_rate.max():,.0f}, "
          f"median ${np.median(rpm):.3f}/mile")

    # 6. December chart file: fill only predicted_rate, keep the seven original columns as they are
    raw = pd.read_csv(DECEMBER, dtype=str)
    order = pd.to_datetime(raw["date"])
    dec_pred = pd.Series(model.predict(de), index=de["date"].values)
    raw["predicted_rate"] = dec_pred.reindex(order.values).values.round(2)
    assert raw["predicted_rate"].notna().all() and (raw["predicted_rate"].astype(float) > 0).all()
    raw.to_csv(DECEMBER, index=False)
    print(f"filled {DECEMBER.relative_to(ROOT)}: Dec 1 ${raw.predicted_rate.iloc[0]:.2f} → Dec 31 ${raw.predicted_rate.iloc[-1]:.2f}")

    # 7. scorer
    if not args.no_score:
        subprocess.run([sys.executable, str(ROOT / "score.py"), "--predictions", str(PREDICTIONS),
                        "--december-predictions", str(DECEMBER), "--output-dir", str(ROOT / "scorer_results")], check=True)


if __name__ == "__main__":
    main()
