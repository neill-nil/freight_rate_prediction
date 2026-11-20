"""01 · Data audit — can we trust the files before we analyse them?

Strategy: structural checks only (schema, IDs, coverage, missingness, duplicates, impossible
values, category hygiene, coordinate consistency). Nothing is fixed here; every problem is
just counted so later steps know what they are dealing with.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import DATA_DIR, WEIGHT_CAP, Report, city_coords, load_raw, set_style

QUANTILES = [0, 0.01, 0.5, 0.99, 1]


def main() -> None:
    set_style()
    train, val, dec = load_raw()
    tmpl = pd.read_csv(DATA_DIR / "validation_predictions_template.csv")
    sets = {"train": train, "validation": val, "december": dec}
    rep = Report(
        "01_data_audit",
        "01 · Data audit",
        "Goal: confirm the files match the brief and list every structural problem before any "
        "analysis. Nothing is fixed here, only counted.",
    )

    # 1. schema ------------------------------------------------------------------------
    all_cols = list(dict.fromkeys([*train.columns, *val.columns, *dec.columns]))
    schema = pd.DataFrame(
        {n: [str(df[c].dtype) if c in df else "—" for c in all_cols] for n, df in sets.items()},
        index=pd.Index(all_cols, name="column"),
    )
    rep.h2("1. Schema")
    rep.table(schema)
    model_inputs = [c for c in val.columns if c != "load_id"]
    rep.bullets(
        [
            f"In train but not validation: `{[c for c in train.columns if c not in val.columns]}` (the target).",
            f"Model inputs **missing from the December file**: `{[c for c in model_inputs if c not in dec.columns]}`.",
        ]
    )

    # 2. size & date coverage ------------------------------------------------------------
    rows = []
    for n, df in sets.items():
        full = pd.date_range(df.date.min(), df.date.max(), freq="D")
        per_day = df.groupby("date").size()
        rows.append(
            {
                "set": n,
                "rows": len(df),
                "first date": df.date.min(),
                "last date": df.date.max(),
                "unique dates": df.date.nunique(),
                "days missing in range": len(full.difference(pd.DatetimeIndex(df.date.unique()))),
                "loads/day mean": per_day.mean(),
                "loads/day min": per_day.min(),
                "loads/day max": per_day.max(),
            }
        )
    rep.h2("2. Size and date coverage")
    rep.table(pd.DataFrame(rows), index=False, digits=1)

    # 3. ID / file-contract checks ----------------------------------------------------------
    checks = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "result": "PASS" if ok else "FAIL", "detail": detail})

    check("train load_id unique", train.load_id.is_unique)
    check("train load_id format TR-NNNNNN", bool(train.load_id.str.fullmatch(r"TR-\d{6}").all()))
    check("validation load_id unique", val.load_id.is_unique)
    check("validation load_id format TE-NNNNNN", bool(val.load_id.str.fullmatch(r"TE-\d{6}").all()))
    check(
        "validation ids == TE-000001..TE-012000 (what score.py expects)",
        set(val.load_id) == {f"TE-{i:06d}" for i in range(1, 12_001)},
    )
    check("template columns == [load_id, predicted_rate]", list(tmpl.columns) == ["load_id", "predicted_rate"])
    check("template ids == validation ids (same set)", set(tmpl.load_id) == set(val.load_id))
    check("template ids in same row order as validation", tmpl.load_id.tolist() == val.load_id.tolist())
    check("template predicted_rate all empty", bool(tmpl.predicted_rate.isna().all()))
    check(
        "December: 31 consecutive dates 2025-12-01..31",
        set(dec.date) == set(pd.date_range("2025-12-01", "2025-12-31")),
    )
    fixed = ["pickup", "delivery", "distance", "equipment", "weight"]
    check(
        "December: fixed fields constant",
        bool(dec[fixed].nunique().eq(1).all()),
        ", ".join(f"{c}={dec[c].iloc[0]}" for c in fixed),
    )
    check("December predicted_rate all empty", bool(dec.predicted_rate.isna().all()))
    rep.h2("3. ID and file-contract checks")
    rep.table(pd.DataFrame(checks), index=False)

    # 4. missing values --------------------------------------------------------------------
    cols = [c for c in train.columns if c != "load_id"]
    miss = pd.DataFrame(
        {
            "train missing": [train[c].isna().sum() for c in cols],
            "train %": [100 * train[c].isna().mean() for c in cols],
            "validation missing": [val[c].isna().sum() if c in val else np.nan for c in cols],
            "validation %": [100 * val[c].isna().mean() if c in val else np.nan for c in cols],
        },
        index=pd.Index(cols, name="column"),
    )
    rep.h2("4. Missing values")
    rep.table(miss[(miss["train missing"] > 0) | (miss["validation missing"] > 0)], digits=2)
    both = (train.weight.isna() & train.market_index.isna()).sum()
    rep.p(f"Rows in train missing **both** weight and market_index: {both} (independent missingness would give ≈{train.weight.isna().mean() * train.market_index.isna().mean() * len(train):.1f}).")

    # 5. duplicates -------------------------------------------------------------------------
    dup_rows = []
    for n, df in (("train", train), ("validation", val)):
        feats = [c for c in df.columns if c not in ("load_id", "posted_rate")]
        dup_rows.append(
            {
                "set": n,
                "exact duplicate rows (ignoring load_id)": df.drop(columns="load_id").duplicated().sum(),
                "identical inputs": df.duplicated(subset=feats).sum(),
                "same lane+date+equipment (not an error by itself)": df.duplicated(
                    subset=["pickup", "delivery", "date", "equipment"]
                ).sum(),
            }
        )
    feats = [c for c in val.columns if c != "load_id"]
    cross = train.merge(val, on=feats, how="inner")
    rep.h2("5. Duplicates")
    rep.table(pd.DataFrame(dup_rows), index=False)
    rep.p(f"Validation rows whose inputs exactly match a train row: {len(cross)}.")

    # 6. numeric ranges & impossible values -----------------------------------------------------
    num = ["distance", "weight", "market_index", "quote_signal", "posted_rate"]
    rng = {}
    for n, df in (("train", train), ("validation", val)):
        for c in num:
            if c in df:
                q = df[c].quantile(QUANTILES)
                rng[(c, n)] = {"min": q[0], "p1": q[0.01], "median": q[0.5], "p99": q[0.99], "max": q[1]}
    rng = pd.DataFrame(rng).T.sort_index()
    rng.index = [f"{c} ({n})" for c, n in rng.index]
    rep.h2("6. Numeric ranges")
    rep.table(rng, digits=3)

    imp_rules = {
        "weight < 0": lambda d: (d.weight < 0).sum(),
        "weight == 0": lambda d: (d.weight == 0).sum(),
        f"weight == {WEIGHT_CAP:,.0f} (suspected cap)": lambda d: (d.weight == WEIGHT_CAP).sum(),
        "distance <= 0": lambda d: (d.distance <= 0).sum(),
        "market_index <= 0": lambda d: (d.market_index <= 0).sum(),
        "quote_signal <= 0": lambda d: (d.quote_signal <= 0).sum(),
        "posted_rate <= 0": lambda d: (d.posted_rate <= 0).sum() if "posted_rate" in d else np.nan,
        "pickup == delivery": lambda d: (d.pickup == d.delivery).sum(),
    }
    imp = pd.DataFrame(
        {n: {k: f(df) for k, f in imp_rules.items()} for n, df in (("train", train), ("validation", val))}
    )
    imp.index.name = "rule"
    rep.h2("7. Impossible or suspicious values")
    rep.table(imp)

    # 7. categoricals ------------------------------------------------------------------------
    eq = pd.DataFrame(
        {n: df.equipment.value_counts(normalize=True).mul(100) for n, df in (("train", train), ("validation", val))}
    )
    eq.index.name = "equipment (% of rows)"
    rep.h2("8. Categorical hygiene")
    rep.table(eq, digits=1)
    names = pd.concat([train.pickup, train.delivery, val.pickup, val.delivery]).dropna().unique()
    stripped = [s for s in names if s != s.strip()]
    lower = pd.Series(names).str.lower().str.strip()
    case_clash = lower[lower.duplicated()].tolist()
    tr_c = set(train.pickup) | set(train.delivery)
    va_c = set(val.pickup) | set(val.delivery)
    lanes_tr = set(zip(train.pickup, train.delivery))
    rep.bullets(
        [
            f"Equipment levels: {sorted(pd.concat([train.equipment, val.equipment]).unique())}",
            f"City names with stray whitespace: {stripped or 'none'}; case-variant clashes: {case_clash or 'none'}",
            f"Cities: train {len(tr_c)}, validation {len(va_c)}",
            f"**Cities only in validation ({len(va_c - tr_c)}):** {sorted(va_c - tr_c)}",
            f"Cities only in train: {sorted(tr_c - va_c) or 'none'}",
            f"Distinct lanes in train: {len(lanes_tr):,} of {len(tr_c) * (len(tr_c) - 1):,} possible "
            f"(median {train.groupby(['pickup', 'delivery']).size().median():.0f} loads per lane)",
        ]
    )

    # 8. coordinates ---------------------------------------------------------------------------
    cc = city_coords(train, val)
    per_city = cc.groupby("city").size()
    coords = cc.drop_duplicates("city").set_index("city")
    bounds = []
    for col in ("lat", "lon"):
        for which, v in (("min", coords[col].min()), ("max", coords[col].max())):
            at = coords.index[coords[col] == v].tolist()
            bounds.append({"coordinate": f"{col} {which}", "value": v, "cities at exactly this value": ", ".join(at)})
    rep.h2("9. Coordinates")
    rep.bullets(
        [
            f"Cities with more than one (lat, lon) across roles/files: {int((per_city > 1).sum())} "
            "→ coordinates are a fixed lookup per city.",
            "Extreme coordinate values below. Several cities sharing a round extreme value is the "
            "signature of clipping to a bounding box.",
        ]
    )
    rep.table(pd.DataFrame(bounds), index=False, digits=5)
    rep.save()


if __name__ == "__main__":
    main()
