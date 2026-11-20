"""Shared helpers for the EDA scripts.

Design rule: everything here is NON-DESTRUCTIVE. Raw columns are never overwritten;
derived values get new names. That lets every script show "raw vs. fixed" side by side,
which is exactly the evidence the report needs.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

# pandas 2.3 + numpy 2.5 emit an internal deprecation about generic timedelta units; not actionable here.
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*generic.*unit.*")

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from cycler import cycler  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports" / "eda"
FIG_DIR = REPORT_DIR / "figures"

# --------------------------------------------------------------------------- palette
# Validated reference palette (CVD-safe for up to 3 series in any chart form).
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, NEUTRAL = "#e1e0d9", "#c3c2b7", "#f0efec"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"

EQUIPMENT = ["Dry Van", "Reefer", "Flatbed"]
EQUIP_COLOR = dict(zip(EQUIPMENT, [BLUE, ORANGE, AQUA]))
SPLIT_COLOR = {"train": BLUE, "validation": ORANGE}
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"]
)
DIV_CMAP = LinearSegmentedColormap.from_list(  # blue (+/low) <-> gray midpoint <-> red
    "div_blue_red", ["#104281", "#3987e5", "#9ec5f4", NEUTRAL, "#f4b4b3", RED, "#a8302f"]
)

# --------------------------------------------------------------------------- domain constants
WEIGHT_CAP = 47_500.0
# Relative-rate thresholds for "corrupted label". Justified in 03_data_quality.py by the
# empty gap in the distribution between ~0.5-0.85 and ~1.15-2.0.
OUTLIER_LO, OUTLIER_HI = 0.5, 2.0

US_HOLIDAYS_2025 = {
    "New Year's Day": "2025-01-01",
    "MLK Day": "2025-01-20",
    "Presidents Day": "2025-02-17",
    "Memorial Day": "2025-05-26",
    "Juneteenth": "2025-06-19",
    "Independence Day": "2025-07-04",
    "Labor Day": "2025-09-01",
    "Columbus Day": "2025-10-13",
    "Veterans Day": "2025-11-11",
    "Thanksgiving": "2025-11-27",
    "Christmas": "2025-12-25",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "text.color": INK,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.5,
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.prop_cycle": cycler(color=[BLUE, ORANGE, AQUA]),
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


# --------------------------------------------------------------------------- loading
def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, validation, december) exactly as on disk, with `date` parsed."""
    train = pd.read_csv(DATA_DIR / "train_test.csv", parse_dates=["date"])
    val = pd.read_csv(DATA_DIR / "validation.csv", parse_dates=["date"])
    dec = pd.read_csv(DATA_DIR / "december_chart_inputs.csv", parse_dates=["date"])
    return train, val, dec


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    d = out["date"]
    out["dow"] = d.dt.dayofweek
    out["dow_name"] = d.dt.day_name().str[:3]
    out["month"] = d.dt.month
    out["day"] = d.dt.day
    out["quarter"] = d.dt.quarter
    out["days_to_quarter_end"] = ((d + pd.offsets.QuarterEnd(0)) - d).dt.days
    out["day_of_quarter"] = (d - d.dt.to_period("Q").dt.start_time).dt.days
    return out


def add_rate_features(train: pd.DataFrame) -> pd.DataFrame:
    out = train.copy()
    out["rpm"] = out["posted_rate"] / out["distance"]
    out["log_rate"] = np.log(out["posted_rate"])
    return out


def daily_market_index(*frames: pd.DataFrame) -> pd.Series:
    """Daily mean market_index across the given frames (index = date)."""
    all_ = pd.concat([f[["date", "market_index"]] for f in frames])
    return all_.groupby("date")["market_index"].mean().rename("market_index_daily")


def relative_rate(train: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """How expensive was this load versus what is typical for its lane and equipment?

    rel_rate = rpm / (lane_base_rpm * equipment_premium)

    * equipment_premium: median over lanes of (lane-equipment median rpm / lane Dry-Van median rpm)
    * lane_base_rpm:     median over all loads in the lane of rpm / premium (Dry-Van equivalent)

    Medians everywhere, so corrupted labels barely move the reference. Using the whole lane
    (~10 loads) instead of lane x equipment (~3 loads) keeps the reference stable.
    Requires an `rpm` column.
    """
    df = train[["pickup", "delivery", "equipment", "rpm"]]
    lane_eq = df.groupby(["pickup", "delivery", "equipment"])["rpm"].median().unstack("equipment")
    premium = {"Dry Van": 1.0}
    for e in EQUIPMENT[1:]:
        premium[e] = float((lane_eq[e] / lane_eq["Dry Van"]).dropna().median())
    prem = df["equipment"].map(premium)
    lane_base = (df["rpm"] / prem).groupby([df["pickup"], df["delivery"]]).transform("median")
    expected = lane_base * prem
    out = pd.DataFrame(
        {"rel_rate": df["rpm"] / expected, "expected_rpm": expected, "lane_base_rpm": lane_base},
        index=train.index,
    )
    return out, premium


def flag_outliers(rel: pd.Series) -> pd.Series:
    return (rel > OUTLIER_HI) | (rel < OUTLIER_LO)


def clean_inputs(df: pd.DataFrame, daily_mi: pd.Series, weight_fill: dict[str, float]) -> pd.DataFrame:
    """Prediction-time input cleaning, as found in 03_data_quality.

    * weight: sign flips → abs(); missing → equipment median of the training data (flagged)
    * market_index: missing → same-date mean (flagged)
    Adds weight_abs, mi, log_distance, log_weight, log_mi, weight_missing, mi_missing.
    """
    out = df.copy()
    out["weight_missing"] = out["weight"].isna()
    out["weight_abs"] = out["weight"].abs().fillna(out["equipment"].map(weight_fill))
    out["mi_missing"] = out["market_index"].isna()
    out["mi"] = out["market_index"].fillna(out["date"].map(daily_mi))
    out["log_distance"] = np.log(out["distance"])
    out["log_weight"] = np.log(out["weight_abs"])
    out["log_mi"] = np.log(out["mi"])
    return out


MODEL_OUTLIER_LOG = 0.4  # |log residual| > 0.4 (≈ ×1.5 or ÷1.5) vs the structural model = corrupted label


def clean_train(train: pd.DataFrame, val: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train rows with corrupted labels removed and inputs cleaned. Returns (clean_train, all_train_with_flags).

    Two passes: (1) rel_rate vs the lane's typical rate (03_data_quality); (2) residual vs the structural
    model, which catches corrupted labels on thin lanes where the corrupted value *is* the lane median (09).
    """
    tr = add_rate_features(train)
    rel, _ = relative_rate(tr)
    tr = tr.join(rel)
    tr["is_outlier_lane"] = flag_outliers(tr["rel_rate"])
    frames = (train,) if val is None else (train, val)
    weight_fill = train.assign(w=train["weight"].abs()).groupby("equipment")["w"].median().to_dict()
    tr = clean_inputs(tr, daily_market_index(*frames), weight_fill)
    cities = sorted(set(tr["pickup"]) | set(tr["delivery"]))
    first = tr[~tr["is_outlier_lane"]]
    beta, _ = ols(structural_matrix(first, cities, "shared")[0], np.log(first["posted_rate"].values))
    Xall = structural_matrix(tr, cities, "shared")[0]
    tr["structural_resid"] = np.log(tr["posted_rate"].values) - np.column_stack([np.ones(len(tr)), Xall]) @ beta
    tr["is_outlier_model"] = tr["structural_resid"].abs() > MODEL_OUTLIER_LOG
    tr["is_outlier"] = tr["is_outlier_lane"] | tr["is_outlier_model"]
    return tr[~tr["is_outlier"]].copy(), tr


WEIGHT_LO, WEIGHT_HI = 20_000.0, 40_000.0  # weight only matters inside this band (09_structure)


def structural_matrix(
    df: pd.DataFrame, cities: list[str], city_mode: str = "shared", form: str = "refined"
) -> tuple[np.ndarray, list[str]]:
    """Design matrix of the multiplicative description of price (OLS on log price).

    form='log'     : log d, log w, log mi                         (first guess, step 04)
    form='refined' : log d, (log d)², (log d)³, clip(w, 20k, 40k), mi   (shapes found in 09_structure)
    plus Reefer and Flatbed indicators, plus city terms:
    city_mode: 'separate' = origin and destination dummies, 'shared' = one effect per city used for
    both ends, 'none' = no city terms. The first city is the reference level.
    """
    reefer = (df["equipment"] == "Reefer").values.astype(float)
    flatbed = (df["equipment"] == "Flatbed").values.astype(float)
    ld = df["log_distance"].values
    if form == "log":
        cols = [ld, df["log_weight"].values, df["log_mi"].values]
        names = ["log_distance", "log_weight", "log_mi"]
    else:
        w = np.clip(df["weight_abs"].values, WEIGHT_LO, WEIGHT_HI) / 1e4
        cols = [ld, ld**2, ld**3, w, df["mi"].values]
        names = ["log_distance", "log_distance^2", "log_distance^3", "weight_clipped_10k", "market_index"]
    cols += [reefer, flatbed]
    names += ["Reefer", "Flatbed"]
    if city_mode != "none":
        o = pd.Categorical(df["pickup"], categories=cities)
        d = pd.Categorical(df["delivery"], categories=cities)
        O = np.eye(len(cities))[o.codes][:, 1:]
        Dm = np.eye(len(cities))[d.codes][:, 1:]
        if city_mode == "separate":
            cols += list(O.T) + list(Dm.T)
            names += [f"o:{c}" for c in cities[1:]] + [f"d:{c}" for c in cities[1:]]
        else:
            cols += list((O + Dm).T)
            names += [f"c:{c}" for c in cities[1:]]
    return np.column_stack(cols), names


def fit_structural(df: pd.DataFrame, city_mode: str = "shared", cities: list[str] | None = None, form: str = "refined") -> dict:
    """OLS fit of the multiplicative structure on clean rows. Returns coefficients, residuals and city effects."""
    cities = cities or sorted(set(df["pickup"]) | set(df["delivery"]))
    X, names = structural_matrix(df, cities, city_mode, form)
    y = np.log(df["posted_rate"].values)
    beta, fitted = ols(X, y)
    coef = pd.Series(beta, index=["const", *names])
    out = {"coef": coef, "resid": pd.Series(y - fitted, index=df.index), "fitted": fitted, "cities": cities,
           "form": form, "city_mode": city_mode}
    if city_mode == "separate":
        o = pd.Series([0.0] + [coef[f"o:{c}"] for c in cities[1:]], index=cities)
        d = pd.Series([0.0] + [coef[f"d:{c}"] for c in cities[1:]], index=cities)
        out["origin"], out["destination"] = o - o.mean(), d - d.mean()
    elif city_mode == "shared":
        s = pd.Series([0.0] + [coef[f"c:{c}"] for c in cities[1:]], index=cities)
        out["city"] = s - s.mean()
    return out


def predict_structural(fit: dict, df: pd.DataFrame) -> np.ndarray:
    """log-price prediction from a fit_structural result (cities must be known to the fit)."""
    X, _ = structural_matrix(df, fit["cities"], fit["city_mode"], fit["form"])
    return np.column_stack([np.ones(len(df)), X]) @ fit["coef"].values


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(h))


def city_coords(*frames: pd.DataFrame) -> pd.DataFrame:
    """One row per (city, lat, lon) seen in either role in any frame."""
    parts = []
    for f in frames:
        for role in ("pickup", "delivery"):
            if f"{role}_lat" in f:
                parts.append(
                    f[[role, f"{role}_lat", f"{role}_lon"]].set_axis(["city", "lat", "lon"], axis=1)
                )
    return pd.concat(parts).drop_duplicates().sort_values("city").reset_index(drop=True)


# --------------------------------------------------------------------------- stats helpers
def binned(x: pd.Series, y: pd.Series, bins) -> pd.DataFrame:
    """Median/IQR/mean of y within bins of x (x_mid = median x in the bin)."""
    frame = pd.DataFrame({"x": x.values, "y": y.values, "bin": pd.cut(x.values, bins)})
    g = frame.groupby("bin", observed=True)
    return g.agg(
        x_mid=("x", "median"),
        median=("y", "median"),
        q25=("y", lambda s: s.quantile(0.25)),
        q75=("y", lambda s: s.quantile(0.75)),
        mean=("y", "mean"),
        n=("y", "size"),
    )


def plot_binned(ax, b: pd.DataFrame, color: str, label: str | None = None, band: bool = True) -> None:
    if band:
        ax.fill_between(b["x_mid"], b["q25"], b["q75"], color=color, alpha=0.14, linewidth=0)
    ax.plot(b["x_mid"], b["median"], color=color, marker="o", label=label)


def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least squares with intercept prepended. Returns (coefficients, fitted)."""
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta, A @ beta


def error_summary(y_true, y_pred) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    ape = np.abs(err) / y_true
    return {
        "MAE ($)": np.mean(np.abs(err)),
        "RMSE ($)": np.sqrt(np.mean(err**2)),
        "MAPE (%)": 100 * ape.mean(),
        "median APE (%)": 100 * np.median(ape),
        "R2": 1 - np.sum(err**2) / np.sum((y_true - y_true.mean()) ** 2),
    }


# --------------------------------------------------------------------------- reporting
def _fmt(v, digits: int) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        a = abs(v)
        if a >= 1000:
            return f"{v:,.0f}"
        if a >= 100:
            return f"{v:.1f}"
        return f"{v:.{digits}f}"
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    return str(v)


def md_table(df: pd.DataFrame, digits: int = 3, index: bool = True) -> str:
    d = (df.reset_index() if index else df).astype(object)
    for c in d.columns:  # integer-valued float columns (ints + NaN) print as ints
        vals = pd.to_numeric(d[c], errors="coerce")
        if d[c].map(lambda v: isinstance(v, (float, np.floating))).all() and vals.notna().any():
            if np.all(np.mod(vals.dropna(), 1) == 0):
                d[c] = [int(v) if pd.notna(v) else np.nan for v in vals]
    header = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "| " + " | ".join("---" for _ in d.columns) + " |"
    rows = ["| " + " | ".join(_fmt(v, digits) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


class Report:
    """Collects markdown + figures for one EDA step and writes reports/eda/<slug>.md."""

    def __init__(self, slug: str, title: str, intro: str = "") -> None:
        self.slug = slug
        self.lines = [f"# {title}", ""]
        if intro:
            self.lines += [intro, ""]
        FIG_DIR.mkdir(parents=True, exist_ok=True)

    def h2(self, text: str) -> None:
        self.lines += [f"## {text}", ""]

    def h3(self, text: str) -> None:
        self.lines += [f"### {text}", ""]

    def p(self, text: str) -> None:
        self.lines += [text, ""]

    def bullets(self, items) -> None:
        self.lines += [f"- {i}" for i in items] + [""]

    def table(self, df: pd.DataFrame, digits: int = 3, index: bool = True) -> None:
        self.lines += [md_table(df, digits, index), ""]

    def figure(self, fig, name: str, caption: str) -> None:
        path = FIG_DIR / f"{self.slug}__{name}.png"
        fig.savefig(path)
        plt.close(fig)
        self.lines += [f"![{caption}](figures/{path.name})", "", f"*{caption}*", ""]

    def save(self) -> Path:
        path = REPORT_DIR / f"{self.slug}.md"
        path.write_text("\n".join(self.lines))
        print(f"wrote {path.relative_to(ROOT)}")
        return path
