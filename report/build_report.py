"""Build the PDF report: report/Freight_Rate_Prediction_Report.pdf

    python report/build_report.py      (run python predict.py first: the report embeds scorer_results/candidate_december.png)

Numbers in the text come from the modelling reports in reports/model/ (steps 01–09) and the final fit; the two
report-specific figures (validation design, model comparison) are drawn here.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import letter  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import inch  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image, KeepTogether, ListFlowable, ListItem, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from freight.reporting import AQUA, BLUE, GRID, INK, INK_2, MUTED, ORANGE, set_style  # noqa: E402

OUT = ROOT / "report" / "Freight_Rate_Prediction_Report.pdf"
FIG = ROOT / "report" / "figures"
TITLE = "Freight Rate Prediction: Validation Approach and Results"
AUTHOR = "Neill Rawani"

# ---------------------------------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------------------------------
FOLD_ROWS = [  # label, train end, test end, clean train loads, test loads, hidden cities
    ("Fold 1", "2025-05-31", "2025-07-31", "20,076", "9,695"),
    ("Fold 2", "2025-06-30", "2025-08-31", "25,320", "9,671"),
    ("Fold 3", "2025-07-31", "2025-09-30", "29,689", "9,429"),
    ("Fold 4", "2025-08-31", "2025-10-31", "32,904", "9,523"),
]


def fig_folds(path: Path) -> None:
    start, d = pd.Timestamp("2025-01-01"), np.timedelta64(1, "D")
    fig, ax = plt.subplots(figsize=(8.6, 3.0))
    rows = FOLD_ROWS + [("Final", "2025-10-31", "2025-12-31", "47,322", "12,000")]
    for i, (lab, tr_end, te_end, ntr, nte) in enumerate(rows):
        y = len(rows) - 1 - i
        tr_end, te_end = pd.Timestamp(tr_end), pd.Timestamp(te_end)
        final = lab == "Final"
        ax.barh(y, (tr_end + d - start).days, left=start, height=0.62, color=BLUE, edgecolor="white", linewidth=2)
        ax.barh(y, (te_end - tr_end).days, left=tr_end + d, height=0.62, color=AQUA if final else ORANGE, edgecolor="white", linewidth=2)
        ax.text(start + np.timedelta64(6, "D"), y, f"train · {ntr} loads" + ("" if final else " · 4 cities hidden"),
                va="center", ha="left", fontsize=7.6, color="white")
        ax.text(tr_end + np.timedelta64(31, "D"), y, (f"predict · {nte}" if final else f"test · {nte}"),
                va="center", ha="center", fontsize=7.6, color="white" if not final else INK)
    for qe in ("2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"):
        ax.axvline(pd.Timestamp(qe) + d, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)), zorder=0)
    ax.text(pd.Timestamp("2025-03-31") + np.timedelta64(3, "D"), len(rows) - 0.48, "quarter end", fontsize=7, color=MUTED, va="bottom")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.set_ylim(-0.6, len(rows) - 0.25)
    ax.set_xlim(start, pd.Timestamp("2026-01-02"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[Patch(color=BLUE, label="training (cleaned inside the fold)"), Patch(color=ORANGE, label="test: next two months, scored"),
                       Patch(color=AQUA, label="final predictions (no labels)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    ax.set_title("Forward-chaining validation: every fold predicts the two months after its training data", fontsize=9.5)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


MODELS = [  # name, MAPE %, RMSE $  (future-month folds; reports/model/05–07)
    ("Structural model (final)", 1.394, 47.68),
    ("Leave-one-fold-out blend of all models", 1.394, 47.76),
    ("Linear / Ridge / Lasso, same inputs", 1.395, 47.69),
    ("Structural + XGBoost on its errors", 1.395, 47.76),
    ("Structural + MLP on its errors", 1.407, 48.46),
    ("Average of structural and LightGBM", 1.455, 49.63),
    ("Neural network (MLP)", 1.587, 54.45),
    ("LightGBM", 1.639, 55.35),
    ("XGBoost", 1.654, 55.71),
    ("Gradient boosting (scikit-learn)", 1.697, 58.47),
    ("Extra trees", 1.952, 67.76),
    ("Random forest", 1.968, 67.65),
    ("k-nearest neighbours", 2.614, 87.72),
]


def fig_models(path: Path) -> None:
    names = [m[0] for m in MODELS][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, k, unit, title in ((axes[0], 1, "%", "MAPE (%)"), (axes[1], 2, "$", "RMSE ($)")):
        vals = [m[k] for m in MODELS][::-1]
        cols = [BLUE if n.startswith("Structural model") else "#c9c7be" for n in names]
        ax.barh(range(len(names)), vals, color=cols, height=0.66)
        for i, v in enumerate(vals):
            ax.text(v * 1.01, i, f"{v:.2f}" if unit == "%" else f"${v:.1f}", va="center", fontsize=7, color=INK_2)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlim(0, max(vals) * 1.16)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(names)), names, fontsize=7.6)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------------
# document
# ---------------------------------------------------------------------------------------------------------------------
def register_fonts() -> None:
    base = Path("/usr/share/fonts/truetype/liberation")
    for name, file in (("Body", "LiberationSans-Regular.ttf"), ("Body-Bold", "LiberationSans-Bold.ttf"),
                       ("Body-Italic", "LiberationSans-Italic.ttf"), ("Body-BoldItalic", "LiberationSans-BoldItalic.ttf"),
                       ("Mono", "LiberationMono-Regular.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(base / file)))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-BoldItalic")


C_INK, C_INK2, C_MUTED = colors.HexColor(INK), colors.HexColor(INK_2), colors.HexColor(MUTED)
C_BLUE, C_GRID, C_HEAD, C_BOX = colors.HexColor(BLUE), colors.HexColor(GRID), colors.HexColor("#f2f1ec"), colors.HexColor("#eef4fc")

S = {
    "title": ParagraphStyle("title", fontName="Body-Bold", fontSize=19, leading=23, textColor=C_INK, spaceAfter=4),
    "sub": ParagraphStyle("sub", fontName="Body", fontSize=9.5, leading=13, textColor=C_INK2, spaceAfter=12),
    "h1": ParagraphStyle("h1", fontName="Body-Bold", fontSize=13, leading=16, textColor=C_INK, spaceBefore=12, spaceAfter=5, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="Body-Bold", fontSize=10.5, leading=13.5, textColor=C_INK, spaceBefore=8, spaceAfter=3, keepWithNext=1),
    "body": ParagraphStyle("body", fontName="Body", fontSize=9.3, leading=13.2, textColor=C_INK, spaceAfter=5, alignment=TA_LEFT),
    "small": ParagraphStyle("small", fontName="Body", fontSize=8, leading=10.6, textColor=C_INK2, spaceAfter=6),
    "cap": ParagraphStyle("cap", fontName="Body-Italic", fontSize=8, leading=10.6, textColor=C_INK2, spaceBefore=2, spaceAfter=9),
    "cell": ParagraphStyle("cell", fontName="Body", fontSize=8.1, leading=10.4, textColor=C_INK),
    "cellb": ParagraphStyle("cellb", fontName="Body-Bold", fontSize=8.1, leading=10.4, textColor=C_INK),
    "box": ParagraphStyle("box", fontName="Body", fontSize=9.1, leading=12.8, textColor=C_INK),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=8.2, leading=11, textColor=C_INK, backColor=C_HEAD,
                           borderPadding=(5, 6, 5, 6), spaceBefore=4, spaceAfter=10),
}
WIDTH = letter[0] - 2 * 0.8 * inch


def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def bullets(items: list[str], style: str = "body") -> ListFlowable:
    return ListFlowable([ListItem(P(t, style), leftIndent=12, value="•") for t in items], bulletType="bullet", start="•",
                        leftIndent=12, bulletFontName="Body", bulletFontSize=8.5, spaceAfter=4)


def table(rows: list[list], widths: list[float], num_cols: tuple[int, ...] = (), bold_rows: tuple[int, ...] = ()) -> Table:
    def cell(text, r, c):
        st = S["cellb" if r == 0 or r in bold_rows else "cell"]
        if c in num_cols:
            st = ParagraphStyle(f"{st.name}-num", parent=st, alignment=2)
        return Paragraph(str(text), st)

    data = [[cell(v, r, c) for c, v in enumerate(row)] for r, row in enumerate(rows)]
    t = Table(data, colWidths=[w * WIDTH for w in widths], repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), C_HEAD), ("LINEBELOW", (0, 0), (-1, 0), 0.8, C_INK2),
             ("LINEBELOW", (0, 1), (-1, -1), 0.4, C_GRID), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    for r in bold_rows:
        style.append(("BACKGROUND", (0, r), (-1, r), C_BOX))
    t.setStyle(TableStyle(style))
    return t


def figure(path: Path, caption: str, width: float = 1.0, head: str | None = None) -> KeepTogether:
    img = plt.imread(path)
    h, w = img.shape[:2]
    im = Image(str(path), width=WIDTH * width, height=WIDTH * width * h / w)
    return KeepTogether(([P(head, "h2")] if head else []) + [im, P(caption, "cap")])


def box(paragraphs: list) -> Table:
    t = Table([[paragraphs]], colWidths=[WIDTH])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), C_BOX), ("LINEBEFORE", (0, 0), (0, -1), 3, C_BLUE),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return t


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Body", 7.5)
    canvas.setFillColor(C_MUTED)
    canvas.drawString(0.8 * inch, 0.5 * inch, TITLE)
    canvas.drawRightString(letter[0] - 0.8 * inch, 0.5 * inch, f"{doc.page}")
    canvas.restoreState()


def story() -> list:
    s: list = []
    s += [P(TITLE, "title"), P(f"Spotter · Machine Learning Engineer assessment · {AUTHOR} · November 2026", "sub")]
    s.append(box([
        P("<b>Summary</b>", "box"), Spacer(1, 3),
        bullets([
            "<b>Task.</b> Predict <font name='Mono'>posted_rate</font> for 12,000 loads in Nov–Dec 2025 from 48,000 labelled loads in Jan–Oct 2025.",
            "<b>Validation.</b> Forward-chaining: four folds, each trained on all months up to a cut-off and tested on the <i>next two months</i>, "
            "with four random cities removed from training so that 11–16% of test loads involve an unseen city (12% in the real validation file). "
            "Cleaning, corrupted-label removal and fitting are redone inside every fold.",
            "<b>Model.</b> A structural model of log price, fitted by least squares: distance curve × equipment × weight × market index × "
            "city and route levels × quarter-end ramp × slow drift.",
            "<b>Expected accuracy on future months.</b> MAPE <b>1.39%</b>, RMSE <b>$47.7</b> (MAE $32.9, median APE 1.18%, worst fold 1.49%; "
            "loads with an unseen city 1.56%).",
            "<b>Alternatives.</b> Nine model families (including LightGBM, XGBoost, random forest, a neural network), ensembles and stacking were "
            "tuned on the same folds; none beat the structural model reliably.",
            "<b>Reproduce.</b> <font name='Mono'>python predict.py</font> (about 3 seconds) writes "
            "<font name='Mono'>validation_predictions.csv</font>, fills the December file and runs <font name='Mono'>score.py</font>.",
        ], "box"),
    ]))

    # 1 ---------------------------------------------------------------------------------------------------------------
    s.append(P("1. What shaped the validation design", "h1"))
    s.append(P("Exploration of the data (12 analysis scripts in <font name='Mono'>eda/</font>, reports in "
               "<font name='Mono'>reports/eda/</font>) found four facts that decide how a model has to be validated:"))
    s.append(bullets([
        "<b>The target period is in the future.</b> Prices drift up about 0.6% per month and climb 2–8% (depending on equipment) over the last "
        "four weeks of every quarter. December ends a quarter. A model must extrapolate in time, so validation must measure extrapolation.",
        "<b>New cities.</b> 8 of the 72 validation cities never appear in training; they account for 12.1% of validation loads.",
        "<b>A leaking feature.</b> <font name='Mono'>quote_signal</font> tracks the price within each month, but the direction of the link "
        "flips from month to month (positive Jan–Mar, negative Apr–May, …) and is absent in August. In Nov–Dec it behaves like August. "
        "A random split rewards it; a time split exposes it (Section 3.1).",
        "<b>Corrupted labels.</b> 1.4% of prices are multiplied or divided by 2.2–5.6, at random across lanes and dates. They cannot be "
        "predicted and would distort both training and scoring.",
    ]))
    s.append(P("Apart from these, the data is very regular: price is a product of independent effects (no interaction found was worth more "
               "than 0.5 basis points) with roughly normal noise of about 1.5–2% per load."))

    # 2 ---------------------------------------------------------------------------------------------------------------
    s.append(P("2. Data quality issues and fixes", "h1"))
    s.append(table([
        ["Issue", "Train", "Validation", "Evidence", "Fix"],
        ["Negative weight", "292", "145", "Magnitudes match normal weights; prices are normal", "Absolute value"],
        ["Missing weight", "300", "165", "Missing at random", "Median weight of the equipment type (training data)"],
        ["Missing market_index", "374", "249", "market_index is a daily value plus small noise (sd 0.025)", "Mean market_index of the same date (inputs only)"],
        ["Corrupted price", "678 (1.41%)", "unknown", "Price per mile ×/÷ 2.2–5.6 vs the lane's typical value", "Removed from training in two passes: lane rate outside "
         "0.5–2× the lane median, then |log residual| > 0.4 vs the structural model (4 more on thin lanes)"],
        ["Values at limits", "–", "–", "Weight clipped to 5,000–47,500 lb; distance floored at 70 mi; 3 cities' coordinates clipped",
         "Kept: weight only matters between 20,000 and 40,000 lb; coordinates are only used to find nearby cities"],
        ["December file", "–", "31 rows", "No market_index or coordinates", "Daily market_index from validation.csv; coordinates from the city lookup"],
        ["quote_signal", "–", "–", "Leakage (Section 3.1)", "Dropped"],
    ], [0.15, 0.1, 0.11, 0.3, 0.34]))
    s.append(P("Every fix is a function applied identically to the training, validation and December files at prediction time. This matters "
               "because the validation file is dirtier than the training data (2.1% missing market_index vs 0.8%; 1.2% negative weights vs 0.6%).",
               "small"))

    # 3 ---------------------------------------------------------------------------------------------------------------
    s.append(P("3. Validation and data split", "h1"))
    s.append(P("3.1 Why not a random split", "h2"))
    s.append(P("A random split mixes the same days into training and test. Under a random 5-fold split, adding "
               "<font name='Mono'>quote_signal</font> to a gradient-boosted model improves MAPE from 1.50% to 1.30% (RMSE $49 → $43). "
               "But when the test period carries no quote signal, as in August and in the real Nov–Dec task, the same feature doubles the error: "
               "trained on Jan–Jul and tested on August, MAPE rises from 1.57% to 3.13% (RMSE $49 → $93). Shuffling "
               "<font name='Mono'>quote_signal</font> inside the random-split test set reproduces the damage (2.57%, $80). A random split would "
               "have selected a model that fails on the real target, and it would also hide the two other future-only difficulties: "
               "extrapolating the drift and pricing unseen cities. The feature was dropped."))
    s.append(figure(ROOT / "reports/eda/figures/08_trap_proof__with_without.png",
                    "Figure 1. Gradient-boosted model with and without quote_signal. Left group: random split. Right groups: train on all earlier months, "
                    "test one month (its quote_signal regime in brackets). Green: test values shuffled, i.e. a future with no signal."))
    s.append(figure(FIG / "folds.png", "Figure 2. The four validation folds and the final fit. Dashed lines mark quarter ends.", head="3.2 The design"))
    s.append(bullets([
        "<b>Forward chaining with a two-month horizon</b>, the same horizon as the real task. Errors are also reported separately for the first "
        "and second test month.",
        "<b>Unseen cities.</b> Four random cities are removed from each fold's training data (11–16% of test loads), so the way the model "
        "prices new cities is tested on every fold.",
        "<b>No leakage.</b> Weight fill values, corrupted-label removal and all fitting use only the fold's training part. Daily market_index "
        "means use inputs only, never labels. Route and city levels for the stacked models were computed out-of-fold.",
        "<b>Scoring.</b> Test loads with corrupted labels (≈1.4%) are excluded, because no model can predict a randomly ×4 price; including "
        "them raises the final model's RMSE from $48 to $630.",
        "<b>Metrics.</b> The competition metric is hidden, so both MAPE (every load counts equally) and RMSE (dominated by long, expensive "
        "loads) are reported for every comparison, with MAE, median APE, seen/unseen-city MAPE, bias and the worst fold.",
        "<b>Selection rule.</b> A more complex model or setting replaces a simpler one only if it improves MAPE by ≥ 0.02 points <i>and</i> "
        "RMSE by ≥ $1, about the size of fold-to-fold noise. Each setting was swept around the value suggested by the data exploration.",
    ]))
    s.append(table([
        ["Fold", "Train", "Test", "Clean training loads", "Test loads", "Hidden cities", "Test loads with an unseen city"],
        ["1", "Jan–May", "Jun–Jul", "20,076", "9,695", "Dayton, Lexington, Milwaukee, Richmond", "15.6%"],
        ["2", "Jan–Jun", "Jul–Aug", "25,320", "9,671", "Indianapolis, Las Vegas, Phoenix, Toledo", "11.0%"],
        ["3", "Jan–Jul", "Aug–Sep", "29,689", "9,429", "Baltimore, Dallas, El Paso, Richmond", "11.0%"],
        ["4", "Jan–Aug", "Sep–Oct", "32,904", "9,523", "Bakersfield, Charleston, Corpus Christi, Raleigh", "13.3%"],
        ["Final", "Jan–Oct", "Nov–Dec", "47,322", "12,000", "– (8 genuinely new cities)", "12.1%"],
    ], [0.07, 0.09, 0.09, 0.14, 0.1, 0.37, 0.14], num_cols=(3, 4, 6)))
    s.append(P("3.3 How robust are the conclusions?", "h2"))
    s.append(table([
        ["Check", "Result for the final model"],
        ["Three more random draws of hidden cities", "1.392–1.400% MAPE, $47.7–48.1 RMSE; the ranking of models never changes"],
        ["Two earlier folds never used for any tuning (train Jan–Mar → test Apr–May; Jan–Apr → May–Jun)", "1.436% and 1.426% MAPE, within the range of the main folds"],
        ["Corrupted-label thresholds tightened or loosened by 20–25%", "MAPE changes by ≤ 0.001. Skipping the second pass costs $4.3 RMSE; no cleaning gives 2.33% / $104"],
        ["Drift after the last training day: stop / half speed / full speed", "1.404 / 1.394 / 1.452% MAPE. Half speed (chosen) is never the worst fold"],
        ["Same months instead of future months (random split)", "≈ 1.24% MAPE, so about 0.15 points of the future-month error is the cost of predicting ahead"],
    ], [0.5, 0.5]))
    s.append(P("<b>Final fit.</b> The chosen settings were then fitted once on all 47,322 clean Jan–Oct loads and applied to the validation file.",
               "body"))

    # 4 ---------------------------------------------------------------------------------------------------------------
    s.append(P("4. Model", "h1"))
    s.append(P("4.1 Form and fitted effects", "h2"))
    s.append(P("log(price) = f(log distance) + weight + market index + equipment + quarter-end ramp × equipment + drift + city(pickup) + "
               "city(delivery) + route(pickup → delivery) + noise", "code"))
    s.append(bullets([
        "Ordinary least squares on log price: the noise is a constant <i>percentage</i> of the price (normal on the log scale, the same at "
        "every distance and for every equipment type), so effects multiply. 75 coefficients plus shrunk route levels.",
        "City levels are shared between pickup and delivery (separately estimated origin and destination levels correlate at 0.99). A city "
        "absent from training gets the inverse-distance-weighted mean level of its 5 nearest cities, which recovers 94% of a held-out city's level.",
        "Route level = the route's mean residual shrunk toward zero, Σ residuals / (n + 3); a route never seen gets 0. The shrinkage "
        "λ = 3 was chosen on the folds and is close to noise variance ÷ route-level variance.",
        "Drift continues beyond the last training day at half speed: a hedge, because the folds split 2–2 between stopping and continuing.",
    ]))
    s.append(table([
        ["Effect", "Fitted on all clean Jan–Oct loads"],
        ["Distance", "Price grows like distance<super>0.85</super> at 250–500 mi, rising to distance<super>0.93</super> at 2,000 mi"],
        ["Equipment", "Reefer +12.5%, Flatbed +7.6% vs Dry Van"],
        ["Weight", "+3.8% per 10,000 lb between 20,000 and 40,000 lb; flat outside that band"],
        ["Market index", "+1.4% per +0.1"],
        ["Quarter-end ramp", "Rises linearly over the last 28 days of each quarter to Dry Van +2.4%, Reefer +5.1%, Flatbed +8.2%"],
        ["Drift", "+0.59% per 30 days"],
        ["City levels (64)", "−3.7% (Syracuse) to +3.2% (Oklahoma City); sd 1.9%"],
        ["Route levels (4,012 directed routes)", "sd 0.8%; 90% within ±1.4%"],
    ], [0.3, 0.7]))
    s.append(P("4.2 How the model was built: one ingredient at a time", "h2"))
    s.append(table([
        ["Step", "Model", "MAPE (%)", "RMSE ($)"],
        ["0", "Naive: typical price per mile × distance", "9.34", "291.6"],
        ["1", "+ distance curve + equipment", "3.90", "125.3"],
        ["2", "+ weight (20,000–40,000 lb band)", "3.35", "105.4"],
        ["3", "+ market index (worse until the time terms are in: it absorbs the drift)", "3.73", "116.1"],
        ["4", "+ city levels", "3.18", "103.4"],
        ["5", "+ quarter-end ramp", "2.87", "90.6"],
        ["6", "+ slow drift", "1.73", "59.0"],
        ["7", "+ unseen cities priced from their 5 nearest cities", "1.67", "57.2"],
        ["8", "Drift at half speed after the training period", "1.62", "55.6"],
        ["9", "Ramp size by equipment (found by studying what residual trees learned)", "1.53", "52.3"],
        ["10", "+ route levels: final model", "1.39", "47.7"],
    ], [0.07, 0.65, 0.14, 0.14], num_cols=(2, 3), bold_rows=(11,)))
    s.append(P("4.3 Alternatives tested on the same folds", "h2"))
    s.append(P("Each family was tuned by random search on the same folds and given the same engineered inputs, including leak-free city and "
               "route levels."))
    s.append(figure(FIG / "models.png", "Figure 3. Future-month error of the best tuned setting of each alternative (naive baseline, 9.34%, not shown)."))
    s.append(bullets([
        "<b>Linear, Ridge and Lasso regression</b> on the same inputs reproduce the structural model; regularisation adds nothing with 47,000 "
        "loads and few inputs.",
        "<b>Tree ensembles</b> (LightGBM 1.64%, XGBoost 1.65%, random forest 1.97%) under-predict by about 0.6%: trees cannot continue a "
        "trend beyond the training period, and they approximate smooth curves with steps. k-nearest neighbours (2.61%) and a standalone "
        "neural network (1.59%) are also worse.",
        "<b>Combining does not help.</b> Averaging with trees dilutes the structural model (with LightGBM: 1.455%). Trees or a neural "
        "network fitted to its errors choose their most regularised settings and tie (1.395–1.407%), a sign that no pattern is left. "
        "A non-negative blend of all models, weighted on three folds and tested on the fourth, also ties (1.394%).",
    ]))
    near = [P("<b>The one near-miss.</b> The structural model plus 400 shallow trees on its errors was about 1% better on average. Before "
              "adopting it, five criteria were fixed in advance and tested on 6 folds × 4 hidden-city draws:")]
    near.append(table([
        ["Criterion", "Result", "Met"],
        ["Better on the two folds never used for tuning", "−0.013 MAPE points, −$2.70", "yes"],
        ["Better in ≥ 75% of the 24 fold × draw comparisons", "MAPE 79%, RMSE 71%", "no"],
        ["Never worse by more than 0.03 MAPE points", "Worst loss +0.032", "no"],
        ["Not worse for loads with an unseen city", "−0.012 points", "yes"],
        ["Typical Nov–Dec adjustment within ±0.2%", "−0.18%", "yes"],
    ], [0.5, 0.36, 0.14]))
    near.append(Spacer(1, 5))
    near.append(P("It lost one fold in every draw, and its gain appeared only when the trees could see all inputs together (no single group of "
                  "inputs explained it), so it was not adopted. The final model is the structural model.", "body"))
    s.append(KeepTogether(near))

    # 5 ---------------------------------------------------------------------------------------------------------------
    s.append(P("5. December prediction chart", "h1"))
    s.append(figure(ROOT / "scorer_results/candidate_december.png",
                    "Figure 4. The fixed December chart produced by score.py: Lexington → Fort Wayne, 360 mi, Dry Van, 32,000 lb; only the date changes."))
    s.append(P("The December file has no market_index, so each day's mean market_index from <font name='Mono'>validation.csv</font> is used. "
               "The predicted price rises from <b>$826</b> on Dec 1 to <b>$872</b> on Dec 31 (this route's Dry Van median in training: $808)."))
    s.append(bullets([
        "<b>Weekly zig-zag</b> (peaks on Thursdays, troughs on Sundays, about $20): carried by the market index, which has a strong weekly cycle.",
        "<b>Climb over the last four weeks</b>: the quarter-end ramp, which reaches +2.4% for Dry Van on Dec 31.",
        "<b>Slow drift</b>: about +0.3% over the month at half speed.",
    ]))
    s.append(figure(ROOT / "reports/model/figures/08_loose_ends__december_drift.png",
                    "Figure 5. Sensitivity to the drift assumption: stopping or continuing at full speed moves Dec 31 to $867 or $877 (±0.6%); the shape is unchanged.",
                    width=0.92))

    # 6 ---------------------------------------------------------------------------------------------------------------
    s.append(P("6. Limitations and risks", "h1"))
    s.append(bullets([
        "<b>Drift after October is an assumption.</b> Stopping it or continuing at full speed instead of half speed would move predictions "
        "by about ±0.2% in November and ±0.5% in December.",
        "<b>No December history.</b> Christmas and year-end effects cannot be learned from Jan–Oct; the model assumes the Dec 31 quarter end "
        "behaves like the March, June and September ones (no holiday effect was found in training).",
        "<b>New cities</b> carry about 14% more error (1.56% vs 1.37% MAPE on the folds), since they have no route history.",
        "<b>Corrupted labels in the hidden answers</b>, if present at the training rate (~1.4%), will dominate any RMSE and affect every model alike.",
    ]))

    # appendix ------------------------------------------------------------------------------------------------------------
    s.append(P("Appendix A. Reproducing the results", "h1"))
    s.append(Preformatted("python -m pip install -r requirements.txt\n"
                          "python predict.py                  # final predictions, December file, scorer chart\n"
                          "python report/build_report.py      # this report\n"
                          "python eda/run_all.py              # data exploration reports (reports/eda/)\n"
                          "python modeling/01_structural.py   # ... to modeling/09_hybrid_check.py, in order (reports/model/)", S["code"]))
    s.append(table([
        ["Path", "Contents"],
        ["predict.py", "Final pipeline: clean, remove corrupted labels, fit, predict, run score.py"],
        ["src/freight/data.py", "Loading and prediction-time cleaning; corrupted-label rule"],
        ["src/freight/models.py", "Naive baseline, structural model, tree and hybrid models"],
        ["src/freight/evaluation.py", "Forward-chaining folds with hidden cities; metrics"],
        ["src/freight/config.py", "Final model settings, each with the experiment that chose it"],
        ["eda/, reports/eda/", "12 exploration scripts and their reports"],
        ["modeling/, reports/model/", "9 modelling experiments (01–09) and their reports"],
    ], [0.3, 0.7]))
    s.append(P("Appendix B. Final model, fold by fold", "h1"))
    s.append(table([
        ["Fold", "MAPE (%)", "RMSE ($)", "MAE ($)", "Median APE (%)", "Seen cities (%)", "Unseen cities (%)", "Month 1 (%)", "Month 2 (%)", "Bias (%)"],
        ["1 Jun–Jul", "1.313", "47.1", "32.1", "1.107", "1.284", "1.468", "1.327", "1.299", "−0.21"],
        ["2 Jul–Aug", "1.494", "51.0", "35.2", "1.263", "1.461", "1.755", "1.379", "1.612", "+0.89"],
        ["3 Aug–Sep", "1.281", "43.7", "30.0", "1.078", "1.263", "1.428", "1.304", "1.258", "+0.28"],
        ["4 Sep–Oct", "1.489", "49.0", "34.4", "1.281", "1.471", "1.606", "1.259", "1.711", "−0.78"],
        ["Mean", "1.394", "47.7", "32.9", "1.182", "1.370", "1.564", "1.317", "1.470", "+0.05"],
    ], [0.12] + [0.0978] * 9, num_cols=tuple(range(1, 10)), bold_rows=(5,)))
    s.append(P("Bias = median of prediction ÷ actual − 1. Folds 2 and 4 show the drift question: fold 2 would have preferred stopping the drift, "
               "fold 4 continuing it.", "small"))
    return s


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    if not (ROOT / "scorer_results/candidate_december.png").exists():
        sys.exit("run python predict.py first (it creates scorer_results/candidate_december.png)")
    set_style()
    fig_folds(FIG / "folds.png")
    fig_models(FIG / "models.png")
    register_fonts()
    doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.8 * inch, rightMargin=0.8 * inch, topMargin=0.75 * inch,
                            bottomMargin=0.8 * inch, title=TITLE, author=AUTHOR)
    doc.build(story(), onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
