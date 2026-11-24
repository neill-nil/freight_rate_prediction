"""Build the PDF report: report/Freight_Rate_Prediction_Report.pdf

    python report/build_report.py      (run python predict.py first: the report embeds scorer_results/candidate_december.png)

Numbers in the text come from the modelling reports in reports/model/ (steps 01–09) and the final fit. The two
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
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY  # noqa: E402
from reportlab.lib.pagesizes import letter  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import inch  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image, KeepTogether, ListFlowable, ListItem, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from freight.reporting import AQUA, BLUE, INK, INK_2, MUTED, ORANGE, set_style  # noqa: E402

OUT = ROOT / "report" / "Freight_Rate_Prediction_Report.pdf"
FIG = ROOT / "report" / "figures"
TITLE = "Freight Rate Prediction"
AUTHOR = "Neill Rawani"
MARGIN = 1.0 * inch
WIDTH = letter[0] - 2 * MARGIN - 12  # frame has 6 pt padding on each side

# ---------------------------------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------------------------------
ROUNDS = [  # label, train end, test end, clean training loads, test loads
    ("Round 1", "2025-05-31", "2025-07-31", "20,076", "9,695"),
    ("Round 2", "2025-06-30", "2025-08-31", "25,320", "9,671"),
    ("Round 3", "2025-07-31", "2025-09-30", "29,689", "9,429"),
    ("Round 4", "2025-08-31", "2025-10-31", "32,904", "9,523"),
]


def fig_rounds(path: Path) -> None:
    start, day = pd.Timestamp("2025-01-01"), np.timedelta64(1, "D")
    fig, ax = plt.subplots(figsize=(8.4, 2.9))
    rows = ROUNDS + [("Final model", "2025-10-31", "2025-12-31", "47,322", "12,000")]
    for i, (lab, tr_end, te_end, ntr, nte) in enumerate(rows):
        y = len(rows) - 1 - i
        tr_end, te_end = pd.Timestamp(tr_end), pd.Timestamp(te_end)
        final = lab == "Final model"
        ax.barh(y, (tr_end + day - start).days, left=start, height=0.62, color=BLUE, edgecolor="white", linewidth=2)
        ax.barh(y, (te_end - tr_end).days, left=tr_end + day, height=0.62, color=AQUA if final else ORANGE, edgecolor="white", linewidth=2)
        ax.text(start + np.timedelta64(6, "D"), y, f"train: {ntr} loads" + ("" if final else ", 4 cities left out"),
                va="center", ha="left", fontsize=7.6, color="white")
        ax.text(tr_end + np.timedelta64(31, "D"), y, f"predict: {nte}" if final else f"test: {nte}",
                va="center", ha="center", fontsize=7.6, color=INK if final else "white")
    for qe in ("2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"):
        ax.axvline(pd.Timestamp(qe) + day, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)), zorder=0)
    ax.text(pd.Timestamp("2025-03-31") + np.timedelta64(3, "D"), len(rows) - 0.48, "end of quarter", fontsize=7, color=MUTED, va="bottom")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.set_ylim(-0.6, len(rows) - 0.25)
    ax.set_xlim(start, pd.Timestamp("2026-01-02"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[Patch(color=BLUE, label="training data"), Patch(color=ORANGE, label="test data (next two months)"),
                       Patch(color=AQUA, label="final predictions")],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


MODELS = [  # name, MAPE %, RMSE $  (future-month rounds; reports/model/05–07)
    ("Final model (regression on log price)", 1.394, 47.68),
    ("Weighted blend of all models", 1.394, 47.76),
    ("Linear, Ridge or Lasso regression", 1.395, 47.69),
    ("Final model + XGBoost on its errors", 1.395, 47.76),
    ("Final model + neural network on its errors", 1.407, 48.46),
    ("Average of final model and LightGBM", 1.455, 49.63),
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
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, k, title in ((axes[0], 1, "MAPE (%)"), (axes[1], 2, "RMSE ($)")):
        vals = [m[k] for m in MODELS][::-1]
        cols = [BLUE if n.startswith("Final model (") else "#c9c7be" for n in names]
        ax.barh(range(len(names)), vals, color=cols, height=0.66)
        for i, v in enumerate(vals):
            ax.text(v * 1.01, i, f"{v:.2f}" if k == 1 else f"{v:.1f}", va="center", fontsize=7, color=INK_2)
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
    for name, file in (("Serif", "LiberationSerif-Regular.ttf"), ("Serif-Bold", "LiberationSerif-Bold.ttf"),
                       ("Serif-Italic", "LiberationSerif-Italic.ttf"), ("Serif-BoldItalic", "LiberationSerif-BoldItalic.ttf"),
                       ("Mono", "LiberationMono-Regular.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(base / file)))
    pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-Bold", italic="Serif-Italic", boldItalic="Serif-BoldItalic")


BLACK, GREY, RULE = colors.HexColor(INK), colors.HexColor(INK_2), colors.HexColor("#555555")
S = {
    "title": ParagraphStyle("title", fontName="Serif-Bold", fontSize=20, leading=24, alignment=TA_CENTER, spaceAfter=6),
    "subtitle": ParagraphStyle("subtitle", fontName="Serif", fontSize=11.5, leading=15, alignment=TA_CENTER, spaceAfter=2),
    "author": ParagraphStyle("author", fontName="Serif", fontSize=10.5, leading=14, alignment=TA_CENTER, textColor=GREY, spaceAfter=18),
    "h1": ParagraphStyle("h1", fontName="Serif-Bold", fontSize=13.5, leading=17, spaceBefore=14, spaceAfter=6, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="Serif-Bold", fontSize=11, leading=14, spaceBefore=9, spaceAfter=4, keepWithNext=1),
    "body": ParagraphStyle("body", fontName="Serif", fontSize=10.5, leading=14.3, alignment=TA_JUSTIFY, spaceAfter=7,
                           allowWidows=0, allowOrphans=0),
    "item": ParagraphStyle("item", fontName="Serif", fontSize=10.5, leading=14.3, spaceAfter=3),
    "cap": ParagraphStyle("cap", fontName="Serif", fontSize=9, leading=11.8, textColor=GREY, spaceBefore=3, spaceAfter=12),
    "tcap": ParagraphStyle("tcap", fontName="Serif", fontSize=9, leading=11.8, textColor=GREY),
    "cell": ParagraphStyle("cell", fontName="Serif", fontSize=9.2, leading=11.6),
    "head": ParagraphStyle("head", fontName="Serif-Bold", fontSize=9.2, leading=11.6),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=8.6, leading=11.5, leftIndent=12, spaceBefore=2, spaceAfter=10),
}


def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def code(name: str) -> str:
    return f"<font name='Mono' size='9.2'>{name}</font>"


def items(texts: list[str], numbered: bool = False) -> ListFlowable:
    kw = {"bulletType": "1", "bulletFormat": "%s."} if numbered else {"bulletType": "bullet", "start": "•"}
    return ListFlowable([ListItem(P(t, "item")) for t in texts], leftIndent=16, bulletFontName="Serif", bulletFontSize=10,
                        spaceAfter=7, **kw)


def table(rows: list[list], widths: list[float], num_cols: tuple[int, ...] = (), caption: str | None = None,
          bold_last: bool = False) -> list:
    def cell(text, r, c):
        st = S["head"] if r == 0 or (bold_last and r == len(rows) - 1) else S["cell"]
        if c in num_cols:
            st = ParagraphStyle(f"{st.name}-r", parent=st, alignment=2)
        return Paragraph(str(text), st)

    data = [[cell(v, r, c) for c, v in enumerate(row)] for r, row in enumerate(rows)]
    h = 0  # index of the header row
    if caption:  # the caption is the table's first row, so it stays with the header if the table continues on the next page
        data = [[P(caption, "tcap")] + [""] * (len(widths) - 1)] + data
        h = 1
    t = Table(data, colWidths=[w * WIDTH for w in widths], repeatRows=h + 1, hAlign="CENTER")
    t.setStyle(TableStyle([("SPAN", (0, 0), (-1, 0))] * bool(caption) +
                          [("LINEABOVE", (0, h), (-1, h), 1.0, RULE), ("LINEBELOW", (0, h), (-1, h), 0.6, RULE),
                           ("LINEBELOW", (0, -1), (-1, -1), 1.0, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 3.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
                           ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
                          + ([("LINEABOVE", (0, -1), (-1, -1), 0.6, RULE)] if bold_last else [])))
    # short tables stay on one page; longer ones may continue on the next page (caption and header repeated)
    return [KeepTogether([t]), Spacer(1, 10)] if len(rows) <= 5 else [t, Spacer(1, 10)]


def figure(path: Path, caption: str, width: float = 1.0, head: str | None = None, head_style: str = "h2") -> KeepTogether:
    img = plt.imread(path)
    h, w = img.shape[:2]
    im = Image(str(path), width=WIDTH * width, height=WIDTH * width * h / w)
    return KeepTogether(([P(head, head_style)] if head else []) + [im, P(caption, "cap")])


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Serif", 9)
    canvas.setFillColor(GREY)
    canvas.drawCentredString(letter[0] / 2, 0.6 * inch, str(doc.page))
    canvas.restoreState()


def story() -> list:
    s: list = []
    s += [P(TITLE, "title"), P("Machine Learning Engineer Take-Home Assessment, Spotter", "subtitle"),
          P(f"{AUTHOR}, November 2026", "author")]

    # summary ---------------------------------------------------------------------------------------------------------
    s.append(P("Summary", "h1"))
    s.append(P("The goal of this project is to predict the posted rate (the price) of truckload freight loads. I trained a model on 48,000 "
               "loads from January to October 2025 and used it to predict 12,000 loads from November and December 2025."))
    s.append(P("Since the task is to predict future months, I tested every model the way it will be used: train on earlier months and "
               "predict the next two months. In each test I also left a few cities out of the training data, because about 12% of the "
               "validation loads involve cities that do not appear in the training data."))
    s.append(P("The final model is a linear regression on the logarithm of the price. It uses distance, equipment type, weight, the market "
               "index, the pickup and delivery cities, the route, the time of the quarter and a slow upward trend in prices. On the future "
               "test months it was off by 1.39% on average (MAPE), with a typical dollar error of $47.70 (RMSE). I also tried gradient-boosted "
               "trees (LightGBM, XGBoost), random forests, k-nearest neighbours, a neural network and several ensembles. None of them did better."))
    s.append(P(f"All results can be reproduced with one command, {code('python predict.py')}, which takes about three seconds."))

    # 1 ----------------------------------------------------------------------------------------------------------------
    s.append(P("1. Data and how accuracy is measured", "h1"))
    s.append(P(f"The training file ({code('train_test.csv')}) has 48,000 loads with known prices, dated 1 January to 31 October 2025. The "
               f"validation file ({code('validation.csv')}) has 12,000 loads dated 1 November to 31 December 2025, without prices. Each load "
               "has a pickup and delivery city with their coordinates, the distance, the equipment type (Dry Van, Reefer or Flatbed), the "
               f"weight, the date, a market index and a column called {code('quote_signal')}."))
    s.append(P("The scoring metric was not given, so I report two measures for every model. Lower is better for both."))
    s.append(items([
        "<b>MAPE</b> (mean absolute percentage error) is the average error as a percentage of the true price. A 2% miss counts the same on "
        "a $300 load and on a $3,000 load.",
        "<b>RMSE</b> (root mean squared error) is the typical error in dollars. Large misses count more, so long and expensive loads have "
        "the most influence.",
    ]))

    # 2 ----------------------------------------------------------------------------------------------------------------
    s.append(P("2. What the data showed", "h1"))
    s.append(P(f"I explored the data in 12 steps (scripts in {code('eda/')}, written reports in {code('reports/eda/')}). These findings "
               "shaped the model:"))
    s.append(items([
        "Price is mostly explained by distance, equipment, weight, the market index and the two cities. These effects multiply rather "
        "than add. For example, a Reefer load costs about 12.5% more than a Dry Van load at any distance. This is why the model works with "
        "the logarithm of the price.",
        "Prices rise slowly over time, by about 0.6% per month, and rise again over the last four weeks of each quarter (by about 2% for "
        "Dry Van and up to 8% for Flatbed). December is the end of a quarter, so the model has to predict this rise.",
        "Eight cities in the validation file never appear in the training data. They account for 12% of the validation loads.",
        f"The {code('quote_signal')} column looks very useful but misleads the model on new months (Section 4.1).",
        "The number of loads per day, city or route had no effect on price, and neither did public holidays.",
    ], numbered=True))
    s.append(P("Once these effects are taken out, what is left looks like random noise of about 1.5 to 2% of the price."))

    # 3 ----------------------------------------------------------------------------------------------------------------
    s.append(P("3. Data quality issues", "h1"))
    s.extend(table([
        ["Issue", "Training", "Validation", "How I handled it"],
        ["Negative weights", "292", "145", "The values match normal weights apart from the sign, so I used the absolute value."],
        ["Missing weights", "300", "165", "Filled with the median weight for that equipment type in the training data."],
        ["Missing market index", "374", "249", "The market index is almost the same for every load on a given day, so I filled it with that day's average."],
        ["Corrupted prices", "678", "unknown", "Removed from the training data (see below)."],
        ["Values cut off at fixed limits (weight at 5,000 and 47,500 lb, distance at 70 miles, coordinates of three cities)", "–", "–",
         "Kept. Weight only affects price between 20,000 and 40,000 lb, and coordinates are only used to find nearby cities."],
        ["December chart file has no market index or coordinates", "–", "31 rows",
         "Used the validation file's daily average market index for the same dates, and the coordinates of the same cities in the other files."],
        [code("quote_signal") + " misleads the model on new months", "–", "–", "Not used (Section 4.1)."],
    ], [0.28, 0.1, 0.11, 0.51], num_cols=(1, 2), caption="<b>Table 1:</b> Data quality issues and how each was handled (number of rows affected)."))
    s.append(P("About 1.4% of the training prices (678 loads) are clearly wrong: they are 2.2 to 5.6 times higher or lower than the normal "
               "price for the same route. They appear at random across routes and dates, so no model could predict them, and they would "
               "distort the fit. I removed them in two steps. First, I flagged loads whose price per mile was less than half or more than "
               "double the median for their route. Then I fitted the model and flagged loads whose price was more than 1.5 times higher or "
               "lower than the model's prediction, which caught 4 more on routes with only a few loads."))
    s.append(P("Every fix is written as a function that is applied in the same way to the training file, the validation file and the "
               "December file. This matters because the validation file has more problems than the training file (2.1% of loads are "
               "missing the market index, compared with 0.8%)."))

    # 4 ----------------------------------------------------------------------------------------------------------------
    s.append(P("4. Validation and data split", "h1"))
    s.append(P("4.1 Why I did not use a random split", "h2"))
    s.append(P("A common approach is to shuffle the data and split it at random. Here that gives misleading results, because loads from "
               f"the same days end up in both the training and the test data. The {code('quote_signal')} column shows the problem clearly. Its "
               "link with price changes direction from month to month (positive in January to March, negative in April and May, and so on), "
               "and in August there is no link at all. The validation months behave like August."))
    s.append(P(f"Figure 1 compares a gradient-boosted tree model trained with and without {code('quote_signal')}. With a random split the "
               "column seems to help: MAPE improves from 1.50% to 1.30%. But when the model is trained on January to July and tested on "
               "August, the same column doubles the error, from 1.57% to 3.13%. A random split would have led me to keep a feature that "
               "fails on the real task, so I dropped it. A random split would also hide the two other things that only happen in the "
               "future: continuing the price trend, and pricing cities that were never seen."))
    s.append(figure(ROOT / "reports/eda/figures/08_trap_proof__with_without.png",
                    f"<b>Figure 1:</b> Error of a gradient-boosted tree model with and without {code('quote_signal')}. The left group uses a "
                    "random split. The other groups train on all earlier months and test on one month (the direction of the column's link with "
                    "price in that month is in brackets). Green bars show the error when the column carries no information in the test month."))
    s.append(figure(FIG / "rounds.png", "<b>Figure 2:</b> The four test rounds and the final model. Dashed lines mark the end of each quarter.",
                    head="4.2 How the data was split"))
    s.append(P("I used a time-based split that copies the real task (Figure 2 and Table 2). There are four test rounds. In each round the "
               "model is trained on all months up to a cut-off date and tested on the two months that follow. In each round I also removed "
               "four randomly chosen cities from the training data, so that 11 to 16% of the test loads involve a city the model has never "
               "seen. This is close to the 12% in the validation file, and it tests how the model handles new cities."))
    s.extend(table([
        ["Round", "Training data", "Test data", "Training loads", "Test loads", "Test loads with an unseen city"],
        ["1", "January to May", "June and July", "20,076", "9,695", "15.6%"],
        ["2", "January to June", "July and August", "25,320", "9,671", "11.0%"],
        ["3", "January to July", "August and September", "29,689", "9,429", "11.0%"],
        ["4", "January to August", "September and October", "32,904", "9,523", "13.3%"],
        ["Final", "January to October", "November and December", "47,322", "12,000", "12.1%"],
    ], [0.08, 0.2, 0.24, 0.14, 0.12, 0.22], num_cols=(3, 4, 5), caption="<b>Table 2:</b> The test rounds. Training loads are counted after cleaning."))
    s.append(P("To keep information from the test months out of training, all cleaning steps (the fill values and the removal of "
               "corrupted prices) and all model fitting use only the training part of each round. Test loads with corrupted prices are "
               "left out of the scores, because no model can predict them. Keeping them would raise the RMSE of any model to around $630."))
    s.append(P("4.3 How I chose between models", "h2"))
    s.append(P("For every comparison I averaged MAPE and RMSE over the four rounds and also looked at the worst round. I only switched to a "
               "more complex model, or changed a setting, when it improved MAPE by at least 0.02 percentage points and RMSE by at least $1 "
               "at the same time. Smaller differences are within the normal variation between rounds, and chasing them would fit these "
               "particular test months rather than the future."))
    s.append(P("4.4 Checks on the validation itself", "h2"))
    s.append(P("I ran some extra checks to make sure the conclusions did not depend on choices made in the validation (Table 3)."))
    s.extend(table([
        ["Check", "Result for the final model"],
        ["Three other random choices of the cities left out", "MAPE 1.39% to 1.40%, RMSE $47.70 to $48.10. The ranking of the models did not change."],
        ["Two extra, earlier rounds never used for tuning (tests on April–May and May–June)", "MAPE 1.44% and 1.43%, in line with the main rounds."],
        ["Stricter or looser limits for removing corrupted prices", "Almost no change. Skipping the second removal step made RMSE $4.30 worse; "
         "no cleaning at all gave 2.33% MAPE and $104 RMSE."],
        ["Price trend after the training data ends: stops, half speed or full speed",
         "MAPE 1.40%, 1.39% and 1.45%. Half speed was never the worst choice in any round, so I used it."],
        ["A random split instead of future months", "MAPE of about 1.24%. So about 0.15 points of the final error comes from having to predict ahead."],
    ], [0.47, 0.53], caption="<b>Table 3:</b> Checks on the validation design."))
    s.append(P("4.5 Final training", "h2"))
    s.append(P("After all choices were made, I trained the final model once on all 47,322 clean loads from January to October and used it "
               "to predict the validation file and the December chart."))

    # 5 ----------------------------------------------------------------------------------------------------------------
    s.append(P("5. The model", "h1"))
    s.append(P("5.1 Building the model step by step", "h2"))
    s.append(P("I started with the simplest reasonable model and added one part at a time, keeping each part only if it reduced the error "
               "on the future test months (Table 4). Adding the market index on its own made things worse (step 3), because without a time "
               "trend it was also picking up the general rise in prices. Once the trend was added in step 6, it helped."))
    s.extend(table([
        ["Step", "Model", "MAPE (%)", "RMSE ($)"],
        ["0", "Typical price per mile times distance", "9.34", "291.6"],
        ["1", "Add a distance curve and the equipment type", "3.90", "125.3"],
        ["2", "Add weight", "3.35", "105.4"],
        ["3", "Add the market index", "3.73", "116.1"],
        ["4", "Add an effect for each city", "3.18", "103.4"],
        ["5", "Add the rise at the end of each quarter", "2.87", "90.6"],
        ["6", "Add the slow upward trend", "1.73", "59.0"],
        ["7", "Price new cities from their nearest neighbours", "1.67", "57.2"],
        ["8", "Slow the trend to half speed after the training data ends", "1.62", "55.6"],
        ["9", "Let the end-of-quarter rise differ by equipment type", "1.53", "52.3"],
        ["10", "Add an effect for each route (final model)", "1.39", "47.7"],
    ], [0.08, 0.64, 0.14, 0.14], num_cols=(2, 3), bold_last=True,
        caption="<b>Table 4:</b> Error on the future test rounds as each part is added (average of the four rounds)."))
    s.append(P("5.2 The final model", "h2"))
    s.append(P("The final model predicts the logarithm of the price as a sum of the parts in Table 5 and is fitted with ordinary least "
               "squares. Working on the log scale means each part changes the price by a percentage. This matches the data: the noise is a "
               "constant percentage of the price at every distance and for every equipment type."))
    s.extend(table([
        ["Part", "How it enters the model", "Effect in the final fit"],
        ["Distance", "A smooth curve in log distance", "Doubling the distance raises the price by about 80 to 90%"],
        ["Equipment", "One term each for Reefer and Flatbed", "Reefer +12.5%, Flatbed +7.6% compared with Dry Van"],
        ["Weight", "Straight line between 20,000 and 40,000 lb, flat outside", "+3.8% per 10,000 lb"],
        ["Market index", "Straight line", "+1.4% for every 0.1 increase"],
        ["End of quarter", "Rises in a straight line over the last 28 days of each quarter", "On the last day: Dry Van +2.4%, Reefer +5.1%, Flatbed +8.2%"],
        ["Trend", "Straight line in time; continues at half speed after October", "+0.59% per 30 days"],
        ["Cities", "One effect per city, the same for pickup and delivery", "From −3.7% (Syracuse) to +3.2% (Oklahoma City)"],
        ["Routes", "Each route's average leftover error, shrunk towards zero", "Most routes within ±1.4%"],
    ], [0.15, 0.43, 0.42], caption="<b>Table 5:</b> The parts of the final model."))
    s.append(P("<b>New cities.</b> A city that is not in the training data gets the average effect of its five nearest cities, weighted by "
               "distance. When I tested this by hiding known cities, it explained 94% of the variation in their real effects."))
    s.append(P("<b>Routes.</b> Each route (for example, Dallas to Phoenix) has its own small price level beyond its two cities. The model "
               "estimates it as the route's total leftover error divided by the number of its loads plus 3, so a route with only a few "
               "loads gets only a small adjustment. Routes not seen in training get no adjustment. Direction matters: Dallas to Phoenix and "
               "Phoenix to Dallas have separate effects."))
    s.append(P("5.3 Other models", "h2"))
    s.append(P("I compared the final model with the standard alternatives (Figure 3). Each was tuned on the same four rounds and given the "
               "same inputs, including the city and route effects. Those effects were computed so that a load's own price was never used "
               "to build its inputs."))
    s.append(figure(FIG / "models.png", "<b>Figure 3:</b> Error on the future test rounds for the best setting of each model "
                    "(average of the four rounds). The naive baseline (9.34%) is not shown."))
    s.append(P("Linear, Ridge and Lasso regression on the same inputs gave the same result as the final model, which is expected since they "
               "are the same kind of model. The tree models (LightGBM, XGBoost, random forest) were clearly worse. They predicted prices "
               "about 0.6% too low on average, because a tree model cannot continue a trend past the last date it was trained on. A neural "
               "network and k-nearest neighbours were also worse."))
    s.append(P("Combining models did not help either. Averaging the final model with a tree model made it worse. Training a second model "
               "to correct the final model's errors gave no gain, and when tuned, these correction models chose their simplest settings, "
               "which suggests there is little pattern left in the errors."))
    s.append(P("The closest result came from gradient-boosted trees trained to correct the final model's errors. On average this was "
               "about 1% better, so before deciding I tested it on six rounds (the four main ones and the two extra ones) with four different "
               "sets of hidden cities, giving 24 comparisons, against conditions I set beforehand. It won 79% of the comparisons on MAPE and "
               "71% on RMSE, below my condition of 75% on both. It lost in one of the six rounds every time, and I could not link its gain "
               "to any particular input. Since it was not reliably better and was harder to explain, I kept the simpler model."))

    # 6 ----------------------------------------------------------------------------------------------------------------
    s.append(P("6. Results", "h1"))
    s.extend(table([
        ["Test round", "MAPE (%)", "RMSE ($)", "MAPE, known cities (%)", "MAPE, new cities (%)"],
        ["1: June and July", "1.31", "47.1", "1.28", "1.47"],
        ["2: July and August", "1.49", "51.0", "1.46", "1.76"],
        ["3: August and September", "1.28", "43.7", "1.26", "1.43"],
        ["4: September and October", "1.49", "49.0", "1.47", "1.61"],
        ["Average", "1.39", "47.7", "1.37", "1.56"],
    ], [0.32, 0.14, 0.14, 0.2, 0.2], num_cols=(1, 2, 3, 4), bold_last=True,
        caption="<b>Table 6:</b> Accuracy of the final model on each test round."))
    s.append(P("The error is fairly even across the rounds, from 1.28% to 1.49%. Loads involving a new city have a slightly higher error "
               "(1.56% compared with 1.37%), because they cannot use route effects. The second test month is a little harder than the "
               "first (1.47% compared with 1.32%), as expected when predicting further ahead. The median error per load is 1.18% and "
               "the average error in dollars is $32.90."))

    # 7 ----------------------------------------------------------------------------------------------------------------
    s.append(figure(ROOT / "scorer_results/candidate_december.png",
                    f"<b>Figure 4:</b> The December prediction chart produced by {code('score.py')}.",
                    head="7. December prediction chart", head_style="h1"))
    s.append(P("The December chart keeps every input fixed except the date: Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb. "
               "Because the December file has no market index, I used the validation file's daily average market index for the same "
               "dates. The predicted price goes from $826 on 1 December to $872 on 31 December. For comparison, the median Dry Van price "
               "on this route in the training data is $808."))
    s.append(P("The chart has three features. The weekly up-and-down pattern, with highs on Thursdays and lows on Sundays, comes from the "
               "market index, which follows the same weekly cycle. The rise over the last four weeks is the end-of-quarter increase, which "
               "reaches 2.4% for Dry Van on 31 December. The slow upward trend adds about 0.3% over the month. The trend is the most "
               "uncertain part: if prices stopped rising after October, the 31 December price would be $867, and if they kept rising at "
               "full speed it would be $877."))

    # 8 ----------------------------------------------------------------------------------------------------------------
    s.append(KeepTogether([P("8. Limitations", "h1"), items([
        "The model assumes the price trend continues at half speed after October. The other reasonable choices would change the "
        "predictions by about 0.2% in November and 0.5% in December.",
        "There is no December in the training data, so holiday or year-end effects cannot be learned. The model assumes the end of "
        "December behaves like the end of March, June and September.",
        "Loads in new cities have about 14% more error, since they have no route history.",
        "If the validation prices contain corrupted values at the same rate as the training data (about 1.4%), those loads will dominate "
        "the RMSE of any model.",
    ])]))

    # 9 ----------------------------------------------------------------------------------------------------------------
    s.append(P("9. Reproducing the results", "h1"))
    s.append(Preformatted("python -m pip install -r requirements.txt\n"
                          "python predict.py                 # predictions, December file and chart\n"
                          "python report/build_report.py     # this report\n"
                          "python eda/run_all.py             # data exploration reports\n"
                          "python modeling/01_structural.py  # modelling experiments 01 to 09, run in order", S["code"]))
    s.extend(table([
        ["File", "What it does"],
        [code("predict.py"), "The final pipeline: cleans the data, removes corrupted prices, fits the model, writes the predictions and runs the scorer"],
        [code("src/freight/data.py"), "Loading and cleaning, and the rule for corrupted prices"],
        [code("src/freight/models.py"), "The baseline, the final model and the tree models"],
        [code("src/freight/evaluation.py"), "The time-based test rounds with hidden cities, and the error measures"],
        [code("src/freight/config.py"), "The final model's settings, with the experiment that chose each one"],
        [code("eda/") + ", " + code("modeling/"), "Exploration and modelling experiments; their reports are in " + code("reports/")],
    ], [0.36, 0.64], caption="<b>Table 7:</b> The main files in the repository."))
    return s


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    if not (ROOT / "scorer_results/candidate_december.png").exists():
        sys.exit("run python predict.py first (it creates scorer_results/candidate_december.png)")
    set_style()
    fig_rounds(FIG / "rounds.png")
    fig_models(FIG / "models.png")
    for old in ("folds.png",):
        (FIG / old).unlink(missing_ok=True)
    register_fonts()
    doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN,
                            bottomMargin=MARGIN, title=TITLE, author=AUTHOR)
    doc.build(story(), onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
