"""Markdown + figure reports for the modelling experiments (same look as the EDA reports)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from cycler import cycler  # noqa: E402

from .data import ROOT  # noqa: E402

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948"
SERIES = [BLUE, ORANGE, AQUA, YELLOW]


def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "text.color": INK, "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 1.6,
        "lines.markersize": 4.5, "font.size": 9, "axes.prop_cycle": cycler(color=SERIES),
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    })


def _fmt(v, digits: int) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.1f}" if abs(v) >= 100 else f"{v:.{digits}f}"
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    return str(v)


def md_table(df: pd.DataFrame, digits: int = 3, index: bool = True) -> str:
    d = df.reset_index() if index else df
    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "| " + " | ".join("---" for _ in d.columns) + " |"]
    lines += ["| " + " | ".join(_fmt(v, digits) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(lines)


class Report:
    def __init__(self, slug: str, title: str, intro: str = "", out_dir: Path = ROOT / "reports" / "model") -> None:
        self.slug, self.out_dir = slug, out_dir
        self.fig_dir = out_dir / "figures"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.lines = [f"# {title}", ""] + ([intro, ""] if intro else [])

    def h2(self, t: str) -> None:
        self.lines += [f"## {t}", ""]

    def h3(self, t: str) -> None:
        self.lines += [f"### {t}", ""]

    def p(self, t: str) -> None:
        self.lines += [t, ""]

    def bullets(self, items) -> None:
        self.lines += [f"- {i}" for i in items] + [""]

    def table(self, df: pd.DataFrame, digits: int = 3, index: bool = True) -> None:
        self.lines += [md_table(df, digits, index), ""]

    def figure(self, fig, name: str, caption: str) -> None:
        path = self.fig_dir / f"{self.slug}__{name}.png"
        fig.savefig(path)
        plt.close(fig)
        self.lines += [f"![{caption}](figures/{path.name})", "", f"*{caption}*", ""]

    def save(self) -> Path:
        path = self.out_dir / f"{self.slug}.md"
        path.write_text("\n".join(self.lines))
        print(f"wrote {path.relative_to(ROOT)}")
        return path
