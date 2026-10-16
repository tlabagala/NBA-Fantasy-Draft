"""Summary charts of the headline backtest results, for the README and write-ups.

Unlike ``src/viz.py``, these do not re-run anything. They chart the published
summary statistics from the head-to-head and multi-season tables in README.md, so
they render in seconds with no data cache or network. If a re-run of
``python src/main.py`` changes those tables, update the numbers below to match.

    python scripts/summary_figures.py

Writes ``outputs/headline_head_to_head.png`` and ``outputs/headline_measurement.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: render straight to file, no GUI

import pandas as pd
import plotnine as p9

OUTPUT_DIR: Path = Path(__file__).resolve().parent.parent / "outputs"

ACCENT = "#185fa5"   # one highlighted series ...
GREY = "#bfbfbf"     # ... against grey for everything else
INK = "#24211f"

# Head-to-head: my eleven opponents draft naively, only my version changes.
# Mean % lineup points vs the naive draft and the 95% CI, as reported in
# README.md (each season is first averaged over all 12 draft slots).
HEAD_TO_HEAD = pd.DataFrame({
    "version": [
        "V2 value-over-replacement board, greedy",
        "V4 aging board, greedy",
        "V3 engine on V2 board",
        "V3 engine on V4 aging board",
    ],
    "pct": [-11.8, -1.4, 7.8, 14.6],
    "lo": [-21.9, -11.2, 4.3, 9.0],
    "hi": [-1.7, 8.3, 11.3, 20.3],
    "shipped": [False, False, False, True],
})

# The same engine (V3 on the V2 board) measured two ways.
MEASUREMENT = pd.DataFrame({
    "design": [
        "Every team drafts off the same board",
        "Opponents held fixed (head-to-head)",
    ],
    "pct": [-0.2, 7.8],
    "lo": [-7.8, 4.3],
    "hi": [7.5, 11.3],
    "highlight": [False, True],
})


def _theme() -> p9.theme:
    return p9.theme_minimal(base_size=11, base_family="Inter") + p9.theme(
        plot_title=p9.element_text(weight="bold", size=13, ha="left", colour=INK),
        plot_subtitle=p9.element_text(colour="#595959", ha="left", size=10),
        plot_caption=p9.element_text(colour="#808080", ha="left", size=8),
        plot_title_position="plot",
        plot_caption_position="plot",
        panel_grid_minor=p9.element_blank(),
        panel_grid_major_y=p9.element_blank(),
        axis_title=p9.element_text(colour="#595959", size=9),
        axis_text=p9.element_text(colour="#737373"),
        axis_text_y=p9.element_text(colour=INK, size=10),
        legend_position="none",
        plot_background=p9.element_rect(fill="white", colour="white"),
    )


def _labels(data: pd.DataFrame, flag: str) -> list:
    """Value labels just outside each interval, on the side away from zero.

    A positive estimate is labelled past the top of its interval and a negative one
    past the bottom, so no label sits on the dashed zero line.
    """
    layers = []
    for positive in (True, False):
        side = data.loc[(data["pct"] >= 0) == positive]
        for highlighted in (True, False):
            rows = side.loc[side[flag] == highlighted]
            if rows.empty:
                continue
            layers.append(p9.geom_text(
                p9.aes(y="hi" if positive else "lo", label="pct"), data=rows,
                format_string="{:+.1f}%", nudge_y=0.6 if positive else -0.6,
                ha="left" if positive else "right", size=10, family="Inter",
                fontweight="bold" if highlighted else "normal",
                colour=ACCENT if highlighted else "#595959"))
    return layers


def _interval_chart(data: pd.DataFrame, label: str, flag: str, *, title: str,
                    subtitle: str, caption: str) -> p9.ggplot:
    data = data.assign(**{label: lambda d: pd.Categorical(
        d[label], categories=list(d[label]), ordered=True)})
    return (
        p9.ggplot(data, p9.aes(x=label, y="pct", colour=flag))
        + p9.geom_hline(yintercept=0, linetype="dashed", colour="#8c8c8c")
        + p9.geom_linerange(p9.aes(ymin="lo", ymax="hi"), size=2.2, alpha=0.45)
        + p9.geom_point(size=3.5)
        + _labels(data, flag)
        + p9.scale_colour_manual(values={True: ACCENT, False: GREY})
        + p9.scale_y_continuous(expand=(0.12, 0, 0.2, 0))
        + p9.coord_flip()
        + p9.labs(title=title, subtitle=subtitle, caption=caption, x="",
                  y="Starting-lineup points vs naive draft (%)")
        + _theme()
    )


def main() -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    charts = {
        "headline_head_to_head.png": (_interval_chart(
            HEAD_TO_HEAD, "version", "shipped",
            title="Only the draft engine beats drafting off last season's totals",
            subtitle="Edge over a naive draft with opponents held fixed: mean and 95% CI\n"
                     "across 13 seasons, each season averaged over all 12 draft slots",
            caption="Source: head_to_head_backtest, target seasons 2013-14 to 2025-26",
        ), (8, 4.4)),
        "headline_measurement.png": (_interval_chart(
            MEASUREMENT, "design", "highlight",
            title="The same engine looks worthless or clearly better, "
                  "depending on the test",
            subtitle="V3 engine on the V2 board vs a naive draft, "
                     "mean edge and 95% CI over 13 seasons",
            caption="Source: multi_season_backtest and head_to_head_backtest",
        ), (8, 3.0)),
    }
    paths = []
    for name, (plot, (width, height)) in charts.items():
        path = OUTPUT_DIR / name
        plot.save(path, width=width, height=height, dpi=200, verbose=False)
        paths.append(path)
    return paths


if __name__ == "__main__":
    for path in main():
        print(path)
