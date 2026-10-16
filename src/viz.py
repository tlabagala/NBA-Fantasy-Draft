"""Presentation charts with plotnine (V5): aging curve, value-by-position, backtest.

These render to PNGs under ``outputs/`` (git-ignored). They use only nba_api-derived
data, so they run without any Yahoo credentials. A non-interactive matplotlib
backend is forced so saving works in a headless/CLI context.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: render straight to file, no GUI

import pandas as pd
import plotnine as p9

from src import columns as C

OUTPUT_DIR: Path = Path(__file__).resolve().parent.parent / "outputs"


def _ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def plot_aging_curve(curve: dict[int, float], target_season: str) -> Path:
    """Line chart of the age -> PPG multiplier curve, flagging survivorship bias."""
    data = pd.DataFrame({"age": list(curve), "multiplier": list(curve.values())})
    plot = (
        p9.ggplot(data, p9.aes("age", "multiplier"))
        + p9.geom_hline(yintercept=1.0, linetype="dashed", colour="grey")
        + p9.geom_line(colour="#2c7fb8", size=1)
        + p9.geom_point(colour="#2c7fb8", size=2)
        + p9.labs(
            title=f"Aging curve (projecting {target_season})",
            subtitle="Year-over-year PPG multiplier. Note survivorship bias: late-career "
                     "decline is understated (curve flattens/rises at the oldest ages).",
            x="Age", y="PPG multiplier (next season / this season)",
        )
        + p9.theme_minimal()
    )
    path = _ensure_output_dir() / "aging_curve.png"
    plot.save(path, width=8, height=5, dpi=120, verbose=False)
    return path


def plot_value_by_position(board: pd.DataFrame, target_season: str) -> Path:
    """Distribution of projected value by coarse position (positional scarcity)."""
    data = board.loc[board[C.COL_COARSE_POSITION] != "UNK",
                     [C.COL_COARSE_POSITION, C.COL_PROJECTED_PTS]]
    plot = (
        p9.ggplot(data, p9.aes(C.COL_COARSE_POSITION, C.COL_PROJECTED_PTS,
                               fill=C.COL_COARSE_POSITION))
        + p9.geom_boxplot(show_legend=False)
        + p9.labs(
            title=f"Projected value by position (projecting {target_season})",
            subtitle="Why value-over-replacement matters: positions differ in depth.",
            x="Coarse position", y="Projected fantasy points",
        )
        + p9.theme_minimal()
    )
    path = _ensure_output_dir() / "value_by_position.png"
    plot.save(path, width=8, height=5, dpi=120, verbose=False)
    return path


def plot_projection_accuracy(comparison: dict) -> Path:
    """Bar chart of projection MAE by method - lower is better.

    Takes the dict from :func:`src.backtest.compare_projection_accuracy_v4`.
    """
    labels = [("V1 prior total", "v1"), ("V2 PPG x GP", "v2"),
              ("V4 aging", "v4_aging_only"), ("V4 shrink", "v4_shrink_only"),
              ("V4 aging+shrink", "v4")]
    data = pd.DataFrame({
        "method": [label for label, _ in labels],
        "mae": [comparison[key]["mae"] for _, key in labels],
    })
    data["method"] = pd.Categorical(data["method"], categories=data["method"], ordered=True)
    plot = (
        p9.ggplot(data, p9.aes("method", "mae", fill="method"))
        + p9.geom_col(show_legend=False)
        + p9.geom_text(p9.aes(label="mae"), va="bottom", size=9, format_string="{:.0f}")
        + p9.labs(
            title=f"Projection accuracy, target {comparison['target_season']} "
                  f"(n={comparison['n_common']})",
            subtitle="Mean absolute error for one target season - lower is better. "
                     "Pooled over 13 seasons, V4 aging beats the V1 baseline.",
            x="Method", y="MAE (fantasy points)",
        )
        + p9.theme_minimal()
        + p9.theme(axis_text_x=p9.element_text(rotation=20, ha="right"))
    )
    path = _ensure_output_dir() / "projection_accuracy.png"
    plot.save(path, width=8, height=5, dpi=120, verbose=False)
    return path


def plot_backtest_distribution(multi_result: dict) -> Path:
    """Grouped bars of the smart engine's margin over each benchmark, per season.

    Benchmarks are ordered (and the legend follows) with the naive baseline first
    and the random floor last, so the chart cannot be read as though beating random
    were the result.
    """
    labels = [
        ("vs naive baseline (prior totals)", "vs_prior"),
        ("vs ADP proxy", "vs_adp"),
        ("vs random (floor only)", "vs_random_floor"),
    ]
    rows = [
        {"season": r["season"], "benchmark": label, "margin": r[key]}
        for r in multi_result["per_season"]
        for label, key in labels
    ]
    data = pd.DataFrame(rows).assign(
        benchmark=lambda df: pd.Categorical(
            df["benchmark"], categories=[label for label, _ in labels], ordered=True)
    )
    plot = (
        p9.ggplot(data, p9.aes("season", "margin", fill="benchmark"))
        + p9.geom_hline(yintercept=0.0, colour="grey")
        + p9.geom_col(position="dodge")
        + p9.labs(
            title="V3 smart engine margin by season (starting-lineup points)",
            subtitle="Baseline = the naive prior-season-totals draft. "
                     "A single season is high variance.",
            x="Target season", y="Smart engine margin over benchmark",
        )
        + p9.theme_minimal()
    )
    path = _ensure_output_dir() / "backtest_distribution.png"
    plot.save(path, width=9, height=5, dpi=120, verbose=False)
    return path
