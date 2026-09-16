"""V5 multi-season distribution backtest tests (synthetic, offline)."""

from __future__ import annotations

import types

import pandas as pd

from src import backtest
from src import columns as C

_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24"]
_POSITIONS = ["G", "F", "C"]


def _league() -> types.SimpleNamespace:
    """A small, deep league so the engine's edge over random is unambiguous."""
    return types.SimpleNamespace(
        NUM_TEAMS=4, ROSTER_SIZE=5, MY_DRAFT_SLOT=1, SEED=42,
        RANDOM_BASELINE_TRIALS=100,
        ROSTER_SLOTS={"G": 1, "F": 1, "C": 1, "UTIL": 1, "BN": 1},
        SLOT_TO_COARSE_POSITION={"G": "G", "F": "F", "C": "C", "UTIL": None, "BN": None},
        FLEX_SLOT_NAMES=frozenset({"UTIL"}), BENCH_SLOT_NAMES=frozenset({"BN"}),
        N_PROJECTION_SEASONS=3, GP_RECENCY_DECAY=0.75, SHRINKAGE_K=25.0,
        AGE_CURVE_MIN=19, AGE_CURVE_MAX=38, AGE_CURVE_SMOOTH=1, RUN_RISK_WEIGHT=1.0,
    )


def _synthetic() -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str]]:
    totals, bios = [], []
    position_map: dict[int, str] = {}
    for pid in range(1, 49):
        position_map[pid] = _POSITIONS[pid % 3]
        for i, season in enumerate(_SEASONS):
            base = 2200 - pid * 35 + (pid % 4) * 8
            totals.append({
                C.COL_SEASON: season, C.COL_PLAYER_ID: pid,
                C.COL_PLAYER_NAME: f"Player {pid:02d}",
                C.COL_TOTAL_FANTASY_PTS: float(base), C.COL_GAMES_PLAYED: 72,
            })
            bios.append({C.COL_SEASON: season, C.COL_PLAYER_ID: pid,
                         C.COL_AGE: float(24 + (pid % 8) + i)})
    return pd.DataFrame(totals), pd.DataFrame(bios), position_map


def test_multi_season_reports_distribution_against_benchmarks() -> None:
    totals, bios, pos = _synthetic()
    targets = ["2022-23", "2023-24"]
    result = backtest.multi_season_backtest(
        totals, bios, targets, _league(), positions_for=lambda s: pos)

    assert result["summary"]["n_seasons"] == 2
    assert len(result["per_season"]) == 2
    for row in result["per_season"]:
        for key in ("smart", "prior_totals", "adp_proxy", "random_floor",
                    "vs_prior", "vs_adp", "vs_random_floor"):
            assert key in row
        # The engine should clear the random floor on the lineup objective.
        assert row["vs_random_floor"] > 0

    s = result["summary"]
    for key in ("vs_prior_mean", "vs_adp_mean", "vs_random_floor_mean",
                "wins_vs_prior", "baseline"):
        assert key in s


def test_naive_baseline_leads_the_reported_columns() -> None:
    """Reporting order encodes the demotion: naive baseline first, random last.

    Anything that renders these rows column-wise (main.py, the notebook, a
    DataFrame dump) inherits the order, so it is worth pinning.
    """
    totals, bios, pos = _synthetic()
    result = backtest.multi_season_backtest(
        totals, bios, ["2023-24"], _league(), positions_for=lambda s: pos)

    keys = list(result["per_season"][0])
    assert keys.index("prior_totals") < keys.index("adp_proxy")
    assert keys.index("adp_proxy") < keys.index("random_floor")
    assert keys.index("random_floor") == len(keys) - 3
    assert result["summary"]["baseline"] == backtest.NAIVE_BASELINE_LABEL
