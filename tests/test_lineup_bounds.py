"""Lineup scoring bounds: which information is allowed to set the starting lineup.

Scoring a roster means picking its best legal starting lineup and summing those
players' ACTUAL points - but "best" needs something to sort by, and that choice is
an assumption worth pinning down:

- ``"hindsight"``   picks starters using the actual totals (upper bound),
- ``"projection"``  picks them using the board the roster was drafted off (lower
  bound).

The invariant that makes them bounds is that hindsight can never score less than
projection-picked, because it selects the argmax of the very thing being scored.
"""

from __future__ import annotations

import types

import pandas as pd
import pytest

from src import backtest as backtest_mod
from src import columns as C
from src import lineup as lineup_mod


def _league(**overrides) -> types.SimpleNamespace:
    cfg = dict(
        NUM_TEAMS=4,
        ROSTER_SIZE=5,
        MY_DRAFT_SLOT=2,
        SEED=42,
        RANDOM_BASELINE_TRIALS=20,
        ROSTER_SLOTS={"G": 1, "F": 1, "C": 1, "UTIL": 1, "BN": 1},
        SLOT_TO_COARSE_POSITION={"G": "G", "F": "F", "C": "C",
                                 "UTIL": None, "BN": None},
        FLEX_SLOT_NAMES=frozenset({"UTIL"}),
        BENCH_SLOT_NAMES=frozenset({"BN"}),
        N_PROJECTION_SEASONS=3,
        GP_RECENCY_DECAY=0.75,
        SHRINKAGE_K=25.0,
        AGE_CURVE_MIN=19,
        AGE_CURVE_MAX=38,
        AGE_CURVE_SMOOTH=1,
        RUN_RISK_WEIGHT=1.0,
    )
    cfg.update(overrides)
    return types.SimpleNamespace(**cfg)


_POSITIONS = ("G", "F", "C")


def test_hindsight_selection_matches_the_original_behaviour() -> None:
    """score_lineup with actual as the selector must equal optimal_lineup_value.

    The hindsight bound is what every backtest did before this parameter existed,
    so it has to be bit-for-bit the same or historical numbers silently move.
    """
    league = _league()
    roster = [1, 2, 3, 4, 5, 6]
    actual = {1: 900.0, 2: 800.0, 3: 700.0, 4: 600.0, 5: 500.0, 6: 400.0}
    pos = {pid: _POSITIONS[pid % 3] for pid in roster}

    assert backtest_mod.score_lineup(roster, actual, actual, pos, league) == (
        lineup_mod.optimal_lineup_value(roster, actual, pos, league))


def test_hindsight_is_never_worse_than_projection_picked() -> None:
    """The bounds must actually bound: hindsight >= projection, always.

    Hindsight selects the argmax of the quantity being summed, so no other
    selection rule can beat it. A violation would mean the scoring is wrong.
    """
    league = _league()
    roster = list(range(1, 9))
    pos = {pid: _POSITIONS[pid % 3] for pid in roster}
    actual = {1: 100.0, 2: 900.0, 3: 200.0, 4: 800.0,
              5: 300.0, 6: 700.0, 7: 400.0, 8: 600.0}
    # A deliberately terrible projection: exactly inverted from the outcome.
    projected = {pid: 1000.0 - value for pid, value in actual.items()}

    bounds = backtest_mod.lineup_bounds(roster, actual, projected, pos, league)
    assert bounds["hindsight"] >= bounds["projection"]
    # With an inverted projection the gap should be strictly positive, not a tie.
    assert bounds["hindsight"] > bounds["projection"]


def test_a_perfect_projection_collapses_the_bounds() -> None:
    """If the projection were exactly right, both bounds must coincide."""
    league = _league()
    roster = list(range(1, 9))
    pos = {pid: _POSITIONS[pid % 3] for pid in roster}
    actual = {pid: float(1000 - 50 * pid) for pid in roster}

    bounds = backtest_mod.lineup_bounds(roster, actual, dict(actual), pos, league)
    assert bounds["hindsight"] == bounds["projection"]


def test_invalid_lineup_value_source_is_rejected() -> None:
    for bad in ("actual", "future", "", "Hindsight"):
        with pytest.raises(ValueError, match="lineup_value_source"):
            backtest_mod._check_lineup_source(bad)


def _synthetic() -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str]]:
    seasons = ["2020-21", "2021-22", "2022-23", "2023-24"]
    totals, bios = [], []
    position_map: dict[int, str] = {}
    for pid in range(1, 49):
        position_map[pid] = _POSITIONS[pid % 3]
        for i, season in enumerate(seasons):
            # Give the target season a different ordering from the priors, so the
            # projection is wrong often enough for the two bounds to diverge.
            base = 2200 - pid * 35 + (pid % 4) * 8
            if season == "2023-24":
                base = 2200 - (49 - pid) * 30 + (pid % 5) * 11
            totals.append({
                C.COL_SEASON: season, C.COL_PLAYER_ID: pid,
                C.COL_PLAYER_NAME: f"Player {pid:02d}",
                C.COL_TOTAL_FANTASY_PTS: float(base), C.COL_GAMES_PLAYED: 72,
            })
            bios.append({C.COL_SEASON: season, C.COL_PLAYER_ID: pid,
                         C.COL_AGE: float(24 + (pid % 8) + i)})
    return pd.DataFrame(totals), pd.DataFrame(bios), position_map


def test_backtest_v3_returns_both_bounds() -> None:
    totals, bios, pos = _synthetic()
    res = backtest_mod.backtest_v3(totals, "2023-24", _league(), pos)

    assert res["lineup_value_source"] == "hindsight"
    for roster_key in ("greedy", "smart", "naive"):
        b = res["lineup_bounds"][roster_key]
        assert b["hindsight"] >= b["projection"], roster_key
    # The headline numbers follow the selected source.
    assert res["smart_lineup"] == res["lineup_bounds"]["smart"]["hindsight"]

    lower = backtest_mod.backtest_v3(totals, "2023-24", _league(), pos,
                                     lineup_value_source="projection")
    assert lower["smart_lineup"] == lower["lineup_bounds"]["smart"]["projection"]
    assert lower["smart_lineup"] <= res["smart_lineup"]


def test_head_to_head_reports_both_bounds_and_they_agree_with_a_rerun() -> None:
    """The bounds carried in one run must match running with that source selected."""
    totals, bios, pos = _synthetic()
    league = _league()
    variants = (("V3", "v2", "smart"),)

    default = backtest_mod.head_to_head_backtest(
        totals, bios, ["2023-24"], league, positions_for=lambda s: pos,
        variants=variants)
    rerun = backtest_mod.head_to_head_backtest(
        totals, bios, ["2023-24"], league, positions_for=lambda s: pos,
        variants=variants, lineup_value_source="projection")

    stats = default["summary"]["variants"]["V3"]
    assert default["summary"]["lineup_value_source"] == "hindsight"
    assert stats["mean_pct"] == stats["mean_pct_hindsight"]
    # The projection bound carried in the default run must equal the dedicated run.
    assert stats["mean_pct_projection"] == pytest.approx(
        rerun["summary"]["variants"]["V3"]["mean_pct"], abs=0.1)
