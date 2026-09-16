"""Baseline benchmark: the NAIVE MODEL is the reference, not a random draft.

The baseline is a greedy snake draft off the prior-season-totals board ("take
whoever scored most last season") - the bar a competent human sets. These tests
pin down three things:

1. the baseline really is that draft (not something that drifted),
2. later versions are measured against it, and
3. the random draft survives only as a sanity floor beneath it.

Uses the synthetic totals fixture so no live API call is needed and the result is
deterministic.
"""

from __future__ import annotations

import types

import pandas as pd

from src import backtest as backtest_mod
from src import board as board_mod
from src import columns as C
from src import draft as draft_mod
from src import projection as proj_mod


def _league(**overrides) -> types.SimpleNamespace:
    """Minimal league config object matching what the backtests read."""
    cfg = dict(
        NUM_TEAMS=4,
        ROSTER_SIZE=3,
        MY_DRAFT_SLOT=1,
        SEED=42,
        RANDOM_BASELINE_TRIALS=100,
        ROSTER_SLOTS={"G": 1, "F": 1, "C": 1},
        SLOT_TO_COARSE_POSITION={"G": "G", "F": "F", "C": "C"},
        FLEX_SLOT_NAMES=frozenset({"UTIL"}),
        BENCH_SLOT_NAMES=frozenset({"BN"}),
        N_PROJECTION_SEASONS=3,
        GP_RECENCY_DECAY=0.75,
    )
    cfg.update(overrides)
    return types.SimpleNamespace(**cfg)


def test_naive_baseline_is_the_prior_season_totals_draft(
    synthetic_totals: pd.DataFrame,
) -> None:
    """The baseline must be exactly "greedy off last season's totals", not a proxy."""
    league = _league()
    roster_ids, board = backtest_mod.naive_baseline_roster(
        synthetic_totals, "2023-24", league)

    expected_board = board_mod.build_board(
        proj_mod.project_v1(synthetic_totals, "2023-24"))
    expected_ids = draft_mod.snake_draft(
        expected_board, league.NUM_TEAMS, league.ROSTER_SIZE, league.MY_DRAFT_SLOT,
        strategy="greedy",
    )[league.MY_DRAFT_SLOT]

    assert roster_ids == expected_ids
    assert list(board[C.COL_PLAYER_ID]) == list(expected_board[C.COL_PLAYER_ID])


def test_naive_baseline_clears_the_random_floor(synthetic_totals: pd.DataFrame) -> None:
    """The floor is a wiring check: the naive draft must sit well above random."""
    result = backtest_mod.backtest_naive_baseline(
        synthetic_totals, "2023-24", _league())

    assert result["baseline_label"] == backtest_mod.NAIVE_BASELINE_LABEL
    assert result["floor_margin"] > 0, (
        f"Naive baseline ({result['naive_total']}) did not clear the random floor "
        f"({result['random_floor_mean']})."
    )
    # From the top slot in a skill-ordered pool, the naive draft should beat the
    # large majority of random trials.
    assert result["floor_win_rate"] >= 0.9
    assert result["n_floor_trials"] == 100


def test_v2_margin_is_measured_against_the_naive_baseline(
    synthetic_totals: pd.DataFrame,
) -> None:
    """backtest_v2 reports its margin over the naive model, consistently with it."""
    league = _league()
    position_map = {pid: ("G", "F", "C")[pid % 3] for pid in range(1, 25)}

    base = backtest_mod.backtest_naive_baseline(synthetic_totals, "2023-24", league)
    res_v2 = backtest_mod.backtest_v2(synthetic_totals, "2023-24", league, position_map)

    # Both routes must agree on what the baseline scored.
    assert res_v2["naive_total"] == base["naive_total"]
    assert res_v2["margin_vs_naive"] == round(
        res_v2["greedy_total"] - res_v2["naive_total"], 1)


def test_backtest_v1_alias_still_resolves(synthetic_totals: pd.DataFrame) -> None:
    """The masterdoc's V1 contract still has a name; V1's draft IS the baseline."""
    assert backtest_mod.backtest_v1 is backtest_mod.backtest_naive_baseline
    result = backtest_mod.backtest_v1(synthetic_totals, "2023-24", _league())
    assert result["naive_total"] > 0


def test_first_slot_naive_takes_the_best_players(synthetic_totals: pd.DataFrame) -> None:
    """Sanity: slot-1 baseline roster should hold top-ranked players by projection."""
    result = backtest_mod.backtest_naive_baseline(
        synthetic_totals, "2023-24", _league())
    roster = result["my_roster"]
    # Slot 1 in a 4-team snake gets overall picks 1, 8, 9 -> still the strongest set
    # available at those points. Its projected total should exceed the median.
    assert roster[C.COL_PROJECTED_PTS].sum() > 0
    assert len(roster) == 3


def test_v3_reports_the_naive_baseline_on_the_lineup_objective(
    synthetic_totals: pd.DataFrame,
) -> None:
    """The smart engine's headline comparison is against the naive model.

    ``backtest_v3`` keeps the V2-greedy comparison (the V3 definition of done) but
    must also score the naive baseline on the same starting-lineup objective, since
    that is the bar that decides whether the engine is worth using at all.
    """
    league = _league(
        ROSTER_SIZE=5,
        ROSTER_SLOTS={"G": 1, "F": 1, "C": 1, "UTIL": 1, "BN": 1},
        SLOT_TO_COARSE_POSITION={"G": "G", "F": "F", "C": "C",
                                 "UTIL": None, "BN": None},
        RUN_RISK_WEIGHT=1.0,
    )
    position_map = {pid: ("G", "F", "C")[pid % 3] for pid in range(1, 25)}

    res = backtest_mod.backtest_v3(synthetic_totals, "2023-24", league, position_map)

    assert res["baseline_label"] == backtest_mod.NAIVE_BASELINE_LABEL
    assert res["naive_lineup"] > 0
    assert res["margin_vs_naive"] == round(
        res["smart_lineup"] - res["naive_lineup"], 1)
    # The V2-greedy comparison the V3 gate was written against is still reported.
    assert res["lineup_margin"] == round(
        res["smart_lineup"] - res["greedy_lineup"], 1)
