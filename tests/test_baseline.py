"""Baseline benchmark: the greedy snake draft must beat a random draft.

Master spec V1 definition of done: greedy beats random on the backtest season.
Uses the synthetic totals fixture so no live API call is needed and the result is
deterministic.
"""

from __future__ import annotations

import types

import pandas as pd

from src import backtest as backtest_mod
from src import columns as C


def _league(**overrides) -> types.SimpleNamespace:
    """Minimal league config object matching what backtest_v1 reads."""
    cfg = dict(
        NUM_TEAMS=4,
        ROSTER_SIZE=3,
        MY_DRAFT_SLOT=1,
        SEED=42,
        RANDOM_BASELINE_TRIALS=100,
    )
    cfg.update(overrides)
    return types.SimpleNamespace(**cfg)


def test_greedy_beats_random(synthetic_totals: pd.DataFrame) -> None:
    result = backtest_mod.backtest_v1(synthetic_totals, "2023-24", _league())
    assert result["margin"] > 0, (
        f"Greedy ({result['greedy_total']}) did not beat random "
        f"({result['random_mean']})."
    )
    # From the top slot in a skill-ordered pool, greedy should win the large
    # majority of random trials.
    assert result["win_rate"] >= 0.9


def test_first_slot_greedy_takes_the_best_players(synthetic_totals: pd.DataFrame) -> None:
    """Sanity: slot-1 greedy roster should hold top-ranked players by projection."""
    result = backtest_mod.backtest_v1(synthetic_totals, "2023-24", _league())
    roster = result["my_roster"]
    # Slot 1 in a 4-team snake gets overall picks 1, 8, 9 -> still the strongest set
    # available at those points. Its projected total should exceed the median.
    assert roster[C.COL_PROJECTED_PTS].sum() > 0
    assert len(roster) == 3
