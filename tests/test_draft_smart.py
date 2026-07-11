"""V3 smart-engine tests: marginal value + positional-run beat greedy when depth
at a position is wasted.
"""

from __future__ import annotations

import types

import pandas as pd

from src import columns as C
from src import draft as draft_mod
from src import lineup as lineup_mod


def _league() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        NUM_TEAMS=2,
        ROSTER_SIZE=4,
        MY_DRAFT_SLOT=1,
        ROSTER_SLOTS={"G": 1, "F": 1, "C": 1, "UTIL": 1},
        SLOT_TO_COARSE_POSITION={"G": "G", "F": "F", "C": "C", "UTIL": None},
        FLEX_SLOT_NAMES=frozenset({"UTIL"}),
        BENCH_SLOT_NAMES=frozenset(),
        RUN_RISK_WEIGHT=1.0,
    )


# A board where the top is stacked with guards; a naive greedy roster ends up with
# idle guards and an empty forward slot, while the smart engine grabs the scarce
# centre and forward.
_VALUES = {1: 100.0, 2: 98.0, 3: 96.0, 4: 94.0, 5: 92.0, 6: 50.0, 7: 45.0, 8: 20.0, 9: 15.0}
_POS = {1: "G", 2: "G", 3: "G", 4: "G", 5: "G", 6: "C", 7: "F", 8: "C", 9: "F"}


def _board() -> pd.DataFrame:
    return pd.DataFrame({
        C.COL_PLAYER_ID: list(_VALUES),
        C.COL_PROJECTED_PTS: list(_VALUES.values()),
    }).sort_values(C.COL_PROJECTED_PTS, ascending=False, ignore_index=True)


def _lineup_value(roster: list[int], league) -> float:
    return lineup_mod.optimal_lineup_value(roster, _VALUES, _POS, league)


def test_smart_engine_beats_greedy_on_lineup_value() -> None:
    league = _league()
    board = _board()
    greedy = draft_mod.snake_draft(board, 2, 4, 1, strategy="greedy")[1]
    smart = draft_mod.snake_draft_smart(board, 2, 4, 1, _POS, league)[1]

    greedy_value = _lineup_value(greedy, league)
    smart_value = _lineup_value(smart, league)
    assert smart_value > greedy_value, (greedy, smart, greedy_value, smart_value)


def test_smart_draft_fills_every_team_and_is_complete() -> None:
    league = _league()
    rosters = draft_mod.snake_draft_smart(_board(), 2, 4, 1, _POS, league)
    assert set(rosters) == {1, 2}
    assert all(len(r) == 4 for r in rosters.values())
    # No player drafted twice across the whole draft.
    picked = [pid for r in rosters.values() for pid in r]
    assert len(picked) == len(set(picked))
