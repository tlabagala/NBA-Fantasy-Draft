"""V2 value-over-replacement board tests.

Checks the scarcity logic: with equal raw projections, a player at a scarcer
position should out-rank one at an abundant position, and unknown-position players
must be scored (against an overall baseline), never dropped.
"""

from __future__ import annotations

import types

import pandas as pd

from src import board as board_mod
from src import columns as C
from src import positions as pos_mod


def _league() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        NUM_TEAMS=2,
        ROSTER_SLOTS={"G": 1, "F": 1, "C": 1},   # 1 of each per team -> 2 drafted per pos
        SLOT_TO_COARSE_POSITION={"G": "G", "F": "F", "C": "C"},
    )


def _projection(values: dict[int, float]) -> pd.DataFrame:
    return pd.DataFrame({
        C.COL_PLAYER_ID: list(values),
        C.COL_PLAYER_NAME: [f"P{pid}" for pid in values],
        C.COL_PROJECTED_PTS: list(values.values()),
    })


def test_unknown_positions_are_scored_not_dropped() -> None:
    proj = _projection({1: 1000.0, 2: 900.0, 3: 800.0})
    board = board_mod.build_board_vor(proj, position_map={}, league=_league())
    assert len(board) == 3                       # nobody dropped
    assert board.attrs["n_unknown_pos"] == 3     # all unmatched, surfaced
    assert (board[C.COL_COARSE_POSITION] == pos_mod.UNKNOWN_POSITION).all()


def test_multi_season_position_merge_prefers_recent() -> None:
    older = pd.DataFrame({C.COL_PLAYER_ID: [1, 2], C.COL_POSITION: ["G", "F"]})
    recent = pd.DataFrame({C.COL_PLAYER_ID: [1, 3], C.COL_POSITION: ["C", "G"]})
    # Most-recent-first ordering; player 1 should take the recent "C".
    merged = pos_mod.coarse_position_map_over_seasons([("2022-23", recent), ("2021-22", older)])
    assert merged == {1: "C", 2: "F", 3: "G"}


def test_scarcer_position_outranks_equal_projection() -> None:
    # Centres: only two with any value, so replacement at C is low -> high VOR.
    # Guards: a deep pool, so replacement at G is high -> lower VOR for equal points.
    values = {
        # guards (deep, strong replacement)
        10: 1000.0, 11: 980.0, 12: 970.0, 13: 960.0, 14: 950.0,
        # centres (shallow, weak replacement)
        20: 1000.0, 21: 300.0, 22: 200.0,
    }
    pos = {10: "G", 11: "G", 12: "G", 13: "G", 14: "G", 20: "C", 21: "C", 22: "C"}
    board = board_mod.build_board_vor(_projection(values), pos, _league())
    by_id = board.set_index(C.COL_PLAYER_ID)
    # Guard 10 and Centre 20 both project 1000, but the centre is scarcer.
    assert by_id.at[20, C.COL_VOR] > by_id.at[10, C.COL_VOR]
    assert by_id.at[20, C.COL_RANK] < by_id.at[10, C.COL_RANK]   # centre ranked higher
