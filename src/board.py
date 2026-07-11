"""Draft board: rank players by projected value.

In V1 the ranking IS the board - players sorted by projected total, descending,
with no value-over-replacement adjustment. V2 adds :func:`build_board_vor`, which
subtracts a per-position replacement-level baseline so the board reflects
positional scarcity (a centre worth 1500 points is more valuable than a guard
worth 1500 if good centres are scarcer).
"""

from __future__ import annotations

import pandas as pd

from src import columns as C
from src import positions as pos_mod


def build_board(projection: pd.DataFrame) -> pd.DataFrame:
    """Rank players by projected total, best first.

    Args:
        projection: output of :func:`src.projection.project_v1` (one row per
            player with ``PROJECTED_PTS``).

    Returns:
        The projection sorted descending by ``PROJECTED_PTS`` with a 1-based
        ``RANK`` column (1 = best available).
    """
    return (
        projection
        .sort_values(C.COL_PROJECTED_PTS, ascending=False, ignore_index=True)
        .assign(**{C.COL_RANK: lambda df: df.index + 1})
    )


def _starters_per_team(league) -> dict[str, float]:
    """Derive coarse-position starter counts per team from the roster slots.

    Rigid slots (PG/SG/G -> G, SF/PF/F -> F, C -> C) count directly; flexible
    slots (UTIL/BN/IL) are spread evenly across the three coarse positions. Tied to
    league size and roster shape, exactly as the V2 spec asks.
    """
    rigid = {p: 0.0 for p in pos_mod.COARSE_POSITIONS}
    flexible = 0.0
    for slot, count in league.ROSTER_SLOTS.items():
        coarse = league.SLOT_TO_COARSE_POSITION.get(slot)
        if coarse in rigid:
            rigid[coarse] += count
        else:
            flexible += count
    flex_each = flexible / len(pos_mod.COARSE_POSITIONS)
    return {p: rigid[p] + flex_each for p in pos_mod.COARSE_POSITIONS}


def _replacement_value(values: pd.Series, index_1based: int) -> float:
    """Return the projected value at the replacement rank (1-based), clamped.

    The replacement-level player is the best player at a position who is expected
    to go *undrafted as a starter*. If there are fewer players than the index, the
    weakest available defines replacement; an empty pool gives 0.
    """
    ordered = values.sort_values(ascending=False).to_numpy()
    if len(ordered) == 0:
        return 0.0
    return float(ordered[min(index_1based, len(ordered)) - 1])


def build_board_vor(
    projection: pd.DataFrame,
    position_map: dict[int, str],
    league,
) -> pd.DataFrame:
    """Rank players by value over replacement (V2).

    For each coarse position, the replacement level is the projected value of the
    player at rank ``round(NUM_TEAMS x starters_per_team[pos])`` within that
    position. A player's VOR is his projected total minus his position's
    replacement level. Players with no known position (``UNK``) are scored against
    an overall replacement level rather than dropped, and their count is reported.

    Args:
        projection: a projection frame with ``PROJECTED_PTS`` (V2 recommended).
        position_map: ``{player_id: "G"|"F"|"C"}`` from
            :func:`src.positions.coarse_position_map`.
        league: the :mod:`src.config` module (NUM_TEAMS, ROSTER_SLOTS, mapping).

    Returns:
        The projection with added ``COARSE_POS`` and ``VOR`` columns, plus a 1-based
        ``RANK`` sorted by VOR descending. The frame carries an ``n_unknown_pos``
        attribute (via ``.attrs``) for surfacing unmatched players.
    """
    board = projection.assign(**{
        C.COL_COARSE_POSITION: lambda df: (
            df[C.COL_PLAYER_ID].map(position_map).fillna(pos_mod.UNKNOWN_POSITION)
        )
    })

    starters = _starters_per_team(league)
    # Per-position replacement levels.
    replacement: dict[str, float] = {}
    for position in pos_mod.COARSE_POSITIONS:
        pool = board.loc[board[C.COL_COARSE_POSITION] == position, C.COL_PROJECTED_PTS]
        index_1based = max(1, round(league.NUM_TEAMS * starters[position]))
        replacement[position] = _replacement_value(pool, index_1based)

    # Overall replacement level for unknown-position players (total starters/team).
    total_starters = sum(starters.values())
    overall_index = max(1, round(league.NUM_TEAMS * total_starters))
    overall_replacement = _replacement_value(board[C.COL_PROJECTED_PTS], overall_index)

    def _baseline(position: str) -> float:
        return replacement.get(position, overall_replacement) if position != pos_mod.UNKNOWN_POSITION else overall_replacement

    out = (
        board
        .assign(**{
            C.COL_VOR: lambda df: df[C.COL_PROJECTED_PTS] - df[C.COL_COARSE_POSITION].map(_baseline)
        })
        .sort_values(C.COL_VOR, ascending=False, ignore_index=True)
        .assign(**{C.COL_RANK: lambda df: df.index + 1})
    )
    out.attrs["n_unknown_pos"] = int((board[C.COL_COARSE_POSITION] == pos_mod.UNKNOWN_POSITION).sum())
    out.attrs["replacement_levels"] = replacement
    return out
