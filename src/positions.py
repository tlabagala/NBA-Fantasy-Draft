"""Coarse positions for the V2 value-over-replacement baseline.

The nba_api roster endpoint reports positions like ``"G"``, ``"F"``, ``"C"``,
``"F-C"``, ``"G-F"``. V2 needs only a single coarse bucket per player - ``G``,
``F`` or ``C`` - taken from the primary (first) listed position. This is a
deliberate simplification: true Yahoo multi-position eligibility is V5. Players we
cannot place get ``UNK`` and are handled (and counted), never silently dropped.
"""

from __future__ import annotations

import pandas as pd

from src import columns as C

COARSE_POSITIONS: tuple[str, ...] = ("G", "F", "C")
UNKNOWN_POSITION: str = "UNK"


def to_coarse_position(raw: str | float | None) -> str:
    """Map a raw roster position string to a coarse bucket {G, F, C} or UNK.

    Uses the primary (first) listed position, e.g. ``"F-C" -> "F"``, ``"G-F" -> "G"``.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return UNKNOWN_POSITION
    primary = str(raw).strip().upper().split("-")[0]
    return primary if primary in COARSE_POSITIONS else UNKNOWN_POSITION


def coarse_position_map(rosters: pd.DataFrame) -> dict[int, str]:
    """Build a player_id -> coarse-position map from a raw roster frame.

    Args:
        rosters: output of :func:`src.ingest.load_team_rosters` (has ``PLAYER_ID``
            and ``POSITION``).

    Returns:
        ``{player_id: "G"|"F"|"C"}``. Players whose position cannot be parsed are
        omitted from the map (callers treat a missing key as ``UNK``).
    """
    parsed = (
        rosters
        .assign(**{C.COL_COARSE_POSITION: lambda df: df[C.COL_POSITION].map(to_coarse_position)})
        .loc[lambda df: df[C.COL_COARSE_POSITION] != UNKNOWN_POSITION]
    )
    return dict(zip(parsed[C.COL_PLAYER_ID], parsed[C.COL_COARSE_POSITION]))


def coarse_position_map_over_seasons(
    rosters_by_season: list[tuple[str, pd.DataFrame]],
) -> dict[int, str]:
    """Combine coarse positions across several seasons, most recent winning.

    A player who missed an entire season (e.g. injured) is absent from that
    season's rosters but present in an earlier one, so combining across the whole
    projection window covers far more players than any single season. Seasons are
    passed most-recent-first; a more recent known position overrides an older one.

    Args:
        rosters_by_season: ``[(season, rosters_df), ...]`` ordered most recent first.

    Returns:
        ``{player_id: "G"|"F"|"C"}`` merged across all supplied seasons.
    """
    combined: dict[int, str] = {}
    for _season, rosters in reversed(rosters_by_season):  # apply oldest first
        combined.update(coarse_position_map(rosters))     # newest applied last, wins
    return combined
