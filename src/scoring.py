"""Apply the league scoring formula and aggregate to season totals.

V1 keeps this deliberately simple: a per-game fantasy-points column is a weighted
sum of box-score stats, and a season total is the sum of those per-game points.
Because the projection is "prior-season total", availability (games missed) is
baked in automatically - a player who only played 40 games has a smaller total.
"""

from __future__ import annotations

import pandas as pd

from src import columns as C


def _validate_scoring_keys(scoring: dict[str, float], logs: pd.DataFrame) -> None:
    """Raise if any SCORING key is not a real, scorable, present column.

    Enforces the "never fabricate" rule: a mistyped stat key fails loudly instead
    of contributing nothing to the score.
    """
    bad_keys = set(scoring) - C.SCORABLE_STAT_COLUMNS
    if bad_keys:
        raise KeyError(
            f"SCORING contains keys that are not scorable nba_api columns: "
            f"{sorted(bad_keys)}. Valid keys: {sorted(C.SCORABLE_STAT_COLUMNS)}."
        )
    missing = set(scoring) - set(logs.columns)
    if missing:
        raise KeyError(f"SCORING references columns absent from the logs: {sorted(missing)}.")


def apply_scoring(logs: pd.DataFrame, scoring: dict[str, float]) -> pd.DataFrame:
    """Add a per-game fantasy-points column (:data:`columns.COL_FANTASY_PTS`).

    Args:
        logs: per-game player logs (one row per player-game).
        scoring: stat-column -> point-weight map (validated against real columns).

    Returns:
        ``logs`` with an added float ``FANTASY_PTS`` column.
    """
    _validate_scoring_keys(scoring, logs)
    weights = pd.Series(scoring, dtype="float64")
    return logs.assign(**{
        C.COL_FANTASY_PTS: lambda df: (
            df[list(weights.index)].mul(weights, axis=1).sum(axis=1)
        )
    })


def season_totals(logs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-game logs to one row per (player, season).

    Requires :func:`apply_scoring` to have been applied first. Games played is the
    count of game rows for that player-season (verified unique per game in ingest).

    Returns:
        Columns: ``SEASON_YEAR``, ``PLAYER_ID``, ``PLAYER_NAME``,
        ``TOTAL_FANTASY_PTS`` (sum), ``GP`` (games played), sorted by total desc.
    """
    if C.COL_FANTASY_PTS not in logs.columns:
        raise KeyError("season_totals requires apply_scoring() to have been run first.")
    return (
        logs
        .groupby([C.COL_SEASON, C.COL_PLAYER_ID, C.COL_PLAYER_NAME], as_index=False)
        .agg(**{
            C.COL_TOTAL_FANTASY_PTS: (C.COL_FANTASY_PTS, "sum"),
            C.COL_GAMES_PLAYED: (C.COL_GAME_ID, "nunique"),
        })
        .sort_values(C.COL_TOTAL_FANTASY_PTS, ascending=False, ignore_index=True)
    )
