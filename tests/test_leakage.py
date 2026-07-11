"""Leakage guard: a projection for season N must use no season-N (or later) data.

This is the non-negotiable time-ordering rule (master spec Sections 3, 7).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import columns as C
from src import projection as proj_mod


def test_projection_uses_no_target_season_rows(synthetic_totals: pd.DataFrame) -> None:
    """project_v1 must draw only from seasons strictly before the target."""
    target = "2023-24"
    projection = proj_mod.project_v1(synthetic_totals, target)

    # The projection values must equal the PRIOR season's totals, never the
    # target season's. If any target-season total leaked in, at least one
    # projected value would match a target-season total that differs from prior.
    prior = proj_mod.prior_season(target)  # "2022-23"
    prior_totals = (
        synthetic_totals
        .loc[synthetic_totals[C.COL_SEASON] == prior]
        .set_index(C.COL_PLAYER_ID)[C.COL_TOTAL_FANTASY_PTS]
        .to_dict()
    )
    for _, row in projection.iterrows():
        assert row[C.COL_PROJECTED_PTS] == prior_totals[row[C.COL_PLAYER_ID]]


def test_guard_fires_when_target_data_present() -> None:
    """If the guard is bypassed by mislabelling, the assertion must trip.

    We simulate a leak by passing a target equal to the earliest season, so there
    is genuinely no prior data, and confirm the projection is empty rather than
    silently borrowing from the target season.
    """
    df = pd.DataFrame({
        C.COL_SEASON: ["2023-24", "2023-24"],
        C.COL_PLAYER_ID: [1, 2],
        C.COL_PLAYER_NAME: ["A", "B"],
        C.COL_TOTAL_FANTASY_PTS: [1000.0, 900.0],
        C.COL_GAMES_PLAYED: [70, 70],
    })
    # Target is the only season present: no prior data is allowed in, so the
    # projection must be empty (no target-season rows borrowed).
    projection = proj_mod.project_v1(df, "2023-24")
    assert projection.empty


def test_guard_raises_under_regression(
    synthetic_totals: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force a leak (prior season == target) and confirm the guard trips.

    Simulates a regression where the season-selection logic is broken so that
    target-season rows would be consumed. The assertion must fire rather than
    silently projecting from the future.
    """
    monkeypatch.setattr(proj_mod, "prior_season", lambda season: season)
    with pytest.raises(AssertionError, match="Leakage"):
        proj_mod.project_v1(synthetic_totals, "2023-24")


def test_prior_season_arithmetic() -> None:
    assert proj_mod.prior_season("2023-24") == "2022-23"
    assert proj_mod.prior_season("2000-01") == "1999-00"
