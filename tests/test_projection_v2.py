"""V2 projection tests: leakage guard, PPGxGP mechanics, and accuracy vs V1.

The accuracy test encodes V2's reason for existing: a player who was injured last
season but is normally productive is badly misjudged by V1's stale prior-season
total, and better handled by V2's games-played-weighted PPG over several seasons.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import backtest as backtest_mod
from src import columns as C
from src import projection as proj_mod


def _row(season: str, pid: int, name: str, total: float, gp: int) -> dict:
    return {
        C.COL_SEASON: season,
        C.COL_PLAYER_ID: pid,
        C.COL_PLAYER_NAME: name,
        C.COL_TOTAL_FANTASY_PTS: total,
        C.COL_GAMES_PLAYED: gp,
    }


@pytest.fixture
def injury_totals() -> pd.DataFrame:
    """Four seasons; some players have an injury-shortened season just before target.

    Target = 2023-24. Prior three seasons available. "Rebound" players are normally
    ~1500 (75 games x 20 PPG) but missed time in 2022-23 (20 games, 400 pts), then
    return to ~1500 in the target. Stable players are flat across all seasons.
    """
    rows: list[dict] = []
    history = ["2020-21", "2021-22", "2022-23"]
    # Stable players: same every season; actual ~ prior, both methods fine.
    for pid in range(1, 9):
        level = 1600 - pid * 50
        for season in history + ["2023-24"]:
            rows.append(_row(season, pid, f"Stable {pid}", float(level), 75))
    # Rebound players: good, then injured in 2022-23, then back in target.
    for pid in range(9, 17):
        good = 1500.0
        rows.append(_row("2020-21", pid, f"Rebound {pid}", good, 75))
        rows.append(_row("2021-22", pid, f"Rebound {pid}", good, 75))
        rows.append(_row("2022-23", pid, f"Rebound {pid}", 400.0, 20))   # injured
        rows.append(_row("2023-24", pid, f"Rebound {pid}", good, 75))     # healthy again
    return pd.DataFrame(rows)


def test_v2_leakage_guard_uses_no_target_rows(injury_totals: pd.DataFrame) -> None:
    proj = proj_mod.project_v2(injury_totals, "2023-24")
    # Sanity: every projected player is real and none of the values equals the
    # target-season total by construction of the guard (target excluded from input).
    assert not proj.empty
    assert {C.COL_PROJECTED_PPG, C.COL_PROJECTED_GP, C.COL_PROJECTED_PTS} <= set(proj.columns)


def test_v2_guard_raises_under_regression(
    injury_totals: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force the window to include the target season; the guard must trip."""
    monkeypatch.setattr(proj_mod, "recent_prior_seasons",
                        lambda totals, target, n: [target])
    with pytest.raises(AssertionError, match="Leakage"):
        proj_mod.project_v2(injury_totals, "2023-24")


def test_ppg_times_gp_equals_projected_total(injury_totals: pd.DataFrame) -> None:
    proj = proj_mod.project_v2(injury_totals, "2023-24")
    recomputed = proj[C.COL_PROJECTED_PPG] * proj[C.COL_PROJECTED_GP]
    pd.testing.assert_series_equal(
        recomputed, proj[C.COL_PROJECTED_PTS], check_names=False, rtol=1e-9,
    )


def test_v2_beats_v1_on_accuracy_for_rebound_players(injury_totals: pd.DataFrame) -> None:
    """On a season with injury-rebound players, V2's MAE must beat V1's."""
    cmp = backtest_mod.compare_projection_accuracy(injury_totals, "2023-24", league=None)
    assert cmp["v2"]["mae"] < cmp["v1"]["mae"], cmp
