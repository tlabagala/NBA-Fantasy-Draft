"""V4 aging-curve and project_v4 tests: shape, leakage-safety, and shrinkage."""

from __future__ import annotations

import pandas as pd

from src import columns as C
from src import config
from src import projection as proj_mod


def _totals_row(season: str, pid: int, name: str, ppg: float, gp: int = 70) -> dict:
    return {
        C.COL_SEASON: season, C.COL_PLAYER_ID: pid, C.COL_PLAYER_NAME: name,
        C.COL_TOTAL_FANTASY_PTS: ppg * gp, C.COL_GAMES_PLAYED: gp,
    }


def _bio_row(season: str, pid: int, age: float) -> dict:
    return {C.COL_SEASON: season, C.COL_PLAYER_ID: pid, C.COL_AGE: age}


def _aging_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Young players improve year over year; old players decline. Target 2023-24."""
    seasons = ["2020-21", "2021-22", "2022-23"]
    totals, bios = [], []
    # Young (age 21,22,23): rising PPG.
    for pid in range(1, 9):
        for i, season in enumerate(seasons):
            totals.append(_totals_row(season, pid, f"Young {pid}", 10.0 + 2 * i))
            bios.append(_bio_row(season, pid, 21 + i))
    # Old (age 33,34,35): falling PPG.
    for pid in range(20, 28):
        for i, season in enumerate(seasons):
            totals.append(_totals_row(season, pid, f"Old {pid}", 20.0 - 2 * i))
            bios.append(_bio_row(season, pid, 33 + i))
    return pd.DataFrame(totals), pd.DataFrame(bios)


def test_curve_rises_young_and_declines_old() -> None:
    totals, bios = _aging_dataset()
    curve = proj_mod.build_aging_curve(totals, bios, "2023-24", config)
    assert curve[21] > 1.05   # improvement in the early 20s
    assert curve[33] < 0.95   # decline in the early 30s


def test_curve_ignores_target_season_data() -> None:
    """Adding target-season rows must not change the curve (leakage-safe)."""
    totals, bios = _aging_dataset()
    curve_before = proj_mod.build_aging_curve(totals, bios, "2023-24", config)
    polluted_totals = pd.concat([totals, pd.DataFrame([
        _totals_row("2023-24", 1, "Young 1", 999.0)])], ignore_index=True)
    polluted_bios = pd.concat([bios, pd.DataFrame([
        _bio_row("2023-24", 1, 24)])], ignore_index=True)
    curve_after = proj_mod.build_aging_curve(polluted_totals, polluted_bios, "2023-24", config)
    assert curve_before == curve_after


def test_aging_factor_chains_years() -> None:
    curve = {30: 0.9, 31: 0.8}
    # Two years forward from age 30: 0.9 * 0.8.
    assert abs(proj_mod._aging_factor(30, 2, curve, config) - 0.72) < 1e-9
    assert proj_mod._aging_factor(30, 0, curve, config) == 1.0


def test_project_v4_is_leakage_safe() -> None:
    """project_v4 output is unchanged by the presence of target-season rows."""
    totals, bios = _aging_dataset()
    pos = {pid: "G" for pid in list(range(1, 9)) + list(range(20, 28))}
    base = proj_mod.project_v4(totals, "2023-24", bios, pos, config)
    polluted = pd.concat([totals, pd.DataFrame([
        _totals_row("2023-24", 1, "Young 1", 999.0)])], ignore_index=True)
    after = proj_mod.project_v4(polluted, "2023-24", bios, pos, config)
    merged = base.merge(after, on=C.COL_PLAYER_ID, suffixes=("_b", "_a"))
    assert (merged[f"{C.COL_PROJECTED_PTS}_b"] == merged[f"{C.COL_PROJECTED_PTS}_a"]).all()


def test_shrinkage_pulls_small_sample_toward_prior() -> None:
    """A high-rate, low-games player is regressed harder than a high-games one."""
    # Two guards: same rate, different sample sizes. Shrinkage should pull the
    # small-sample player further toward the (lower) positional prior.
    rows = [
        _totals_row("2022-23", 1, "Big sample", 30.0, gp=70),
        _totals_row("2022-23", 2, "Small sample", 30.0, gp=10),
        _totals_row("2022-23", 3, "Filler", 10.0, gp=70),  # drags the prior down
    ]
    bios = pd.DataFrame([_bio_row("2022-23", p, 27) for p in (1, 2, 3)])
    totals = pd.DataFrame(rows)
    pos = {1: "G", 2: "G", 3: "G"}
    proj = proj_mod.project_v4(totals, "2023-24", bios, pos, config,
                               use_aging=False, use_shrinkage=True)
    by_id = proj.set_index(C.COL_PLAYER_ID)[C.COL_PROJECTED_PPG]
    # Both shrink below 30; the 10-game player shrinks further than the 70-game one.
    assert by_id[1] < 30.0
    assert by_id[2] < by_id[1]
