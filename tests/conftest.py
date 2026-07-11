"""Shared pytest fixtures and path setup.

Puts the repo root on sys.path so `import src...` works when running `pytest`
from anywhere, and provides a small synthetic season-totals frame so the tests
never need a live nba_api call.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import columns as C  # noqa: E402


@pytest.fixture
def synthetic_totals() -> pd.DataFrame:
    """Three seasons of season totals for a handful of players.

    Designed so the greedy draft must beat random: a few clearly-superior players
    persist across seasons, so taking the best available each pick wins.
    """
    rows = []
    # 24 players, stable skill ordering across seasons so prior-season totals are
    # informative. Player i has a base value; small per-season wobble keeps it real.
    seasons = ["2021-22", "2022-23", "2023-24"]
    for season_idx, season in enumerate(seasons):
        for pid in range(1, 25):
            base = 2000 - pid * 60          # strict skill ladder
            wobble = ((pid + season_idx) % 5) * 10
            rows.append({
                C.COL_SEASON: season,
                C.COL_PLAYER_ID: pid,
                C.COL_PLAYER_NAME: f"Player {pid:02d}",
                C.COL_TOTAL_FANTASY_PTS: float(base + wobble),
                C.COL_GAMES_PLAYED: 70,
            })
    return pd.DataFrame(rows)
