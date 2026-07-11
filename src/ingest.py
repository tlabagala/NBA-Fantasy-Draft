"""Data ingestion: pull per-game player logs from ``nba_api`` and cache to Parquet.

Design rules honoured here (master spec Sections 3, 6, 7):
- Minimise calls: one league-wide ``PlayerGameLogs`` call returns every player's
  per-game logs for a whole season, so we make at most one live call per season.
- Be polite to NBA.com: a short sleep between live calls, retry-with-backoff on
  failure, and a per-call timeout.
- Cache: each season is written to ``data/`` as Parquet and never re-pulled unless
  the cache file is missing. ``data/`` is git-ignored and disposable.
- Never assume the schema: every pull (live or cached) is validated against the
  verified column constants in :mod:`src.columns`.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import (
    commonteamroster,
    leaguedashplayerbiostats,
    playergamelogs,
)
from nba_api.stats.static import teams as static_teams

from src import columns as C
from src import config

# Cache lives at <repo>/data/ . Resolve relative to this file so it works
# regardless of the current working directory.
DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"


def _cache_path(season: str) -> Path:
    """Return the Parquet cache path for a single season (e.g. 'logs_2022-23.parquet')."""
    return DATA_DIR / f"logs_{season}.parquet"


def _roster_cache_path(season: str) -> Path:
    """Return the Parquet cache path for a season's league-wide rosters (V2)."""
    return DATA_DIR / f"rosters_{season}.parquet"


def _bios_cache_path(season: str) -> Path:
    """Return the Parquet cache path for a season's player bios (V4 age source)."""
    return DATA_DIR / f"bios_{season}.parquet"


def validate_schema(df: pd.DataFrame, *, season: str) -> pd.DataFrame:
    """Assert the frame contains every column V1 depends on; return it unchanged.

    Guards against a silent ``nba_api`` schema change corrupting the pipeline. If
    a required column is missing we raise rather than guess.
    """
    missing = C.REQUIRED_RAW_COLUMNS - set(df.columns)
    if missing:
        raise KeyError(
            f"nba_api pull for {season} is missing expected columns {sorted(missing)}. "
            f"The API schema may have changed - update src/columns.py."
        )
    return df


def _pull_one_season_live(season: str) -> pd.DataFrame:
    """Pull one season's league-wide per-game logs with retry-and-backoff.

    A single call returns all players' regular-season game logs for the season,
    which keeps us well within polite request limits.
    """
    last_error: Exception | None = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            frames = playergamelogs.PlayerGameLogs(
                season_nullable=season,
                season_type_nullable=config.SEASON_TYPE,
                timeout=config.REQUEST_TIMEOUT_SECONDS,
            ).get_data_frames()
            return validate_schema(frames[0], season=season)
        except Exception as error:  # noqa: BLE001 - network/JSON errors are broad and retryable
            last_error = error
            backoff = config.BACKOFF_BASE_SECONDS ** attempt
            print(
                f"  ! pull for {season} failed (attempt {attempt}/{config.MAX_RETRIES}): "
                f"{error!r}; retrying in {backoff:.1f}s"
            )
            time.sleep(backoff)
    raise RuntimeError(f"nba_api pull for {season} failed after "
                       f"{config.MAX_RETRIES} attempts") from last_error


def load_player_logs(seasons: list[str]) -> pd.DataFrame:
    """Load per-game player logs for the given seasons, using the Parquet cache.

    For each season: read from ``data/`` if cached, otherwise pull once from
    ``nba_api``, validate the schema, write the cache, and sleep briefly before any
    further live call. Returns one concatenated frame of all requested seasons,
    one row per (player, game).

    Args:
        seasons: season strings in ``nba_api`` form, e.g. ``["2022-23", "2023-24"]``.

    Returns:
        A DataFrame with at least the columns in :data:`src.columns.REQUIRED_RAW_COLUMNS`.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    per_season: list[pd.DataFrame] = []
    made_live_call = False

    for season in seasons:
        cache = _cache_path(season)
        if cache.exists():
            print(f"  - {season}: reading cache {cache.name}")
            df = validate_schema(pd.read_parquet(cache), season=season)
        else:
            # Be polite: brief delay between consecutive live calls.
            if made_live_call:
                time.sleep(config.SLEEP_BETWEEN_CALLS_SECONDS)
            print(f"  - {season}: pulling live from nba_api ...")
            df = _pull_one_season_live(season)
            df.to_parquet(cache, index=False)
            made_live_call = True
            print(f"    cached {len(df):,} rows -> {cache.name}")
        per_season.append(df)

    return pd.concat(per_season, ignore_index=True)


def _validate_roster_schema(df: pd.DataFrame, *, team_id: int, season: str) -> pd.DataFrame:
    """Assert a roster frame contains the columns V2 needs; return it unchanged."""
    missing = C.REQUIRED_ROSTER_COLUMNS - set(df.columns)
    if missing:
        raise KeyError(
            f"CommonTeamRoster for team {team_id} ({season}) is missing "
            f"{sorted(missing)}. The API schema may have changed - update src/columns.py."
        )
    return df


def _pull_one_roster_live(team_id: int, season: str) -> pd.DataFrame:
    """Pull one team's roster for a season with retry-and-backoff (V2)."""
    last_error: Exception | None = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            frames = commonteamroster.CommonTeamRoster(
                team_id=team_id, season=season,
                timeout=config.REQUEST_TIMEOUT_SECONDS,
            ).get_data_frames()
            return _validate_roster_schema(frames[0], team_id=team_id, season=season)
        except Exception as error:  # noqa: BLE001 - network/JSON errors are broad and retryable
            last_error = error
            backoff = config.BACKOFF_BASE_SECONDS ** attempt
            print(f"  ! roster pull team {team_id} {season} failed "
                  f"(attempt {attempt}/{config.MAX_RETRIES}): {error!r}; "
                  f"retrying in {backoff:.1f}s")
            time.sleep(backoff)
    raise RuntimeError(f"Roster pull for team {team_id} ({season}) failed after "
                       f"{config.MAX_RETRIES} attempts") from last_error


def load_team_rosters(season: str) -> pd.DataFrame:
    """Load every team's roster for one season (the V2 coarse-position source).

    Makes one ``CommonTeamRoster`` call per team (30), rate-limited and retried,
    then caches the concatenated result to ``data/rosters_{season}.parquet`` so the
    30 calls are made at most once. Used only to attach a single coarse position
    per player for value-over-replacement; full Yahoo multi-position eligibility is
    V5.

    Args:
        season: season string, e.g. ``"2022-23"``. To stay leakage-free, callers
            pass the season *prior* to the draft target.

    Returns:
        DataFrame with at least ``PLAYER_ID`` and ``POSITION`` (one row per player).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = _roster_cache_path(season)
    if cache.exists():
        print(f"  - rosters {season}: reading cache {cache.name}")
        return _validate_roster_schema(pd.read_parquet(cache), team_id=-1, season=season)

    print(f"  - rosters {season}: pulling {len(static_teams.get_teams())} team "
          f"rosters live from nba_api ...")
    frames: list[pd.DataFrame] = []
    for i, team in enumerate(static_teams.get_teams()):
        if i > 0:
            time.sleep(config.SLEEP_BETWEEN_CALLS_SECONDS)  # be polite between calls
        frames.append(_pull_one_roster_live(team["id"], season))

    rosters = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=[C.COL_PLAYER_ID], keep="first")
        .reset_index(drop=True)
    )
    rosters.to_parquet(cache, index=False)
    print(f"    cached {len(rosters):,} players -> {cache.name}")
    return rosters


def _validate_bio_schema(df: pd.DataFrame, *, season: str) -> pd.DataFrame:
    """Assert a bios frame has the columns V4 needs; return it unchanged."""
    missing = C.REQUIRED_BIO_COLUMNS - set(df.columns)
    if missing:
        raise KeyError(
            f"LeagueDashPlayerBioStats for {season} is missing {sorted(missing)}. "
            f"The API schema may have changed - update src/columns.py."
        )
    return df


def _pull_one_bios_live(season: str) -> pd.DataFrame:
    """Pull one season's league-wide player bios with retry-and-backoff (V4)."""
    last_error: Exception | None = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            frames = leaguedashplayerbiostats.LeagueDashPlayerBioStats(
                season=season, season_type_all_star=config.SEASON_TYPE,
                timeout=config.REQUEST_TIMEOUT_SECONDS,
            ).get_data_frames()
            return _validate_bio_schema(frames[0], season=season)
        except Exception as error:  # noqa: BLE001 - network/JSON errors are broad and retryable
            last_error = error
            backoff = config.BACKOFF_BASE_SECONDS ** attempt
            print(f"  ! bios pull {season} failed (attempt {attempt}/{config.MAX_RETRIES}): "
                  f"{error!r}; retrying in {backoff:.1f}s")
            time.sleep(backoff)
    raise RuntimeError(f"Bios pull for {season} failed after "
                       f"{config.MAX_RETRIES} attempts") from last_error


def load_player_bios(seasons: list[str]) -> pd.DataFrame:
    """Load per-season player ages (the V4 age source), one call per season, cached.

    Returns a long frame with ``SEASON_YEAR``, ``PLAYER_ID`` and ``AGE`` - one row
    per player-season. Used to build the aging curve and to locate each player on
    it. Each season is a single league-wide call, written to ``data/`` as Parquet.

    Args:
        seasons: season strings, e.g. ``["2022-23", "2023-24"]``.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    per_season: list[pd.DataFrame] = []
    made_live_call = False
    for season in seasons:
        cache = _bios_cache_path(season)
        if cache.exists():
            print(f"  - bios {season}: reading cache {cache.name}")
            df = _validate_bio_schema(pd.read_parquet(cache), season=season)
        else:
            if made_live_call:
                time.sleep(config.SLEEP_BETWEEN_CALLS_SECONDS)
            print(f"  - bios {season}: pulling live from nba_api ...")
            df = _pull_one_bios_live(season)
            df = df.assign(**{C.COL_SEASON: season})  # endpoint omits the season label
            df.to_parquet(cache, index=False)
            made_live_call = True
            print(f"    cached {len(df):,} player bios -> {cache.name}")
        # Cached frames already carry SEASON_YEAR; ensure live and cached agree.
        if C.COL_SEASON not in df.columns:
            df = df.assign(**{C.COL_SEASON: season})
        per_season.append(df[[C.COL_SEASON, C.COL_PLAYER_ID, C.COL_AGE]])
    return pd.concat(per_season, ignore_index=True)
