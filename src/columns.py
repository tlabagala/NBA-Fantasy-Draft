"""Named constants for the ACTUAL ``nba_api`` column names.

Per the master spec (Section 6), we never hardcode *assumed* column names. These
constants were taken from a real ``PlayerGameLogs`` pull (season 2022-23, regular
season) and verified at runtime before being committed. The probe returned a
70-column frame; only the fields V1 actually uses are named here.

If a future ``nba_api`` release renames a column, this is the single place to fix
it, and :func:`src.ingest.validate_schema` will fail loudly rather than silently
producing wrong numbers.
"""

from __future__ import annotations

# --- Identity / time-ordering fields ---
COL_SEASON: str = "SEASON_YEAR"            # e.g. "2022-23"; defines the season a row belongs to
COL_PLAYER_ID: str = "PLAYER_ID"           # NBA.com integer player id (unique key for a player)
COL_PLAYER_NAME: str = "PLAYER_NAME"       # display name (for human-readable boards)
COL_TEAM_ABBR: str = "TEAM_ABBREVIATION"   # team the player suited up for in that game
COL_GAME_ID: str = "GAME_ID"               # one row per (player, game); unique within a player-season
COL_GAME_DATE: str = "GAME_DATE"           # game date string

# --- Box-score stat fields that a points league can score ---
# Verified to exist in the live pull. SCORING keys in config.py must be a subset
# of these (enforced at runtime so a typo cannot silently score nothing).
COL_MIN: str = "MIN"
COL_FGM: str = "FGM"
COL_FGA: str = "FGA"
COL_FG3M: str = "FG3M"
COL_FG3A: str = "FG3A"
COL_FTM: str = "FTM"
COL_FTA: str = "FTA"
COL_OREB: str = "OREB"
COL_DREB: str = "DREB"
COL_REB: str = "REB"
COL_AST: str = "AST"
COL_STL: str = "STL"
COL_BLK: str = "BLK"
COL_TOV: str = "TOV"
COL_PF: str = "PF"
COL_PTS: str = "PTS"

# The set of box-score columns that may legitimately appear as SCORING keys.
SCORABLE_STAT_COLUMNS: frozenset[str] = frozenset({
    COL_MIN, COL_FGM, COL_FGA, COL_FG3M, COL_FG3A, COL_FTM, COL_FTA,
    COL_OREB, COL_DREB, COL_REB, COL_AST, COL_STL, COL_BLK, COL_TOV,
    COL_PF, COL_PTS,
})

# Columns we require every raw pull to contain. ingest validates against this so a
# silent schema change cannot corrupt the pipeline.
REQUIRED_RAW_COLUMNS: frozenset[str] = frozenset({
    COL_SEASON, COL_PLAYER_ID, COL_PLAYER_NAME, COL_TEAM_ABBR,
    COL_GAME_ID, COL_GAME_DATE,
}) | SCORABLE_STAT_COLUMNS

# --- Roster endpoint (CommonTeamRoster) columns, verified at runtime (V2) ---
# Used only as the coarse-position source for value-over-replacement. Note the
# roster endpoint names the player column "PLAYER" (not "PLAYER_NAME").
COL_ROSTER_PLAYER: str = "PLAYER"           # display name on the roster endpoint
COL_POSITION: str = "POSITION"              # raw position, e.g. "G", "F-C", "G-F"

# Columns we require every roster pull to contain.
REQUIRED_ROSTER_COLUMNS: frozenset[str] = frozenset({COL_PLAYER_ID, COL_POSITION})

# --- Player bio endpoint (LeagueDashPlayerBioStats) columns, verified (V4) ---
# Used only as the age source for the aging curve and shrinkage. One league-wide
# call per season (far cheaper than the 30-team roster pull).
COL_AGE: str = "AGE"                         # player age that season (float, ~19-43)
REQUIRED_BIO_COLUMNS: frozenset[str] = frozenset({COL_PLAYER_ID, COL_AGE})

# --- Derived column names produced by this pipeline (not from the API) ---
COL_FANTASY_PTS: str = "FANTASY_PTS"        # per-game fantasy points (scoring.apply_scoring)
COL_TOTAL_FANTASY_PTS: str = "TOTAL_FANTASY_PTS"  # season total (scoring.season_totals)
COL_GAMES_PLAYED: str = "GP"                # games played in a season (row count)
COL_PROJECTED_PTS: str = "PROJECTED_PTS"    # projection output (projection.project_v1/_v2)
COL_PROJECTED_PPG: str = "PROJECTED_PPG"    # V2: projected points-per-game
COL_PROJECTED_GP: str = "PROJECTED_GP"      # V2: projected games-played (durability)
COL_RANK: str = "RANK"                      # board rank, 1 = best (board.build_board)
COL_COARSE_POSITION: str = "COARSE_POS"     # V2: G / F / C / UNK (positions.py)
COL_VOR: str = "VOR"                        # V2: value over replacement (board.build_board_vor)
COL_PPG: str = "PPG"                        # V4: per-game fantasy points for a player-season
