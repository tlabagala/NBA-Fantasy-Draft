"""Live Yahoo Fantasy integration via yfpy (V5).

Pulls the real scoring formula, positional eligibility, and draft results from your
Yahoo league, replacing the placeholder ``config.SCORING``.

IMPORTANT - CREDENTIALS AND OAUTH (master spec Section 5):
- Credentials live only in a git-ignored ``.env`` (see ``.env.example``). They are
  never hardcoded or committed.
- The one-time browser OAuth authorisation must be performed BY YOU. The first call
  here opens a browser; yfpy then caches a token. This code sets everything up but
  does not (and cannot) complete the authorisation on your behalf.

This module could not be executed during the build (no credentials), so treat it as
carefully-structured scaffolding: verify the field names against your league on the
first authorised run. yfpy is imported lazily so the rest of the project works
without it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import columns as C
from src import config

_ENV_PATH = Path(__file__).resolve().parent.parent

# Yahoo stat abbreviation -> nba_api column. Counting stats only (a points league
# scores these). Percentages and derived double/triple-doubles have no raw nba_api
# column and are surfaced as unmapped rather than guessed.
YAHOO_ABBR_TO_NBA: dict[str, str] = {
    "PTS": C.COL_PTS,
    "REB": C.COL_REB, "TREB": C.COL_REB,
    "AST": C.COL_AST,
    "ST": C.COL_STL, "STL": C.COL_STL,
    "BLK": C.COL_BLK,
    "TO": C.COL_TOV, "TOV": C.COL_TOV,
    "FGM": C.COL_FGM, "FGA": C.COL_FGA,
    "FTM": C.COL_FTM, "FTA": C.COL_FTA,
    "3PTM": C.COL_FG3M, "TPM": C.COL_FG3M, "3PM": C.COL_FG3M,
    "3PTA": C.COL_FG3A, "TPA": C.COL_FG3A, "3PA": C.COL_FG3A,
    "OREB": C.COL_OREB, "DREB": C.COL_DREB,
    "PF": C.COL_PF,
}


def make_query():
    """Construct a yfpy query from the .env credentials.

    Raises:
        RuntimeError: if Yahoo is not configured (``config.USE_YAHOO`` is False).
    """
    if not config.USE_YAHOO:
        raise RuntimeError(
            "Yahoo is not configured. Copy .env.example to .env and fill in your "
            "Yahoo consumer key/secret and league id, then run again to authorise."
        )
    from yfpy.query import YahooFantasySportsQuery  # lazy: only needed when live

    return YahooFantasySportsQuery(
        league_id=config.YAHOO_LEAGUE_ID,
        game_code=config.YAHOO_GAME_CODE,
        yahoo_consumer_key=config.YAHOO_CONSUMER_KEY,
        yahoo_consumer_secret=config.YAHOO_CONSUMER_SECRET,
        env_file_location=_ENV_PATH,
        save_token_data_to_env_file=True,
        browser_callback=True,  # opens the browser for YOUR one-time authorisation
    )


def get_live_scoring(query=None) -> tuple[dict[str, float], list[str]]:
    """Return the league's scoring as an nba_api ``{column: weight}`` map.

    Returns:
        ``(scoring, unmapped)`` - the scoring dict ready to drop into the pipeline,
        and a list of Yahoo stat names that could not be mapped to a raw nba_api
        column (e.g. double-doubles, shooting percentages) for you to review.
    """
    query = query or make_query()
    settings = query.get_league_settings()

    # stat_categories names each stat (id, abbreviation); stat_modifiers gives the
    # point value per stat id. Join them.
    abbr_by_id: dict[int, str] = {}
    for stat in getattr(settings.stat_categories, "stats", []):
        abbr_by_id[int(stat.stat_id)] = (getattr(stat, "abbr", None)
                                         or getattr(stat, "display_name", "")
                                         or getattr(stat, "name", ""))

    scoring: dict[str, float] = {}
    unmapped: list[str] = []
    for mod in getattr(settings.stat_modifiers, "stats", []):
        stat_id = int(mod.stat_id)
        value = float(getattr(mod, "value", 0.0) or 0.0)
        abbr = abbr_by_id.get(stat_id, str(stat_id)).upper().strip()
        column = YAHOO_ABBR_TO_NBA.get(abbr)
        if column is None:
            unmapped.append(abbr)
        elif value != 0.0:
            scoring[column] = scoring.get(column, 0.0) + value
    return scoring, unmapped


def get_positional_eligibility(query=None) -> pd.DataFrame:
    """Return every league player's Yahoo id, name, and eligible positions.

    Returns:
        DataFrame with columns ``yahoo_id``, ``yahoo_name`` and
        ``eligible_positions`` (a list like ``["PG", "SG", "G", "UTIL"]``).
    """
    query = query or make_query()
    players = query.get_league_players()  # paginated under the hood by yfpy
    rows = []
    for player in players:
        positions = getattr(player, "eligible_positions", []) or []
        # yfpy may wrap each position in an object; normalise to plain strings.
        positions = [getattr(p, "position", p) for p in positions]
        rows.append({
            "yahoo_id": getattr(player, "player_id", None),
            "yahoo_name": str(getattr(player, "full_name", "")
                              or getattr(getattr(player, "name", None), "full", "")),
            "eligible_positions": positions,
        })
    return pd.DataFrame(rows)


def get_draft_results(query=None) -> pd.DataFrame:
    """Return the league draft results as an ADP table (overall pick order).

    Returns:
        DataFrame with ``pick`` (overall), ``round``, and ``yahoo_id``; lower pick
        number means drafted earlier (a real ADP signal for the ADP benchmark).
    """
    query = query or make_query()
    results = query.get_league_draft_results()
    rows = []
    for r in results:
        rows.append({
            "pick": int(getattr(r, "pick", 0)),
            "round": int(getattr(r, "round", 0)),
            "yahoo_id": _player_id_from_key(getattr(r, "player_key", "")),
        })
    return pd.DataFrame(rows).sort_values("pick", ignore_index=True)


def yahoo_positions_to_coarse(eligible_positions: list[str]) -> str:
    """Collapse Yahoo eligible positions to the coarse ``G/F/C`` the engine uses.

    Picks the most specific frontcourt/backcourt signal: any centre eligibility ->
    ``C``, else any forward -> ``F``, else any guard -> ``G``. (Full multi-position
    eligibility is a richer model; this keeps the V2/V3 engine working unchanged.)
    """
    positions = {str(p).upper() for p in eligible_positions}
    if "C" in positions:
        return "C"
    if positions & {"SF", "PF", "F"}:
        return "F"
    if positions & {"PG", "SG", "G"}:
        return "G"
    from src import positions as pos_mod
    return pos_mod.UNKNOWN_POSITION


def _player_id_from_key(player_key: str) -> str:
    """Yahoo player keys look like '454.p.5583'; the id is the trailing segment."""
    return str(player_key).split(".")[-1] if player_key else ""
