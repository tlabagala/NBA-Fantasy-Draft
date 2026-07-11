"""V5 Yahoo pure-helper tests (no network, no credentials).

The live yfpy calls cannot be tested without credentials, but the deterministic
mapping helpers can and should be.
"""

from __future__ import annotations

from src import columns as C
from src import positions as pos_mod
from src import yahoo


def test_yahoo_positions_to_coarse() -> None:
    assert yahoo.yahoo_positions_to_coarse(["PG", "SG", "G", "UTIL"]) == "G"
    assert yahoo.yahoo_positions_to_coarse(["SF", "PF", "F"]) == "F"
    assert yahoo.yahoo_positions_to_coarse(["C"]) == "C"
    assert yahoo.yahoo_positions_to_coarse(["PF", "C"]) == "C"   # centre eligibility wins
    assert yahoo.yahoo_positions_to_coarse(["UTIL"]) == pos_mod.UNKNOWN_POSITION


def test_yahoo_stat_abbreviations_map_to_real_columns() -> None:
    # Every mapped target must be a genuine scorable nba_api column.
    for column in yahoo.YAHOO_ABBR_TO_NBA.values():
        assert column in C.SCORABLE_STAT_COLUMNS
    # A representative points-league set is covered.
    for abbr in ("PTS", "REB", "AST", "ST", "BLK", "TO", "3PTM"):
        assert abbr in yahoo.YAHOO_ABBR_TO_NBA


def test_make_query_raises_without_credentials(monkeypatch) -> None:
    from src import config
    monkeypatch.setattr(config, "USE_YAHOO", False)
    try:
        yahoo.make_query()
        assert False, "expected RuntimeError when Yahoo is not configured"
    except RuntimeError as error:
        assert ".env" in str(error)
