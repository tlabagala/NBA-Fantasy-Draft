"""V5 reconciliation tests: name normalisation and the crosswalk join."""

from __future__ import annotations

import pandas as pd

from src import columns as C
from src import reconcile


def test_normalise_name_strips_accents_punctuation_suffixes() -> None:
    assert reconcile.normalise_name("Nikola Jokić") == "nikola jokic"
    assert reconcile.normalise_name("Jaren Jackson Jr.") == "jaren jackson"
    assert reconcile.normalise_name("De'Aaron Fox") == "de aaron fox"
    assert reconcile.normalise_name("Luka Dončić") == "luka doncic"


def _nba(rows: list[tuple[int, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[C.COL_PLAYER_ID, C.COL_PLAYER_NAME])


def _yahoo(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["yahoo_id", "yahoo_name"])


def test_crosswalk_matches_on_normalised_names() -> None:
    nba = _nba([(1, "Nikola Jokić"), (2, "Jaren Jackson Jr."), (3, "Only In NBA")])
    yahoo = _yahoo([("p1", "Nikola Jokic"), ("p2", "Jaren Jackson"), ("p9", "Only In Yahoo")])
    cw = reconcile.build_crosswalk(nba, yahoo)
    matched = dict(zip(cw.matches[C.COL_PLAYER_ID], cw.matches["yahoo_id"]))
    assert matched == {1: "p1", 2: "p2"}
    # Unmatched on each side are surfaced, never silently dropped.
    assert set(cw.unmatched_nba[C.COL_PLAYER_ID]) == {3}
    assert set(cw.unmatched_yahoo["yahoo_id"]) == {"p9"}


def test_crosswalk_does_not_misjoin_ambiguous_names() -> None:
    # Two different players share a normalised name -> left for manual review.
    nba = _nba([(1, "Chris Johnson"), (2, "Chris Johnson")])
    yahoo = _yahoo([("p1", "Chris Johnson")])
    cw = reconcile.build_crosswalk(nba, yahoo)
    assert cw.matches.empty
    assert set(cw.unmatched_nba[C.COL_PLAYER_ID]) == {1, 2}
    assert set(cw.unmatched_yahoo["yahoo_id"]) == {"p1"}


def test_public_crosswalk_takes_precedence() -> None:
    nba = _nba([(10, "Some Name")])
    yahoo = _yahoo([("y10", "Totally Different Spelling")])
    public = pd.DataFrame({C.COL_PLAYER_ID: [10], "yahoo_id": ["y10"]})
    cw = reconcile.build_crosswalk(nba, yahoo, public_crosswalk=public)
    assert dict(zip(cw.matches[C.COL_PLAYER_ID], cw.matches["yahoo_id"])) == {10: "y10"}
    assert cw.unmatched_nba.empty and cw.unmatched_yahoo.empty
