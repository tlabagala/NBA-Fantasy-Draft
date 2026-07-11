"""Reconcile nba_api players with Yahoo players (V5).

The two sources use different identity systems - ``nba_api`` uses NBA.com integer
ids, Yahoo uses its own player keys - so any join between them must be bridged
here, and exactly here (master spec Section 6).

Strategy, in order of preference:
1. An existing public id crosswalk, if one is supplied (a DataFrame mapping NBA.com
   ids to Yahoo ids). This is the most reliable and is preferred when available.
2. Otherwise, match on **normalised names**: lower-cased, accent-stripped,
   punctuation- and suffix-removed. Ambiguous or unmatched players on EITHER side
   are surfaced for manual review - never silently dropped or mis-joined.

This module is pure data wrangling with no network calls, so it is fully unit
tested even though the live Yahoo half (``yahoo.py``) cannot be run without
credentials.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import pandas as pd

from src import columns as C

# Generational suffixes to strip so "Jaren Jackson Jr." matches "Jaren Jackson".
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalise_name(name: str) -> str:
    """Return a canonical match key for a player name.

    Lower-cases, strips accents (Jokic == Jokic), removes punctuation and
    generational suffixes, and collapses whitespace. Designed so the same human
    matches across sources despite cosmetic spelling differences.
    """
    decomposed = unicodedata.normalize("NFKD", str(name))
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii").lower()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", ascii_only)  # drop ' . , - etc.
    tokens = [t for t in cleaned.split() if t not in _SUFFIXES]
    return " ".join(tokens)


@dataclass
class Crosswalk:
    """Result of a reconciliation: the matches plus the unmatched stragglers."""

    matches: pd.DataFrame                       # nba_id, yahoo_id, nba_name, yahoo_name
    unmatched_nba: pd.DataFrame = field(default_factory=pd.DataFrame)
    unmatched_yahoo: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        return (f"{len(self.matches)} matched, "
                f"{len(self.unmatched_nba)} nba-only, "
                f"{len(self.unmatched_yahoo)} yahoo-only (review these)")


def build_crosswalk(
    nba_players: pd.DataFrame,
    yahoo_players: pd.DataFrame,
    public_crosswalk: pd.DataFrame | None = None,
) -> Crosswalk:
    """Join nba_api players to Yahoo players, surfacing anything unmatched.

    Args:
        nba_players: columns ``PLAYER_ID`` and ``PLAYER_NAME`` (NBA.com side).
        yahoo_players: columns ``yahoo_id`` and ``yahoo_name`` (Yahoo side).
        public_crosswalk: optional preferred id map with columns ``PLAYER_ID`` and
            ``yahoo_id``; when supplied it is used directly and names are only a
            fallback for whatever it does not cover.

    Returns:
        A :class:`Crosswalk`. Matches are de-duplicated; any name that is ambiguous
        (maps to more than one player on the other side) is treated as unmatched and
        surfaced rather than guessed.
    """
    nba = nba_players[[C.COL_PLAYER_ID, C.COL_PLAYER_NAME]].drop_duplicates().copy()
    yahoo = yahoo_players[["yahoo_id", "yahoo_name"]].drop_duplicates().copy()

    matched_rows: list[dict] = []
    matched_nba_ids: set = set()
    matched_yahoo_ids: set = set()

    # 1. Use the public crosswalk first where it covers a player.
    if public_crosswalk is not None and not public_crosswalk.empty:
        names_nba = dict(zip(nba[C.COL_PLAYER_ID], nba[C.COL_PLAYER_NAME]))
        names_yh = dict(zip(yahoo["yahoo_id"], yahoo["yahoo_name"]))
        for _, row in public_crosswalk.iterrows():
            nid, yid = row[C.COL_PLAYER_ID], row["yahoo_id"]
            if nid in names_nba and yid in names_yh:
                matched_rows.append({
                    C.COL_PLAYER_ID: nid, "yahoo_id": yid,
                    C.COL_PLAYER_NAME: names_nba[nid], "yahoo_name": names_yh[yid],
                })
                matched_nba_ids.add(nid)
                matched_yahoo_ids.add(yid)

    # 2. Name match on whatever remains.
    nba_rem = nba.loc[~nba[C.COL_PLAYER_ID].isin(matched_nba_ids)].assign(
        _key=lambda df: df[C.COL_PLAYER_NAME].map(normalise_name))
    yahoo_rem = yahoo.loc[~yahoo["yahoo_id"].isin(matched_yahoo_ids)].assign(
        _key=lambda df: df["yahoo_name"].map(normalise_name))

    # Keys that are unique on BOTH sides are safe to join; everything else is left
    # for manual review (do not mis-join on an ambiguous name).
    nba_counts = nba_rem["_key"].value_counts()
    yahoo_counts = yahoo_rem["_key"].value_counts()
    unique_keys = set(nba_counts[nba_counts == 1].index) & set(yahoo_counts[yahoo_counts == 1].index)

    joined = nba_rem.loc[nba_rem["_key"].isin(unique_keys)].merge(
        yahoo_rem.loc[yahoo_rem["_key"].isin(unique_keys)], on="_key", how="inner")
    for _, row in joined.iterrows():
        matched_rows.append({
            C.COL_PLAYER_ID: row[C.COL_PLAYER_ID], "yahoo_id": row["yahoo_id"],
            C.COL_PLAYER_NAME: row[C.COL_PLAYER_NAME], "yahoo_name": row["yahoo_name"],
        })
        matched_nba_ids.add(row[C.COL_PLAYER_ID])
        matched_yahoo_ids.add(row["yahoo_id"])

    matches = pd.DataFrame(matched_rows, columns=[
        C.COL_PLAYER_ID, "yahoo_id", C.COL_PLAYER_NAME, "yahoo_name"])
    return Crosswalk(
        matches=matches,
        unmatched_nba=nba.loc[~nba[C.COL_PLAYER_ID].isin(matched_nba_ids)].reset_index(drop=True),
        unmatched_yahoo=yahoo.loc[~yahoo["yahoo_id"].isin(matched_yahoo_ids)].reset_index(drop=True),
    )
