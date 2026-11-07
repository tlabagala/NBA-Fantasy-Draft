"""V1 projection: prior-season total as the projection for the target season.

This is the naive, defensible baseline ("next season = last season"). Everything
later (recency weighting, points-per-game x games-played, aging, shrinkage) is V2+.

The single most important property here is the LEAKAGE GUARD: a projection for
season N may use only data from seasons strictly before N. This is enforced with
an explicit assertion, not just by convention (master spec Sections 3 and 7).
"""

from __future__ import annotations

import pandas as pd

from src import columns as C
from src import config
from src import positions as pos_mod


def next_season(season: str) -> str:
    """Return the season immediately after ``season`` (inverse of :func:`prior_season`)."""
    start_year = int(season.split("-")[0])
    nxt = start_year + 1
    return f"{nxt}-{str(nxt + 1)[-2:]}"


def prior_season(season: str) -> str:
    """Return the season immediately before ``season``.

    Seasons are NBA-style strings, e.g. ``"2023-24"`` -> ``"2022-23"``. The end
    year is derived from the start year so it stays correct across century-style
    rollovers within reason.
    """
    start_year = int(season.split("-")[0])
    prev_start = start_year - 1
    return f"{prev_start}-{str(prev_start + 1)[-2:]}"


def project_v1(totals: pd.DataFrame, target_season: str) -> pd.DataFrame:
    """Project each player's ``target_season`` total as their prior-season total.

    Args:
        totals: season totals (one row per player-season) from
            :func:`src.scoring.season_totals`.
        target_season: the season being projected/drafted for, e.g. ``"2023-24"``.

    Returns:
        One row per player with a projected total in ``PROJECTED_PTS``, drawn from
        the immediately prior season. Players with no prior-season history are
        absent (V1 has no rookie signal - a documented limitation).

    Raises:
        AssertionError: if any input row used to build the projection belongs to
            ``target_season`` or any later season (leakage).
    """
    prev = prior_season(target_season)

    # The rows the projection actually consumes: the immediately prior season.
    used = totals.loc[totals[C.COL_SEASON] == prev]

    # --- Leakage guard (explicit, not by convention) ---
    # Assert on the rows we are about to USE, so the check is protective rather
    # than tautological: if the selection above were ever loosened (e.g. to `>=`
    # or to `<= target`) so that target-season-or-later rows slipped in, this trips.
    leaked = used.loc[used[C.COL_SEASON] >= target_season]
    assert leaked.empty, (
        f"Leakage: {len(leaked)} rows from {target_season} or later entered the "
        f"projection input for {target_season}."
    )

    return (
        used
        .loc[:, [C.COL_PLAYER_ID, C.COL_PLAYER_NAME, C.COL_TOTAL_FANTASY_PTS]]
        .rename(columns={C.COL_TOTAL_FANTASY_PTS: C.COL_PROJECTED_PTS})
        .reset_index(drop=True)
    )


def recent_prior_seasons(totals: pd.DataFrame, target_season: str,
                         n_seasons: int) -> list[str]:
    """Return up to ``n_seasons`` distinct seasons strictly before the target,
    most recent first. This is the V2 recency window."""
    priors = sorted(
        {s for s in totals[C.COL_SEASON].unique() if s < target_season},
        reverse=True,
    )
    return priors[:n_seasons]


def project_v2(
    totals: pd.DataFrame,
    target_season: str,
    n_seasons: int = config.N_PROJECTION_SEASONS,
    gp_decay: float = config.GP_RECENCY_DECAY,
) -> pd.DataFrame:
    """Project ``target_season`` as projected points-per-game x projected games-played.

    Improves on V1 in two ways (master spec V2):
    - **Recency-weighted across 2-3 seasons.** Points-per-game is games-played-
      weighted (total points / total games over the window), so a low-minutes
      season counts for less. The window is the ``n_seasons`` most recent prior
      seasons only.
    - **Durability modelled directly.** The total is PPG x a separate projected
      games-played estimate (a recency-weighted mean of past games played), rather
      than a single stale season total. This stops a player who missed time last
      season from being permanently buried, and vice versa.

    The same hard leakage guard as V1 applies: only seasons strictly before the
    target may enter.

    Args:
        totals: season totals (one row per player-season).
        target_season: season being projected, e.g. ``"2023-24"``.
        n_seasons: size of the recency window (most recent prior seasons).
        gp_decay: per-season decay for the projected games-played estimate.

    Returns:
        One row per player with ``PROJECTED_PPG``, ``PROJECTED_GP`` and
        ``PROJECTED_PTS`` (= PPG x GP). Players with no games in the window are
        excluded.

    Raises:
        AssertionError: if any consumed row is from ``target_season`` or later.
    """
    window = recent_prior_seasons(totals, target_season, n_seasons)

    used = totals.loc[totals[C.COL_SEASON].isin(window)].copy()

    # --- Leakage guard (explicit, protective) ---
    leaked = used.loc[used[C.COL_SEASON] >= target_season]
    assert leaked.empty, (
        f"Leakage: {len(leaked)} rows from {target_season} or later entered the "
        f"V2 projection input for {target_season}."
    )

    # Build a complete (player x window-season) grid so a season a player MISSED
    # counts as 0 games - that absence is the durability signal. But only from each
    # player's first appearance onward, so a player who entered the league mid-window
    # (e.g. a sophomore) is not penalised for seasons before he existed.
    players = used[[C.COL_PLAYER_ID, C.COL_PLAYER_NAME]].drop_duplicates()
    first_season = (
        used.groupby(C.COL_PLAYER_ID, as_index=False)[C.COL_SEASON]
        .min().rename(columns={C.COL_SEASON: "_first_season"})
    )
    grid = (
        players
        .merge(first_season, on=C.COL_PLAYER_ID)
        .merge(pd.DataFrame({C.COL_SEASON: window}), how="cross")
        .loc[lambda df: df[C.COL_SEASON] >= df["_first_season"]]
    )
    filled = (
        grid
        .merge(
            used[[C.COL_PLAYER_ID, C.COL_SEASON,
                  C.COL_TOTAL_FANTASY_PTS, C.COL_GAMES_PLAYED]],
            on=[C.COL_PLAYER_ID, C.COL_SEASON], how="left",
        )
        .fillna({C.COL_TOTAL_FANTASY_PTS: 0.0, C.COL_GAMES_PLAYED: 0})
    )

    # Recency weight: most recent prior season has rank 0 -> weight 1; older seasons
    # decay by gp_decay per year. The points-per-game rate is games-played-weighted
    # (total points / total games) so it reflects scoring when on court; durability
    # is carried entirely by the games-played estimate (which includes the zeros).
    season_rank = {season: rank for rank, season in enumerate(window)}
    filled = filled.assign(
        _w=lambda df: df[C.COL_SEASON].map(season_rank).rpow(gp_decay),
        _wgp=lambda df: df["_w"] * df[C.COL_GAMES_PLAYED],
    )

    return (
        filled
        .groupby([C.COL_PLAYER_ID, C.COL_PLAYER_NAME], as_index=False)
        .agg(
            _pts_total=(C.COL_TOTAL_FANTASY_PTS, "sum"),
            _gp_total=(C.COL_GAMES_PLAYED, "sum"),
            _wsum=("_w", "sum"),
            _wgp_sum=("_wgp", "sum"),
        )
        .loc[lambda df: df["_gp_total"] > 0]
        .assign(**{
            C.COL_PROJECTED_PPG: lambda df: df["_pts_total"] / df["_gp_total"],
            C.COL_PROJECTED_GP: lambda df: df["_wgp_sum"] / df["_wsum"],
            C.COL_PROJECTED_PTS: lambda df: (
                (df["_pts_total"] / df["_gp_total"]) * (df["_wgp_sum"] / df["_wsum"])
            ),
        })
        .loc[:, [C.COL_PLAYER_ID, C.COL_PLAYER_NAME,
                 C.COL_PROJECTED_PPG, C.COL_PROJECTED_GP, C.COL_PROJECTED_PTS]]
        .sort_values(C.COL_PROJECTED_PTS, ascending=False, ignore_index=True)
    )


# ===========================================================================
# V4: aging curve + empirical-Bayes shrinkage on top of the V2 PPG x GP split.
# ===========================================================================

def _ppg_by_player_season(totals: pd.DataFrame) -> pd.DataFrame:
    """Per-game fantasy rate for each player-season (total points / games)."""
    return totals.assign(**{
        C.COL_PPG: lambda df: df[C.COL_TOTAL_FANTASY_PTS] / df[C.COL_GAMES_PLAYED]
    })


def build_aging_curve(
    totals: pd.DataFrame,
    bios: pd.DataFrame,
    target_season: str,
    league,
    min_games: int = 20,
) -> dict[int, float]:
    """Estimate an age -> year-over-year PPG multiplier curve from prior seasons.

    For every consecutive-season pair that a player completed (both seasons before
    the target, both with at least ``min_games`` games), we take the ratio of the
    second season's per-game rate to the first, and summarise it by the player's age
    in the first season. The result is the expected multiplicative change in rate
    from age A to A+1 - rising into the mid-20s, then declining.

    KNOWN FLAW - SURVIVORSHIP BIAS: this uses only players who appeared in BOTH
    seasons. Players who age out and leave the league are excluded, so late-career
    decline is understated (the curve looks flatter at older ages than reality). We
    document this rather than hide it; a fuller fix (treating league exits as a
    decline signal) needs careful handling of injury-vs-decline and is left for
    later. The leakage guard still holds: only seasons before the target are used.

    Returns:
        ``{age: multiplier}`` for ages in ``[AGE_CURVE_MIN, AGE_CURVE_MAX]``;
        ages with no data fall back to 1.0 (no adjustment).
    """
    ppg = _ppg_by_player_season(totals).loc[lambda df: df[C.COL_SEASON] < target_season]

    # --- Leakage guard ---
    assert (ppg[C.COL_SEASON] < target_season).all(), (
        f"Leakage: aging curve for {target_season} drew on non-prior seasons.")

    age_lookup = {(int(r[C.COL_PLAYER_ID]), r[C.COL_SEASON]): r[C.COL_AGE]
                  for _, r in bios.loc[bios[C.COL_SEASON] < target_season].iterrows()}

    pairs = (
        ppg.assign(_next=lambda df: df[C.COL_SEASON].map(next_season))
        .merge(
            ppg[[C.COL_PLAYER_ID, C.COL_SEASON, C.COL_PPG, C.COL_GAMES_PLAYED]],
            left_on=[C.COL_PLAYER_ID, "_next"], right_on=[C.COL_PLAYER_ID, C.COL_SEASON],
            suffixes=("_1", "_2"),
        )
        .loc[lambda df: (df[f"{C.COL_GAMES_PLAYED}_1"] >= min_games)
             & (df[f"{C.COL_GAMES_PLAYED}_2"] >= min_games)
             & (df[f"{C.COL_PPG}_1"] > 0)]
    )
    pairs = pairs.assign(
        _age=lambda df: [age_lookup.get((int(pid), s))
                         for pid, s in zip(df[C.COL_PLAYER_ID], df[f"{C.COL_SEASON}_1"])],
        _ratio=lambda df: df[f"{C.COL_PPG}_2"] / df[f"{C.COL_PPG}_1"],
    ).dropna(subset=["_age"])

    by_age = (
        pairs.assign(_age=lambda df: df["_age"].round().astype(int))
        .groupby("_age")["_ratio"].median()
    )

    # Smooth and clamp onto the configured age band; fall back to 1.0 where empty.
    ages = range(league.AGE_CURVE_MIN, league.AGE_CURVE_MAX + 1)
    smooth = league.AGE_CURVE_SMOOTH
    curve: dict[int, float] = {}
    for age in ages:
        window_vals = [by_age[a] for a in range(age - smooth, age + smooth + 1) if a in by_age.index]
        curve[age] = float(sum(window_vals) / len(window_vals)) if window_vals else 1.0
    return curve


def _clamp_age(age: int, league) -> int:
    return max(league.AGE_CURVE_MIN, min(league.AGE_CURVE_MAX, age))


def _aging_factor(age_last: int, years_forward: int, curve: dict[int, float], league) -> float:
    """Chain the per-year multipliers from ``age_last`` forward ``years_forward`` years."""
    factor = 1.0
    for age in range(age_last, age_last + max(years_forward, 0)):
        factor *= curve.get(_clamp_age(age, league), 1.0)
    return factor


def project_v4(
    totals: pd.DataFrame,
    target_season: str,
    bios: pd.DataFrame,
    position_map: dict[int, str],
    league,
    use_aging: bool = True,
    use_shrinkage: bool = True,
) -> pd.DataFrame:
    """V4 projection: V2's PPG x GP, with shrinkage and an aging adjustment on PPG.

    Pipeline for each player's per-game rate (then multiplied by V2's projected
    games-played):

    1. **Empirical-Bayes shrinkage** towards a role (position) prior, weighted by
       games played: ``(n*ppg + k*prior) / (n + k)``. Extreme small-sample seasons
       regress to the positional mean and are trusted less the fewer games they had.
    2. **Aging adjustment**: multiply by the chained aging-curve factor from the
       player's most recent prior-season age forward to the target season.

    Both ``use_aging`` and ``use_shrinkage`` can be toggled so each component's
    contribution can be measured separately (V4 keeps only what helps). The leakage
    guard from :func:`project_v2` covers the production inputs; the aging curve and
    role prior are built from prior seasons only.

    Returns:
        One row per player with ``PROJECTED_PPG`` (adjusted), ``PROJECTED_GP`` and
        ``PROJECTED_PTS``. Carries ``n_unknown_age`` in ``.attrs``.
    """
    # Take the recency window from ``league``, not from the literal config module.
    # Line below used to call project_v2(totals, target_season) with no override, so
    # the base projection silently ignored the league object while the shrinkage
    # window (next line) respected it - the same league could be described by two
    # different windows in one call. It also meant a caller passing a custom league
    # (src/app_ui.py's setup screen, every test double) could not actually change the
    # projection window.
    base = project_v2(totals, target_season,
                      n_seasons=league.N_PROJECTION_SEASONS,
                      gp_decay=league.GP_RECENCY_DECAY)

    window = recent_prior_seasons(totals, target_season, league.N_PROJECTION_SEASONS)
    window_totals = totals.loc[totals[C.COL_SEASON].isin(window)]
    n_games = window_totals.groupby(C.COL_PLAYER_ID)[C.COL_GAMES_PLAYED].sum()

    # Role (position) prior: games-weighted mean PPG per coarse position over the
    # window, with an overall prior for unknown-position players.
    pos_series = window_totals[C.COL_PLAYER_ID].map(position_map).fillna(pos_mod.UNKNOWN_POSITION)
    pos_agg = (
        window_totals.assign(_pos=pos_series.to_numpy())
        .groupby("_pos")
        .agg(_t=(C.COL_TOTAL_FANTASY_PTS, "sum"), _g=(C.COL_GAMES_PLAYED, "sum"))
    )
    pos_prior = (pos_agg["_t"] / pos_agg["_g"]).to_dict()
    overall_prior = (window_totals[C.COL_TOTAL_FANTASY_PTS].sum()
                     / window_totals[C.COL_GAMES_PLAYED].sum())

    # Age in each player's most recent prior season, and that season's year.
    prior_bios = bios.loc[bios[C.COL_SEASON] < target_season]
    last_age: dict[int, int] = {}
    last_year: dict[int, int] = {}
    for pid, sub in prior_bios.groupby(C.COL_PLAYER_ID):
        recent = sub.loc[sub[C.COL_SEASON].idxmax()]
        last_age[int(pid)] = int(round(recent[C.COL_AGE]))
        last_year[int(pid)] = int(recent[C.COL_SEASON].split("-")[0])

    curve = build_aging_curve(totals, bios, target_season, league) if use_aging else {}
    target_year = int(target_season.split("-")[0])
    k = league.SHRINKAGE_K

    unknown_age = 0
    ppg_out: list[float] = []
    for _, row in base.iterrows():
        pid = int(row[C.COL_PLAYER_ID])
        ppg = row[C.COL_PROJECTED_PPG]
        if use_shrinkage:
            position = position_map.get(pid, pos_mod.UNKNOWN_POSITION)
            prior = pos_prior.get(position, overall_prior)
            n = float(n_games.get(pid, 0.0))
            ppg = (n * ppg + k * prior) / (n + k) if (n + k) > 0 else ppg
        if use_aging:
            if pid in last_age:
                years_forward = target_year - last_year[pid]
                ppg *= _aging_factor(last_age[pid], years_forward, curve, league)
            else:
                unknown_age += 1
        ppg_out.append(ppg)

    out = base.assign(**{
        C.COL_PROJECTED_PPG: ppg_out,
        C.COL_PROJECTED_PTS: [p * gp for p, gp in zip(ppg_out, base[C.COL_PROJECTED_GP])],
    }).sort_values(C.COL_PROJECTED_PTS, ascending=False, ignore_index=True)
    out.attrs["n_unknown_age"] = unknown_age
    return out
