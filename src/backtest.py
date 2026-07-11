"""V1 backtest: greedy snake draft vs a random draft, scored on actual results.

Method (time-ordered, no leakage):
1. Project ``target_season`` using only prior-season data (projection.project_v1,
   which carries its own leakage guard).
2. Build the board and run the greedy snake draft; take your slot's roster.
3. Score every roster by each player's ACTUAL ``target_season`` total - the truth
   the projection could not see.
4. Benchmark: run many seeded random drafts, score your slot's roster in each, and
   average. Report the margin (greedy minus mean random).

Benchmark honesty (master spec Section 7): this simulates the FULL draft - every
team drafts under the same strategy - rather than replaying a real historical
draft, because no faithful replay of a past draft exists. A single random draft is
high variance, so the baseline is the mean over many seeded trials; the per-trial
spread is reported too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import board as board_mod
from src import columns as C
from src import draft as draft_mod
from src import lineup as lineup_mod
from src import positions as pos_mod
from src import projection as proj_mod


def _actual_totals_map(totals: pd.DataFrame, target_season: str) -> dict[int, float]:
    """Map player_id -> actual fantasy total for the target season (truth set)."""
    actual = totals.loc[totals[C.COL_SEASON] == target_season]
    return dict(zip(actual[C.COL_PLAYER_ID], actual[C.COL_TOTAL_FANTASY_PTS]))


def _score_roster(player_ids: list[int], actual: dict[int, float]) -> float:
    """Sum a roster's ACTUAL target-season points; a player who did not play scores 0."""
    return float(sum(actual.get(pid, 0.0) for pid in player_ids))


def _roster_detail(player_ids: list[int], board: pd.DataFrame,
                   actual: dict[int, float]) -> pd.DataFrame:
    """Human-readable roster: name, projected total, actual total (for printing)."""
    name_by_id = dict(zip(board[C.COL_PLAYER_ID], board[C.COL_PLAYER_NAME]))
    proj_by_id = dict(zip(board[C.COL_PLAYER_ID], board[C.COL_PROJECTED_PTS]))
    return pd.DataFrame({
        C.COL_PLAYER_ID: player_ids,
        C.COL_PLAYER_NAME: [name_by_id.get(pid, "<unknown>") for pid in player_ids],
        C.COL_PROJECTED_PTS: [round(proj_by_id.get(pid, float("nan")), 1) for pid in player_ids],
        "ACTUAL_PTS": [round(actual.get(pid, 0.0), 1) for pid in player_ids],
    })


def backtest_v1(totals: pd.DataFrame, target_season: str, league) -> dict:
    """Run the V1 backtest and return greedy vs random results with the margin.

    Args:
        totals: all season totals (one row per player-season).
        target_season: season to draft and score on.
        league: the :mod:`src.config` module (reads NUM_TEAMS, ROSTER_SIZE,
            MY_DRAFT_SLOT, SEED, RANDOM_BASELINE_TRIALS).

    Returns:
        dict with keys: ``greedy_total``, ``random_mean``, ``random_std``,
        ``random_min``, ``random_max``, ``margin``, ``win_rate`` (share of random
        trials the greedy roster beats), ``my_roster`` (DataFrame), ``n_trials``.
    """
    projection = proj_mod.project_v1(totals, target_season)
    board = board_mod.build_board(projection)
    actual = _actual_totals_map(totals, target_season)

    # --- Greedy draft (the model under test) ---
    greedy_rosters = draft_mod.snake_draft(
        board, league.NUM_TEAMS, league.ROSTER_SIZE, league.MY_DRAFT_SLOT,
        strategy="greedy",
    )
    my_roster_ids = greedy_rosters[league.MY_DRAFT_SLOT]
    greedy_total = _score_roster(my_roster_ids, actual)

    # --- Random baseline (mean over many seeded trials) ---
    trial_totals: list[float] = []
    for trial in range(league.RANDOM_BASELINE_TRIALS):
        random_rosters = draft_mod.snake_draft(
            board, league.NUM_TEAMS, league.ROSTER_SIZE, league.MY_DRAFT_SLOT,
            strategy="random", seed=league.SEED + trial,
        )
        trial_totals.append(_score_roster(random_rosters[league.MY_DRAFT_SLOT], actual))

    trials = pd.Series(trial_totals, dtype="float64")
    return {
        "greedy_total": round(greedy_total, 1),
        "random_mean": round(float(trials.mean()), 1),
        "random_std": round(float(trials.std(ddof=1)), 1),
        "random_min": round(float(trials.min()), 1),
        "random_max": round(float(trials.max()), 1),
        "margin": round(greedy_total - float(trials.mean()), 1),
        "win_rate": round(float((trials < greedy_total).mean()), 3),
        "my_roster": _roster_detail(my_roster_ids, board, actual),
        "n_trials": league.RANDOM_BASELINE_TRIALS,
    }


# ===========================================================================
# V2: projection accuracy (time-ordered MAE/RMSE) and the VOR-board backtest.
# ===========================================================================

def projection_accuracy(projection: pd.DataFrame, actual: dict[int, float],
                        restrict_ids: set[int] | None = None) -> dict:
    """Compute time-ordered MAE/RMSE of a projection against actual totals.

    Args:
        projection: any projection frame with ``PLAYER_ID`` and ``PROJECTED_PTS``.
        actual: player_id -> actual target-season total (the held-out truth).
        restrict_ids: if given, only score these players (used to compare two
            projections on an identical player set, which is the fair comparison).

    Returns:
        ``{"mae": ..., "rmse": ..., "n": ...}`` over players that have an actual
        total (and are in ``restrict_ids`` when supplied).
    """
    df = projection
    if restrict_ids is not None:
        df = df.loc[df[C.COL_PLAYER_ID].isin(restrict_ids)]
    df = df.loc[df[C.COL_PLAYER_ID].isin(actual)]
    error = df[C.COL_PROJECTED_PTS].to_numpy() - df[C.COL_PLAYER_ID].map(actual).to_numpy()
    return {
        "mae": round(float(np.abs(error).mean()), 1) if len(error) else float("nan"),
        "rmse": round(float(np.sqrt(np.square(error).mean())), 1) if len(error) else float("nan"),
        "n": int(len(error)),
    }


def compare_projection_accuracy(totals: pd.DataFrame, target_season: str,
                                league) -> dict:
    """Compare V1 vs V2 projection accuracy on a single target season.

    Both projections are scored on the SAME set of players (those each method can
    project AND that have an actual total) so the comparison is apples-to-apples.
    """
    v1 = proj_mod.project_v1(totals, target_season)
    v2 = proj_mod.project_v2(totals, target_season)
    actual = _actual_totals_map(totals, target_season)
    common = set(v1[C.COL_PLAYER_ID]) & set(v2[C.COL_PLAYER_ID]) & set(actual)
    return {
        "target_season": target_season,
        "n_common": len(common),
        "v1": projection_accuracy(v1, actual, common),
        "v2": projection_accuracy(v2, actual, common),
    }


def compare_projection_accuracy_v4(
    totals: pd.DataFrame, target_season: str, bios: pd.DataFrame,
    position_map: dict[int, str], league,
) -> dict:
    """Compare V1, V2 and V4 (and V4's two components alone) on one target season.

    Every projection is scored on the SAME players (those all methods can project
    AND that have an actual total) so the comparison is apples-to-apples. Reports
    each variant's MAE/RMSE, isolating whether aging and shrinkage each help.
    """
    v1 = proj_mod.project_v1(totals, target_season)
    v2 = proj_mod.project_v2(totals, target_season)
    v4_aging = proj_mod.project_v4(totals, target_season, bios, position_map, league,
                                   use_aging=True, use_shrinkage=False)
    v4_shrink = proj_mod.project_v4(totals, target_season, bios, position_map, league,
                                    use_aging=False, use_shrinkage=True)
    v4 = proj_mod.project_v4(totals, target_season, bios, position_map, league,
                             use_aging=True, use_shrinkage=True)
    actual = _actual_totals_map(totals, target_season)
    common = (set(v1[C.COL_PLAYER_ID]) & set(v2[C.COL_PLAYER_ID])
              & set(v4[C.COL_PLAYER_ID]) & set(actual))
    return {
        "target_season": target_season,
        "n_common": len(common),
        "n_unknown_age": v4.attrs.get("n_unknown_age", 0),
        "v1": projection_accuracy(v1, actual, common),
        "v2": projection_accuracy(v2, actual, common),
        "v4_aging_only": projection_accuracy(v4_aging, actual, common),
        "v4_shrink_only": projection_accuracy(v4_shrink, actual, common),
        "v4": projection_accuracy(v4, actual, common),
    }


def evaluate_projection_seasons(totals: pd.DataFrame, league) -> list[dict]:
    """Run :func:`compare_projection_accuracy` for every season that has a prior.

    Pooling across multiple target seasons makes the accuracy comparison less
    dependent on a single high-variance season.
    """
    seasons = sorted(totals[C.COL_SEASON].unique())
    results = []
    for target in seasons[1:]:  # every season except the earliest has a prior
        results.append(compare_projection_accuracy(totals, target, league))
    return results


def _roster_detail_v2(player_ids: list[int], board: pd.DataFrame,
                      actual: dict[int, float]) -> pd.DataFrame:
    """Human-readable V2 roster: name, position, projected, VOR, actual."""
    by_id = board.set_index(C.COL_PLAYER_ID)
    return pd.DataFrame({
        C.COL_PLAYER_NAME: [by_id.at[pid, C.COL_PLAYER_NAME] for pid in player_ids],
        C.COL_COARSE_POSITION: [by_id.at[pid, C.COL_COARSE_POSITION] for pid in player_ids],
        C.COL_PROJECTED_PTS: [round(by_id.at[pid, C.COL_PROJECTED_PTS], 1) for pid in player_ids],
        C.COL_VOR: [round(by_id.at[pid, C.COL_VOR], 1) for pid in player_ids],
        "ACTUAL_PTS": [round(actual.get(pid, 0.0), 1) for pid in player_ids],
    })


def backtest_v2(totals: pd.DataFrame, target_season: str, league,
                position_map: dict[int, str]) -> dict:
    """Run the V2 draft (recency PPGxGP projection + VOR board) and score it.

    Drafts greedily off the value-over-replacement board and scores the resulting
    roster on actual target-season totals - directly comparable to ``backtest_v1``
    (same league shape, same scoring, same held-out season).

    Returns:
        dict with ``greedy_total``, ``my_roster`` (DataFrame), ``n_unknown_pos``
        (players the position source could not place), and ``replacement_levels``.
    """
    projection = proj_mod.project_v2(totals, target_season)
    board = board_mod.build_board_vor(projection, position_map, league)
    actual = _actual_totals_map(totals, target_season)

    greedy_rosters = draft_mod.snake_draft(
        board, league.NUM_TEAMS, league.ROSTER_SIZE, league.MY_DRAFT_SLOT,
        strategy="greedy",
    )
    my_roster_ids = greedy_rosters[league.MY_DRAFT_SLOT]
    return {
        "greedy_total": round(_score_roster(my_roster_ids, actual), 1),
        "my_roster": _roster_detail_v2(my_roster_ids, board, actual),
        "n_unknown_pos": board.attrs.get("n_unknown_pos", 0),
        "replacement_levels": board.attrs.get("replacement_levels", {}),
    }


# ===========================================================================
# V3: smart engine vs V2 greedy, scored on the realistic starting-lineup objective.
# ===========================================================================

def _lineup_roster_detail(roster_ids: list[int], board: pd.DataFrame,
                          actual: dict[int, float], starter_ids: set[int]) -> pd.DataFrame:
    """Roster table marked with which players make the actual starting lineup."""
    by_id = board.set_index(C.COL_PLAYER_ID)
    return pd.DataFrame({
        C.COL_PLAYER_NAME: [by_id.at[pid, C.COL_PLAYER_NAME] for pid in roster_ids],
        C.COL_COARSE_POSITION: [by_id.at[pid, C.COL_COARSE_POSITION] for pid in roster_ids],
        C.COL_PROJECTED_PTS: [round(by_id.at[pid, C.COL_PROJECTED_PTS], 1) for pid in roster_ids],
        "ACTUAL_PTS": [round(actual.get(pid, 0.0), 1) for pid in roster_ids],
        "STARTER": ["*" if pid in starter_ids else "" for pid in roster_ids],
    })


def backtest_v3(totals: pd.DataFrame, target_season: str, league,
                position_map: dict[int, str]) -> dict:
    """Compare the V3 smart engine against the V2 greedy engine on the SAME board.

    Both engines draft off the identical V2 value-over-replacement board, so the
    only thing that varies is the draft logic - isolating the engine's effect (the
    V3 definition of done: beat V2's greedy roster). Rosters are scored on the
    realistic objective: the best legal STARTING LINEUP using actual target-season
    points (bench players do not score). The whole-roster sum is reported too for
    continuity with V1/V2.

    Opponents draft greedily in both runs; the engine is only as good as that
    opponent model - the binding constraint flagged in the spec.

    Returns:
        dict with greedy and smart lineup totals, the margin, whole-roster sums,
        roster tables, and the run-risk weight used.
    """
    projection = proj_mod.project_v2(totals, target_season)
    board = board_mod.build_board_vor(projection, position_map, league)
    actual = _actual_totals_map(totals, target_season)
    pos_by_id = {pid: position_map.get(pid, pos_mod.UNKNOWN_POSITION)
                 for pid in board[C.COL_PLAYER_ID]}

    greedy_rosters = draft_mod.snake_draft(
        board, league.NUM_TEAMS, league.ROSTER_SIZE, league.MY_DRAFT_SLOT, strategy="greedy",
    )
    smart_rosters = draft_mod.snake_draft_smart(
        board, league.NUM_TEAMS, league.ROSTER_SIZE, league.MY_DRAFT_SLOT,
        position_map, league,
    )
    greedy_ids = greedy_rosters[league.MY_DRAFT_SLOT]
    smart_ids = smart_rosters[league.MY_DRAFT_SLOT]

    greedy_starters, greedy_lineup = lineup_mod.optimal_lineup(greedy_ids, actual, pos_by_id, league)
    smart_starters, smart_lineup = lineup_mod.optimal_lineup(smart_ids, actual, pos_by_id, league)

    return {
        "greedy_lineup": round(greedy_lineup, 1),
        "smart_lineup": round(smart_lineup, 1),
        "lineup_margin": round(smart_lineup - greedy_lineup, 1),
        "greedy_total": round(_score_roster(greedy_ids, actual), 1),
        "smart_total": round(_score_roster(smart_ids, actual), 1),
        "greedy_roster": _lineup_roster_detail(greedy_ids, board, actual, set(greedy_starters)),
        "smart_roster": _lineup_roster_detail(smart_ids, board, actual, set(smart_starters)),
        "run_risk_weight": league.RUN_RISK_WEIGHT,
    }


# ===========================================================================
# V5: multi-season distribution of the engine against named benchmarks.
# ===========================================================================

def _lineup_of_draft(board: pd.DataFrame, league, actual: dict[int, float],
                     pos_by_id: dict[int, str], *, strategy: str,
                     position_map: dict[int, str] | None = None,
                     seed: int | None = None) -> float:
    """Draft my roster off ``board`` with the given strategy and score its lineup."""
    if strategy == "smart":
        rosters = draft_mod.snake_draft_smart(
            board, league.NUM_TEAMS, league.ROSTER_SIZE, league.MY_DRAFT_SLOT,
            position_map or {}, league)
    else:
        rosters = draft_mod.snake_draft(
            board, league.NUM_TEAMS, league.ROSTER_SIZE, league.MY_DRAFT_SLOT,
            strategy=strategy, seed=seed)
    ids = rosters[league.MY_DRAFT_SLOT]
    return lineup_mod.optimal_lineup_value(ids, actual, pos_by_id, league)


def multi_season_backtest(totals: pd.DataFrame, bios: pd.DataFrame,
                          target_seasons: list[str], league, positions_for,
                          adp_board_for=None) -> dict:
    """Run the V3 engine across several seasons; report a distribution vs benchmarks.

    A single-season result is high variance (the recurring caveat), so the
    trustworthy verdict is a distribution across seasons. The smart engine (V2 board
    + V3 logic) is benchmarked against three named drafts, all scored on the
    realistic starting-lineup objective using actual points:

    - **random** (mean over seeded trials),
    - **prior-season-totals** (the V1 board), and
    - **ADP** - by default an ADP *proxy* from the aging-adjusted V4 ranking (a
      reasonable stand-in for market consensus); pass ``adp_board_for`` to use a
      real ADP source (Yahoo draft results via :mod:`src.yahoo`).

    Args:
        positions_for: ``season -> {player_id: coarse_position}`` callable.
        adp_board_for: optional ``season -> board`` callable for a real ADP draft.

    Returns:
        dict with ``per_season`` rows and a ``summary`` of mean/std margins.
    """
    rows: list[dict] = []
    for target in target_seasons:
        position_map = positions_for(target)
        actual = _actual_totals_map(totals, target)

        v2_board = board_mod.build_board_vor(proj_mod.project_v2(totals, target),
                                             position_map, league)
        prior_board = board_mod.build_board(proj_mod.project_v1(totals, target))
        if adp_board_for is not None:
            adp_board = adp_board_for(target)
        else:
            adp_board = board_mod.build_board(proj_mod.project_v4(
                totals, target, bios, position_map, league,
                use_aging=True, use_shrinkage=False))

        all_ids = (set(v2_board[C.COL_PLAYER_ID]) | set(prior_board[C.COL_PLAYER_ID])
                   | set(adp_board[C.COL_PLAYER_ID]))
        pos_by_id = {pid: position_map.get(pid, pos_mod.UNKNOWN_POSITION) for pid in all_ids}

        smart = _lineup_of_draft(v2_board, league, actual, pos_by_id,
                                 strategy="smart", position_map=position_map)
        prior = _lineup_of_draft(prior_board, league, actual, pos_by_id, strategy="greedy")
        adp = _lineup_of_draft(adp_board, league, actual, pos_by_id, strategy="greedy")
        randoms = [
            _lineup_of_draft(v2_board, league, actual, pos_by_id, strategy="random",
                             seed=league.SEED + t)
            for t in range(league.RANDOM_BASELINE_TRIALS)
        ]
        random_mean = float(pd.Series(randoms).mean())
        rows.append({
            "season": target,
            "smart": round(smart, 1),
            "prior_totals": round(prior, 1),
            "adp_proxy": round(adp, 1),
            "random_mean": round(random_mean, 1),
            "vs_prior": round(smart - prior, 1),
            "vs_adp": round(smart - adp, 1),
            "vs_random": round(smart - random_mean, 1),
        })

    df = pd.DataFrame(rows)
    summary = {
        "seasons": list(df["season"]),
        "vs_prior_mean": round(float(df["vs_prior"].mean()), 1),
        "vs_prior_std": round(float(df["vs_prior"].std(ddof=0)), 1),
        "vs_adp_mean": round(float(df["vs_adp"].mean()), 1),
        "vs_adp_std": round(float(df["vs_adp"].std(ddof=0)), 1),
        "vs_random_mean": round(float(df["vs_random"].mean()), 1),
        "wins_vs_prior": int((df["vs_prior"] > 0).sum()),
        "wins_vs_adp": int((df["vs_adp"] > 0).sum()),
        "n_seasons": len(df),
    }
    return {"per_season": rows, "summary": summary}
