"""Backtests: every engine is benchmarked against a NAIVE MODEL baseline.

The reference baseline is the **naive prior-season-totals draft**: a greedy snake
draft off the V1 board, i.e. "take whoever scored most last season". That is the
model a competent-but-unsophisticated human actually drafts, so it is the bar worth
clearing (master spec Section 3 rule 2, which names prior-season-totals and ADP as
the naive drafts). See :func:`naive_baseline_roster`.

The random draft is retained only as a **sanity floor** - evidence that the board
and the pick order are wired up at all, not evidence that the engine is good.
Beating a random draft is a low bar (nobody drafts randomly) and it is reported
last and labelled as a floor everywhere, never as the headline.

Method (time-ordered, no leakage):
1. Project ``target_season`` using only prior-season data (the projection carries
   its own leakage guard).
2. Build the board and run the draft; take your slot's roster.
3. Score every roster by each player's ACTUAL ``target_season`` total - the truth
   the projection could not see.
4. Report the margin over the naive baseline, then the random floor beneath it.

Benchmark honesty (master spec Section 7): this simulates the FULL draft - every
team drafts under the same strategy - rather than replaying a real historical
draft, because no faithful replay of a past draft exists. A single random draft is
high variance, so the floor is the mean over many seeded trials.
"""
from __future__ import annotations

import functools

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


# ---------------------------------------------------------------------------
# Lineup scoring: which information sets the starting lineup.
# ---------------------------------------------------------------------------
# Scoring a roster means picking its best legal starting lineup and summing those
# players' ACTUAL points. But "best" needs a value to sort by, and the choice is a
# real assumption, not a detail:
#
#   "hindsight"  - pick starters using the actual season totals. This is the best
#                  lineup the roster COULD have produced, an upper bound. It is the
#                  original behaviour and remains the default.
#   "projection" - pick starters using the projection the roster was drafted off,
#                  then score those starters on actual points. This is the lineup
#                  you would have set in October and never touched: a lower bound.
#
# Neither is what a real manager achieves. Lineups are set weekly, using in-season
# information that is legitimately available (unlike at the draft), so reality sits
# between the bounds and probably nearer the hindsight end. Reporting both stops the
# optimistic number standing alone. Note this is an EVALUATION choice: no draft
# decision anywhere sees target-season data, which the leakage guards enforce.
LINEUP_VALUE_SOURCES: tuple[str, ...] = ("hindsight", "projection")


def score_lineup(roster: list[int], actual: dict[int, float],
                 selection_values: dict[int, float],
                 position_by_id: dict[int, str], league) -> float:
    """Actual points of the starting lineup chosen by ``selection_values``.

    Args:
        roster: drafted player ids.
        actual: player id -> ACTUAL target-season total (what the lineup scores).
        selection_values: player id -> the value used to DECIDE who starts. Pass
            ``actual`` for the hindsight bound, or the board's projection for the
            projection-picked bound.
        position_by_id, league: as elsewhere.
    """
    starters, _ = lineup_mod.optimal_lineup(roster, selection_values,
                                            position_by_id, league)
    return float(sum(actual.get(pid, 0.0) for pid in starters))


def lineup_bounds(roster: list[int], actual: dict[int, float],
                  projected: dict[int, float], position_by_id: dict[int, str],
                  league) -> dict[str, float]:
    """Both bounds for one roster: ``{"hindsight": ..., "projection": ...}``.

    Computed together because the expensive part is the draft, not the scoring, so
    a caller never has to re-run a backtest to see the other bound.
    """
    return {
        "hindsight": score_lineup(roster, actual, actual, position_by_id, league),
        "projection": score_lineup(roster, actual, projected, position_by_id, league),
    }


def _check_lineup_source(lineup_value_source: str) -> str:
    if lineup_value_source not in LINEUP_VALUE_SOURCES:
        raise ValueError(
            f"lineup_value_source {lineup_value_source!r} must be one of "
            f"{list(LINEUP_VALUE_SOURCES)}.")
    return lineup_value_source


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


# ===========================================================================
# The naive baseline: the reference model every engine is measured against.
# ===========================================================================
# "Take whoever scored most last season." No value over replacement, no lineup
# construction, no aging - just the prior-season total, drafted greedily. This is
# the model a competent human approximates with a printed cheat sheet, and it is
# the bar the spec asks the draft to clear (Section 3, rule 2).

NAIVE_BASELINE_LABEL: str = "naive (prior-season totals)"
RANDOM_FLOOR_LABEL: str = "random draft (sanity floor)"


def naive_baseline_board(totals: pd.DataFrame, target_season: str) -> pd.DataFrame:
    """Board for the naive reference model: prior-season totals, ranked best first.

    Carries the V1 leakage guard via :func:`src.projection.project_v1`, so the
    baseline is held to exactly the same time-ordering rule as the engines it is
    compared against - otherwise the comparison would be meaningless.
    """
    return board_mod.build_board(proj_mod.project_v1(totals, target_season))


def naive_baseline_roster(totals: pd.DataFrame, target_season: str,
                          league) -> tuple[list[int], pd.DataFrame]:
    """My roster from a greedy snake draft off the naive prior-season-totals board.

    Args:
        totals: all season totals (one row per player-season).
        target_season: season to draft and score on.
        league: config module or compatible namespace (NUM_TEAMS, ROSTER_SIZE,
            MY_DRAFT_SLOT).

    Returns:
        ``(my_roster_ids, board)`` - the baseline roster drafted from my slot, and
        the board it came off (kept so callers can render projections alongside).
    """
    board = naive_baseline_board(totals, target_season)
    rosters = draft_mod.snake_draft(
        board, league.NUM_TEAMS, league.ROSTER_SIZE, league.MY_DRAFT_SLOT,
        strategy="greedy",
    )
    return rosters[league.MY_DRAFT_SLOT], board


def _random_trial_totals(board: pd.DataFrame, actual: dict[int, float],
                         league) -> pd.Series:
    """Actual points of my roster in each seeded random draft off ``board``."""
    return pd.Series(
        [
            _score_roster(
                draft_mod.snake_draft(
                    board, league.NUM_TEAMS, league.ROSTER_SIZE, league.MY_DRAFT_SLOT,
                    strategy="random", seed=league.SEED + trial,
                )[league.MY_DRAFT_SLOT],
                actual,
            )
            for trial in range(league.RANDOM_BASELINE_TRIALS)
        ],
        dtype="float64",
    )


def random_floor(board: pd.DataFrame, actual: dict[int, float], league) -> dict:
    """Mean actual points of many seeded RANDOM drafts - the sanity floor only.

    A single random draft is high variance, so this averages
    ``league.RANDOM_BASELINE_TRIALS`` seeded trials. This is deliberately NOT the
    baseline: beating it proves only that the pipeline is wired up, because nobody
    drafts randomly. Report it beneath the naive baseline, never as the headline.

    Returns:
        ``{"mean", "std", "min", "max", "n_trials", "trials"}`` - ``trials`` is the
        per-trial series, so a caller needing a win-rate does not redraft.
    """
    trials = _random_trial_totals(board, actual, league)
    return {
        "mean": round(float(trials.mean()), 1),
        "std": round(float(trials.std(ddof=1)), 1),
        "min": round(float(trials.min()), 1),
        "max": round(float(trials.max()), 1),
        "n_trials": league.RANDOM_BASELINE_TRIALS,
        "trials": trials,
    }


def backtest_naive_baseline(totals: pd.DataFrame, target_season: str, league) -> dict:
    """Score the naive baseline itself, with the random draft beneath it as a floor.

    This establishes the number every later version is measured against. Note that
    the naive model IS the V1 draft (V1's projection is the prior-season total), so
    there is no "V1 versus baseline" margin to report - V1 *is* the baseline. The
    meaningful margins start at V2 (:func:`backtest_v2`) and V3
    (:func:`backtest_v3`).

    Args:
        totals: all season totals (one row per player-season).
        target_season: season to draft and score on.
        league: the :mod:`src.config` module (reads NUM_TEAMS, ROSTER_SIZE,
            MY_DRAFT_SLOT, SEED, RANDOM_BASELINE_TRIALS).

    Returns:
        dict with ``naive_total`` (THE baseline: the naive roster's actual points),
        ``my_roster`` (DataFrame), and the random floor under
        ``random_floor_mean`` / ``_std`` / ``_min`` / ``_max``, ``n_floor_trials``,
        ``floor_margin`` and ``floor_win_rate`` (share of random trials the naive
        roster beats - a wiring check, not a result).
    """
    my_roster_ids, board = naive_baseline_roster(totals, target_season, league)
    actual = _actual_totals_map(totals, target_season)
    naive_total = _score_roster(my_roster_ids, actual)

    floor = random_floor(board, actual, league)
    trials = floor["trials"]

    return {
        "target_season": target_season,
        "baseline_label": NAIVE_BASELINE_LABEL,
        "naive_total": round(naive_total, 1),
        "my_roster": _roster_detail(my_roster_ids, board, actual),
        "random_floor_mean": floor["mean"],
        "random_floor_std": floor["std"],
        "random_floor_min": floor["min"],
        "random_floor_max": floor["max"],
        "n_floor_trials": floor["n_trials"],
        "floor_margin": round(naive_total - floor["mean"], 1),
        "floor_win_rate": round(float((trials < naive_total).mean()), 3),
    }


# Historical name kept so the masterdoc's V1 contract still resolves. V1's draft is
# the naive baseline, so the two are the same run under two names.
backtest_v1 = backtest_naive_baseline


# ===========================================================================
# V2: projection accuracy (error, association and ranking) and the VOR-board backtest.
# ===========================================================================

def _ndcg_at_k(order_by_projection: np.ndarray, actual_values: np.ndarray,
               k: int) -> float:
    """Normalised discounted cumulative gain of a projected ordering.

    Relevance is the player's actual points, discounted by log2 of rank, so getting
    the first pick right counts for far more than the hundredth. Normalised by the
    perfect ordering, giving 1.0 for a flawless board.
    """
    k = min(k, len(actual_values))
    if k == 0:
        return float("nan")
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    gained = np.clip(order_by_projection[:k], 0, None)
    ideal = np.clip(np.sort(actual_values)[::-1][:k], 0, None)
    idcg = float((ideal * discount).sum())
    return float((gained * discount).sum() / idcg) if idcg else float("nan")


def projection_accuracy(projection: pd.DataFrame, actual: dict[int, float],
                        restrict_ids: set[int] | None = None, *,
                        draftable_k: int | None = None,
                        round1_k: int | None = None) -> dict:
    """Score a projection against actual totals on error, correlation and RANKING.

    MAE alone is a poor test for this problem, for two reasons:

    1. **A draft board is a ranking.** Nothing consumes the projected point value -
       the draft consumes the *order*. A projection can be biased by 200 points on
       every player and still produce a perfect board.
    2. **Only the top of the board matters.** In a 12-team, 13-round league exactly
       ``NUM_TEAMS x ROSTER_SIZE`` players are drafted; being wrong about the 400th
       ranked player costs nothing, yet MAE weights him like the first pick.

    So this reports three families:

    - **Error size**: ``mae``, ``rmse``, plus ``medae`` (median absolute error,
      robust to the injury blow-ups that fatten the tail) and ``p90_ae`` (the tail
      itself). ``bias`` is the SIGNED mean error - positive means the projection
      systematically over-predicts, which MAE cannot show.
    - **Association**: ``pearson``, ``r2`` and ``spearman``. Spearman is the rank
      correlation and is the closest single number to "is this a good board".
    - **Top-of-board**: ``precision_at_k`` (share of the players you would draft who
      really belonged in the top k), ``ndcg_at_k`` (rank-weighted, so the early
      picks dominate), ``spearman_at_k`` (ordering quality *among the players you
      would actually draft*) and ``precision_at_round1``.

    Args:
        projection: any projection frame with ``PLAYER_ID`` and ``PROJECTED_PTS``.
        actual: player_id -> actual target-season total (the held-out truth).
        restrict_ids: if given, only score these players (used to compare two
            projections on an identical player set, which is the fair comparison).
        draftable_k: size of the drafted pool, normally ``NUM_TEAMS x ROSTER_SIZE``.
            Top-of-board metrics are ``nan`` when this is omitted.
        round1_k: size of round one, normally ``NUM_TEAMS``.

    Returns:
        A dict of the metrics above; ``n`` is the number of players scored.

    Note:
        ``spearman_at_k`` conditions on the projection's own top k - the players you
        would act on - rather than on the actual top k, so it answers "of the players
        I drafted, did I order them correctly", not "did I find the right players"
        (which is ``precision_at_k``'s job).
    """
    df = projection
    if restrict_ids is not None:
        df = df.loc[df[C.COL_PLAYER_ID].isin(restrict_ids)]
    df = df.loc[df[C.COL_PLAYER_ID].isin(actual)]

    projected = df[C.COL_PROJECTED_PTS].to_numpy(dtype="float64")
    truth = df[C.COL_PLAYER_ID].map(actual).to_numpy(dtype="float64")
    n = len(truth)
    nan = float("nan")
    if n == 0:
        return {"n": 0, "mae": nan, "rmse": nan, "medae": nan, "bias": nan,
                "p90_ae": nan, "pearson": nan, "r2": nan, "spearman": nan,
                "k": draftable_k, "precision_at_k": nan, "ndcg_at_k": nan,
                "spearman_at_k": nan, "precision_at_round1": nan}

    error = projected - truth
    abs_error = np.abs(error)
    ss_tot = float(np.square(truth - truth.mean()).sum())
    proj_s, true_s = pd.Series(projected), pd.Series(truth)

    out = {
        "n": int(n),
        "mae": round(float(abs_error.mean()), 1),
        "rmse": round(float(np.sqrt(np.square(error).mean())), 1),
        "medae": round(float(np.median(abs_error)), 1),
        "bias": round(float(error.mean()), 1),
        "p90_ae": round(float(np.quantile(abs_error, 0.90)), 1),
        "pearson": round(float(proj_s.corr(true_s)), 3) if n > 1 else nan,
        "r2": round(1.0 - float(np.square(error).sum()) / ss_tot, 3) if ss_tot else nan,
        "spearman": round(float(proj_s.corr(true_s, method="spearman")), 3) if n > 1 else nan,
        "k": draftable_k,
        "precision_at_k": nan, "ndcg_at_k": nan, "spearman_at_k": nan,
        "precision_at_round1": nan,
    }
    if not draftable_k:
        return out

    order = np.argsort(-projected, kind="stable")   # best projected first
    ids = df[C.COL_PLAYER_ID].to_numpy()
    k = min(draftable_k, n)
    projected_top = set(ids[order[:k]])
    actual_top = set(ids[np.argsort(-truth, kind="stable")[:k]])
    out["precision_at_k"] = round(len(projected_top & actual_top) / k, 3)
    out["ndcg_at_k"] = round(_ndcg_at_k(truth[order], truth, draftable_k), 3)
    if k > 1:
        out["spearman_at_k"] = round(float(
            pd.Series(projected[order[:k]]).corr(
                pd.Series(truth[order[:k]]), method="spearman")), 3)
    if round1_k:
        r = min(round1_k, n)
        top_r = set(ids[order[:r]])
        actual_r = set(ids[np.argsort(-truth, kind="stable")[:r]])
        out["precision_at_round1"] = round(len(top_r & actual_r) / r, 3)
    return out


def _ranking_k(league) -> dict[str, int | None]:
    """Top-of-board cutoffs derived from the league shape.

    Returns empty cutoffs when ``league`` is None - the error and association
    metrics still compute, the top-of-board ones come back as ``nan``. Some callers
    legitimately have no league to hand (pure projection comparisons in tests).
    """
    if league is None:
        return {"draftable_k": None, "round1_k": None}
    return {"draftable_k": league.NUM_TEAMS * league.ROSTER_SIZE,
            "round1_k": league.NUM_TEAMS}


def compare_projection_accuracy(totals: pd.DataFrame, target_season: str,
                                league) -> dict:
    """Compare V1 vs V2 projection accuracy on a single target season.

    Both projections are scored on the SAME set of players (those each method can
    project AND that have an actual total) so the comparison is apples-to-apples.
    Ranking metrics use the league's own drafted-pool size, so they measure the part
    of the board that is actually consumed.
    """
    v1 = proj_mod.project_v1(totals, target_season)
    v2 = proj_mod.project_v2(totals, target_season)
    actual = _actual_totals_map(totals, target_season)
    common = set(v1[C.COL_PLAYER_ID]) & set(v2[C.COL_PLAYER_ID]) & set(actual)
    score = functools.partial(projection_accuracy, actual=actual,
                              restrict_ids=common, **_ranking_k(league))
    return {
        "target_season": target_season,
        "n_common": len(common),
        "v1": score(v1),
        "v2": score(v2),
    }


def compare_projection_accuracy_v4(
    totals: pd.DataFrame, target_season: str, bios: pd.DataFrame,
    position_map: dict[int, str], league,
) -> dict:
    """Compare V1, V2 and V4 (and V4's two components alone) on one target season.

    Every projection is scored on the SAME players (those all methods can project
    AND that have an actual total) so the comparison is apples-to-apples. Reports
    each variant's full metric set (error size, association and top-of-board
    ranking - see :func:`projection_accuracy`), isolating whether aging and
    shrinkage each help.
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
    score = functools.partial(projection_accuracy, actual=actual,
                              restrict_ids=common, **_ranking_k(league))
    return {
        "target_season": target_season,
        "n_common": len(common),
        "n_unknown_age": v4.attrs.get("n_unknown_age", 0),
        "v1": score(v1),
        "v2": score(v2),
        "v4_aging_only": score(v4_aging),
        "v4_shrink_only": score(v4_shrink),
        "v4": score(v4),
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
    roster on actual target-season totals, then reports the margin over the NAIVE
    BASELINE (:func:`naive_baseline_roster`) - same league shape, same scoring,
    same held-out season, so the only thing that differs is the board.

    Returns:
        dict with ``greedy_total``, ``naive_total`` (the baseline),
        ``margin_vs_naive`` and ``pct_vs_naive``, ``my_roster`` (DataFrame),
        ``n_unknown_pos`` (players the position source could not place), and
        ``replacement_levels``.
    """
    projection = proj_mod.project_v2(totals, target_season)
    board = board_mod.build_board_vor(projection, position_map, league)
    actual = _actual_totals_map(totals, target_season)

    greedy_rosters = draft_mod.snake_draft(
        board, league.NUM_TEAMS, league.ROSTER_SIZE, league.MY_DRAFT_SLOT,
        strategy="greedy",
    )
    my_roster_ids = greedy_rosters[league.MY_DRAFT_SLOT]
    greedy_total = _score_roster(my_roster_ids, actual)

    naive_ids, _naive_board = naive_baseline_roster(totals, target_season, league)
    naive_total = _score_roster(naive_ids, actual)

    return {
        "greedy_total": round(greedy_total, 1),
        "baseline_label": NAIVE_BASELINE_LABEL,
        "naive_total": round(naive_total, 1),
        "margin_vs_naive": round(greedy_total - naive_total, 1),
        "pct_vs_naive": (round((greedy_total - naive_total) / naive_total * 100, 1)
                         if naive_total else float("nan")),
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
                position_map: dict[int, str],
                lineup_value_source: str = "hindsight") -> dict:
    """Compare the V3 smart engine against the V2 greedy engine on the SAME board.

    Both engines draft off the identical V2 value-over-replacement board, so the
    only thing that varies is the draft logic - isolating the engine's effect (the
    V3 definition of done: beat V2's greedy roster). Rosters are scored on the
    realistic objective: the best legal STARTING LINEUP using actual target-season
    points (bench players do not score). The whole-roster sum is reported too for
    continuity with V1/V2.

    The NAIVE BASELINE (:func:`naive_baseline_roster`, drafted off prior-season
    totals) is scored on the same lineup objective and returned alongside, because
    that - not the V2 greedy engine and certainly not a random draft - is the bar
    the engine has to clear to be worth using.

    Opponents draft greedily in all runs; the engine is only as good as that
    opponent model - the binding constraint flagged in the spec.

    Args:
        lineup_value_source: which information picks the starting lineup -
            ``"hindsight"`` (upper bound, the default) or ``"projection"`` (lower
            bound). See :data:`LINEUP_VALUE_SOURCES`. Both bounds are returned
            under ``lineup_bounds`` regardless.

    Returns:
        dict with greedy, smart and naive lineup totals on the chosen source, the
        margins over each, whole-roster sums, roster tables, the run-risk weight
        used, and ``lineup_bounds`` holding both bounds for every roster.
    """
    _check_lineup_source(lineup_value_source)
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

    # Starters are marked on the hindsight lineup so the printed roster tables stay
    # comparable across sources; the reported totals follow lineup_value_source.
    greedy_starters, _ = lineup_mod.optimal_lineup(greedy_ids, actual, pos_by_id, league)
    smart_starters, _ = lineup_mod.optimal_lineup(smart_ids, actual, pos_by_id, league)
    projected = dict(zip(board[C.COL_PLAYER_ID], board[C.COL_PROJECTED_PTS]))
    greedy_bounds = lineup_bounds(greedy_ids, actual, projected, pos_by_id, league)
    smart_bounds = lineup_bounds(smart_ids, actual, projected, pos_by_id, league)
    greedy_lineup = greedy_bounds[lineup_value_source]
    smart_lineup = smart_bounds[lineup_value_source]

    # The naive baseline on the same objective. Its board is prior-season totals,
    # so players it drafts may be absent from the V2 board's position map; they
    # fall back to UNK (flex-eligible only), exactly as in the live engine.
    naive_ids, naive_board = naive_baseline_roster(totals, target_season, league)
    naive_pos_by_id = dict(pos_by_id)
    for pid in naive_ids:
        naive_pos_by_id.setdefault(pid, position_map.get(pid, pos_mod.UNKNOWN_POSITION))
    naive_projected = dict(zip(naive_board[C.COL_PLAYER_ID],
                               naive_board[C.COL_PROJECTED_PTS]))
    naive_bounds_ = lineup_bounds(naive_ids, actual, naive_projected,
                                  naive_pos_by_id, league)
    naive_lineup = naive_bounds_[lineup_value_source]

    return {
        "greedy_lineup": round(greedy_lineup, 1),
        "smart_lineup": round(smart_lineup, 1),
        "lineup_margin": round(smart_lineup - greedy_lineup, 1),
        "baseline_label": NAIVE_BASELINE_LABEL,
        "naive_lineup": round(naive_lineup, 1),
        "margin_vs_naive": round(smart_lineup - naive_lineup, 1),
        "pct_vs_naive": (round((smart_lineup - naive_lineup) / naive_lineup * 100, 1)
                         if naive_lineup else float("nan")),
        "greedy_total": round(_score_roster(greedy_ids, actual), 1),
        "smart_total": round(_score_roster(smart_ids, actual), 1),
        "greedy_roster": _lineup_roster_detail(greedy_ids, board, actual, set(greedy_starters)),
        "smart_roster": _lineup_roster_detail(smart_ids, board, actual, set(smart_starters)),
        "run_risk_weight": league.RUN_RISK_WEIGHT,
        "lineup_value_source": lineup_value_source,
        "lineup_bounds": {
            "greedy": {k: round(v, 1) for k, v in greedy_bounds.items()},
            "smart": {k: round(v, 1) for k, v in smart_bounds.items()},
            "naive": {k: round(v, 1) for k, v in naive_bounds_.items()},
        },
    }


# ===========================================================================
# V5: multi-season distribution of the engine against named benchmarks.
# ===========================================================================

def _lineup_of_draft(board: pd.DataFrame, league, actual: dict[int, float],
                     pos_by_id: dict[int, str], *, strategy: str,
                     position_map: dict[int, str] | None = None,
                     seed: int | None = None,
                     lineup_value_source: str = "hindsight") -> float:
    """Draft my roster off ``board`` with the given strategy and score its lineup.

    Starters are chosen by ``lineup_value_source``: the actual totals (hindsight
    upper bound) or ``board``'s own projection (projection-picked lower bound).
    """
    if strategy == "smart":
        rosters = draft_mod.snake_draft_smart(
            board, league.NUM_TEAMS, league.ROSTER_SIZE, league.MY_DRAFT_SLOT,
            position_map or {}, league)
    else:
        rosters = draft_mod.snake_draft(
            board, league.NUM_TEAMS, league.ROSTER_SIZE, league.MY_DRAFT_SLOT,
            strategy=strategy, seed=seed)
    ids = rosters[league.MY_DRAFT_SLOT]
    selection = (actual if lineup_value_source == "hindsight"
                 else dict(zip(board[C.COL_PLAYER_ID], board[C.COL_PROJECTED_PTS])))
    return score_lineup(ids, actual, selection, pos_by_id, league)


def multi_season_backtest(totals: pd.DataFrame, bios: pd.DataFrame,
                          target_seasons: list[str], league, positions_for,
                          adp_board_for=None,
                          lineup_value_source: str = "hindsight") -> dict:
    """Run the V3 engine across several seasons; report a distribution vs benchmarks.

    A single-season result is high variance (the recurring caveat), so the
    trustworthy verdict is a distribution across seasons. The smart engine (V2 board
    + V3 logic) is benchmarked on the realistic starting-lineup objective using
    actual points, against, in order of how much they matter:

    1. **prior-season totals** - THE naive baseline (:data:`NAIVE_BASELINE_LABEL`).
       This is the headline margin: the bar a competent human sets.
    2. **ADP** - by default an ADP *proxy* from the aging-adjusted V4 ranking (a
       reasonable stand-in for market consensus); pass ``adp_board_for`` to use a
       real ADP source (Yahoo draft results via :mod:`src.yahoo`).
    3. **random floor** - mean over seeded trials, reported last and labelled a
       floor. Beating it is a wiring check, not a result.

    Args:
        positions_for: ``season -> {player_id: coarse_position}`` callable.
        adp_board_for: optional ``season -> board`` callable for a real ADP draft.
        lineup_value_source: which information picks the starting lineup -
            ``"hindsight"`` (upper bound, the default) or ``"projection"`` (lower
            bound). See :data:`LINEUP_VALUE_SOURCES`.

    Returns:
        dict with ``per_season`` rows (naive baseline first, random floor last) and
        a ``summary`` whose ``baseline`` names the reference model and whose
        ``lineup_value_source`` names the scoring assumption.
    """
    _check_lineup_source(lineup_value_source)
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
                                 strategy="smart", position_map=position_map,
                                 lineup_value_source=lineup_value_source)
        prior = _lineup_of_draft(prior_board, league, actual, pos_by_id,
                                 strategy="greedy",
                                 lineup_value_source=lineup_value_source)
        adp = _lineup_of_draft(adp_board, league, actual, pos_by_id,
                               strategy="greedy",
                               lineup_value_source=lineup_value_source)
        randoms = [
            _lineup_of_draft(v2_board, league, actual, pos_by_id, strategy="random",
                             seed=league.SEED + t,
                             lineup_value_source=lineup_value_source)
            for t in range(league.RANDOM_BASELINE_TRIALS)
        ]
        random_floor_mean = float(pd.Series(randoms).mean())
        def _pct(baseline: float) -> float:
            """Percentage improvement of the smart engine over a baseline."""
            return round((smart - baseline) / baseline * 100, 1) if baseline else float("nan")

        # Key order is the reporting order: the naive baseline leads, the random
        # floor trails. Anything reading this frame column-wise inherits that.
        rows.append({
            "season": target,
            "prior_totals": round(prior, 1),   # THE baseline (naive model)
            "smart": round(smart, 1),
            "vs_prior": round(smart - prior, 1),
            "pct_vs_prior": _pct(prior),
            "adp_proxy": round(adp, 1),
            "vs_adp": round(smart - adp, 1),
            "pct_vs_adp": _pct(adp),
            "random_floor": round(random_floor_mean, 1),   # sanity floor only
            "vs_random_floor": round(smart - random_floor_mean, 1),
            "pct_vs_random_floor": _pct(random_floor_mean),
        })

    df = pd.DataFrame(rows)
    summary = {
        "seasons": list(df["season"]),
        "baseline": NAIVE_BASELINE_LABEL,
        "lineup_value_source": lineup_value_source,
        # --- headline: the engine against the naive baseline ---
        "vs_prior_mean": round(float(df["vs_prior"].mean()), 1),
        "vs_prior_std": round(float(df["vs_prior"].std(ddof=0)), 1),
        "pct_vs_prior_mean": round(float(df["pct_vs_prior"].mean()), 1),
        "pct_vs_prior_min": round(float(df["pct_vs_prior"].min()), 1),
        "pct_vs_prior_max": round(float(df["pct_vs_prior"].max()), 1),
        "wins_vs_prior": int((df["vs_prior"] > 0).sum()),
        # --- secondary: against market consensus ---
        "vs_adp_mean": round(float(df["vs_adp"].mean()), 1),
        "vs_adp_std": round(float(df["vs_adp"].std(ddof=0)), 1),
        "pct_vs_adp_mean": round(float(df["pct_vs_adp"].mean()), 1),
        "wins_vs_adp": int((df["vs_adp"] > 0).sum()),
        # --- floor: a wiring check, never the headline ---
        "vs_random_floor_mean": round(float(df["vs_random_floor"].mean()), 1),
        "pct_vs_random_floor_mean": round(float(df["pct_vs_random_floor"].mean()), 1),
        "wins_vs_random_floor": int((df["vs_random_floor"] > 0).sum()),
        "n_seasons": len(df),
    }
    return {"per_season": rows, "summary": summary}


# ===========================================================================
# Head-to-head: hold the LEAGUE fixed, vary only MY version.
# ===========================================================================
# Every backtest above hands the same board to all twelve teams, so two runs differ
# by an entire league's worth of information. That inflates any version whose board
# is better in the abstract: if everyone drafts off the aging-adjusted board, the
# draft as a whole improves and my slot inherits some of it, whether or not I did
# anything smarter than my opponents.
#
# This section asks the narrower, decision-relevant question instead: my eleven
# opponents all draft naively (best available by last season's totals), and only MY
# board and strategy change. The margin over drafting naively myself, into that same
# room, is the edge a version actually buys me.

# (label, which board I draft off, how I draft off it)
DEFAULT_HEAD_TO_HEAD_VARIANTS: tuple[tuple[str, str, str], ...] = (
    ("V2 VOR board, greedy", "v2", "greedy"),
    ("V3 engine on V2 board", "v2", "smart"),
    ("V4 aging board, greedy", "v4", "greedy"),
    ("V4 board + V3 engine", "v4", "smart"),
)


def head_to_head_backtest(totals: pd.DataFrame, bios: pd.DataFrame,
                          target_seasons: list[str], league, positions_for,
                          variants=DEFAULT_HEAD_TO_HEAD_VARIANTS,
                          slots: list[int] | None = None,
                          lineup_value_source: str = "hindsight") -> dict:
    """Measure each version's edge over naive, with the opposing league held fixed.

    For every target season, all opponents draft greedily off the naive
    prior-season-totals board. I then draft into that same room once per variant,
    plus once naively myself - which is the baseline, so its margin is 0 by
    construction. Everything is scored on the starting-lineup objective with actual
    points, exactly as elsewhere.

    **Every draft slot is evaluated, not just mine.** Running one slot makes the
    result far noisier than it looks: measured across three seasons, the spread
    across slots within a single season is as large as the spread between seasons,
    so a single-slot verdict attributes to "seasons" a lot of variance that is
    really about where you happened to pick. Averaging the twelve slots into one
    number per season cut the seasons needed to resolve the V3 engine's edge at 80%
    power from ~31 to ~5 - the cheapest variance reduction available here, since it
    needs no extra data at all.

    This is the honest counterpart to :func:`multi_season_backtest`. Read them
    together: where they disagree, this one is measuring the thing a draft assistant
    is for, and the other is measuring what happens when the whole league upgrades.

    Args:
        totals: all season totals (one row per player-season).
        bios: player ages, for the V4 aging board.
        target_seasons: the held-out seasons to run.
        league: the :mod:`src.config` module or a compatible namespace.
        positions_for: ``season -> {player_id: coarse_position}`` callable.
        variants: ``[(label, board, strategy), ...]`` where ``board`` is
            ``"naive"``, ``"v2"`` or ``"v4"`` and ``strategy`` is ``"greedy"`` or
            ``"smart"``. Defaults to :data:`DEFAULT_HEAD_TO_HEAD_VARIANTS`.
        slots: draft slots to average over. Defaults to every slot
            ``1..league.NUM_TEAMS``; pass ``[league.MY_DRAFT_SLOT]`` for the old
            single-slot behaviour, which is faster but much noisier.
        lineup_value_source: which information picks the starting lineup -
            ``"hindsight"`` (upper bound, the default) or ``"projection"`` (lower
            bound). See :data:`LINEUP_VALUE_SOURCES`. **Both** bounds are always
            computed and reported in the summary, because the cost here is the
            drafting, not the scoring - this only selects which one drives
            ``per_season`` and the headline summary figures.

    Returns:
        dict with:

        - ``per_season`` - one row per season, slot-AVERAGED. Absolute points and
          absolute margins are means across slots (so margin still equals variant
          minus baseline); ``pct:`` columns are the mean of the per-slot
          percentages, which is the equally-weighted "average edge" and does NOT
          equal the ratio of the two averaged columns.
        - ``per_slot_season`` - the full (season x slot) grid behind it.
        - ``summary`` - per-variant mean/worst/best of the slot-averaged season
          figures, seasons won, ``slot_spread_pct`` (mean within-season standard
          deviation across slots), ``share_positive`` over every slot-season
          observation, and ``mean_pct_hindsight`` / ``mean_pct_projection`` giving
          both lineup-scoring bounds side by side.

    Note:
        Opponents remain non-reactive: they never adjust to what I take. This is a
        better opponent model than "everyone runs my strategy", not a good one.
    """
    _check_lineup_source(lineup_value_source)
    slot_list = list(slots) if slots is not None else list(range(1, league.NUM_TEAMS + 1))
    if not slot_list:
        raise ValueError("slots must name at least one draft slot.")
    bad = [s for s in slot_list if s < 1 or s > league.NUM_TEAMS]
    if bad:
        raise ValueError(f"slots {bad} are outside 1..{league.NUM_TEAMS}.")

    grid: list[dict] = []
    for target in target_seasons:
        position_map = positions_for(target)
        actual = _actual_totals_map(totals, target)

        # Boards depend only on the season, so build them once and reuse across
        # every slot - otherwise slot averaging would re-run the whole projection
        # and board pipeline NUM_TEAMS times per season for no reason.
        boards = {
            "naive": naive_baseline_board(totals, target),
            "v2": board_mod.build_board_vor(
                proj_mod.project_v2(totals, target), position_map, league),
            "v4": board_mod.build_board_vor(
                proj_mod.project_v4(totals, target, bios, position_map, league,
                                    use_aging=True, use_shrinkage=False),
                position_map, league),
        }
        all_ids: set[int] = set()
        for board in boards.values():
            all_ids |= set(board[C.COL_PLAYER_ID])
        pos_by_id = {pid: position_map.get(pid, pos_mod.UNKNOWN_POSITION) for pid in all_ids}

        for slot in slot_list:
            def _bounds(board_key: str, strategy: str) -> dict[str, float]:
                """Draft off ``board_key``, then score BOTH lineup bounds at once.

                One draft, two scorings - the draft is the expensive half, so the
                second bound is close to free.
                """
                roster = draft_mod.snake_draft_vs_fixed_opponents(
                    boards[board_key], boards["naive"], league.NUM_TEAMS,
                    league.ROSTER_SIZE, slot, position_map, league, strategy=strategy,
                )
                projected = dict(zip(boards[board_key][C.COL_PLAYER_ID],
                                     boards[board_key][C.COL_PROJECTED_PTS]))
                return lineup_bounds(roster, actual, projected, pos_by_id, league)

            base = _bounds("naive", "greedy")
            baseline = base[lineup_value_source]
            row = {"season": target, "slot": slot, "naive_baseline": baseline}
            for label, board_key, strategy in variants:
                bounds = _bounds(board_key, strategy)
                value = bounds[lineup_value_source]
                row[label] = value
                row[f"vs naive: {label}"] = value - baseline
                row[f"pct: {label}"] = ((value - baseline) / baseline * 100
                                        if baseline else float("nan"))
                # Both bounds, so the optimistic number never stands alone.
                for source in LINEUP_VALUE_SOURCES:
                    ref = base[source]
                    row[f"pct[{source}]: {label}"] = (
                        (bounds[source] - ref) / ref * 100 if ref else float("nan"))
            grid.append(row)

    grid_df = pd.DataFrame(grid)
    # Slot-average each season. Absolute columns and absolute margins average
    # linearly, so margin == variant - baseline still holds after averaging; the
    # pct columns are the mean of per-slot percentages (see the docstring).
    season_df = grid_df.drop(columns="slot").groupby("season", as_index=False).mean()
    rows = [
        {k: (v if k == "season" else round(float(v), 1)) for k, v in row.items()}
        for row in season_df.to_dict("records")
    ]

    summary = {
        "baseline": NAIVE_BASELINE_LABEL,
        "seasons": list(season_df["season"]),
        "n_seasons": len(season_df),
        "slots": slot_list,
        "n_slots": len(slot_list),
        "lineup_value_source": lineup_value_source,
        "variants": {},
    }
    for label, _board_key, _strategy in variants:
        margins, pcts = season_df[f"vs naive: {label}"], season_df[f"pct: {label}"]
        per_slot = grid_df[f"pct: {label}"]
        spread = (grid_df.groupby("season")[f"pct: {label}"].std(ddof=1).mean()
                  if len(slot_list) > 1 else 0.0)
        summary["variants"][label] = {
            "mean_margin": round(float(margins.mean()), 1),
            "mean_pct": round(float(pcts.mean()), 1),
            "mean_pct_hindsight": round(float(season_df[f"pct[hindsight]: {label}"].mean()), 1),
            "mean_pct_projection": round(float(season_df[f"pct[projection]: {label}"].mean()), 1),
            "worst_pct": round(float(pcts.min()), 1),
            "best_pct": round(float(pcts.max()), 1),
            "seasons_won": int((margins > 0).sum()),
            "slot_spread_pct": round(float(spread), 1),
            "share_positive": round(float((per_slot > 0).mean()), 3),
        }
    return {"per_season": rows, "per_slot_season": grid_df.to_dict("records"),
            "summary": summary}
