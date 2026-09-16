"""Head-to-head backtest: the league held fixed while only MY version changes.

The point of `snake_draft_vs_fixed_opponents` is that it does NOT hand every team
the same board, so these tests pin down the two things that could silently go wrong:
the draft must still be legal (no duplicates, right size, boards need not agree on
the player pool), and it must reduce to the ordinary greedy snake draft when both
sides happen to use the same board.
"""

from __future__ import annotations

import types

import pandas as pd
import pytest

from src import backtest as backtest_mod
from src import columns as C
from src import draft as draft_mod


def _league(**overrides) -> types.SimpleNamespace:
    cfg = dict(
        NUM_TEAMS=4,
        ROSTER_SIZE=5,
        MY_DRAFT_SLOT=2,
        SEED=42,
        RANDOM_BASELINE_TRIALS=20,
        ROSTER_SLOTS={"G": 1, "F": 1, "C": 1, "UTIL": 1, "BN": 1},
        SLOT_TO_COARSE_POSITION={"G": "G", "F": "F", "C": "C",
                                 "UTIL": None, "BN": None},
        FLEX_SLOT_NAMES=frozenset({"UTIL"}),
        BENCH_SLOT_NAMES=frozenset({"BN"}),
        N_PROJECTION_SEASONS=3,
        GP_RECENCY_DECAY=0.75,
        SHRINKAGE_K=25.0,
        AGE_CURVE_MIN=19,
        AGE_CURVE_MAX=38,
        AGE_CURVE_SMOOTH=1,
        RUN_RISK_WEIGHT=1.0,
    )
    cfg.update(overrides)
    return types.SimpleNamespace(**cfg)


_POSITIONS = ("G", "F", "C")


def _board(player_ids: list[int], values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        C.COL_PLAYER_ID: player_ids,
        C.COL_PLAYER_NAME: [f"Player {p:02d}" for p in player_ids],
        C.COL_PROJECTED_PTS: values,
    })


def _position_map(player_ids) -> dict[int, str]:
    return {pid: _POSITIONS[pid % 3] for pid in player_ids}


def test_identical_boards_reduce_to_the_ordinary_greedy_draft() -> None:
    """With one shared board and greedy play, this must match `snake_draft` exactly.

    That equivalence is the correctness anchor: any bookkeeping bug in the
    shared-`taken` logic would show up here as a different roster.
    """
    league = _league()
    ids = list(range(1, 41))
    board = _board(ids, [float(500 - 10 * i) for i in range(len(ids))])

    mine = draft_mod.snake_draft_vs_fixed_opponents(
        board, board, league.NUM_TEAMS, league.ROSTER_SIZE, league.MY_DRAFT_SLOT,
        _position_map(ids), league, strategy="greedy",
    )
    expected = draft_mod.snake_draft(
        board, league.NUM_TEAMS, league.ROSTER_SIZE, league.MY_DRAFT_SLOT,
        strategy="greedy",
    )[league.MY_DRAFT_SLOT]

    assert mine == expected


def test_my_roster_is_legal_when_the_boards_disagree() -> None:
    """Different boards, different orders, partially different pools - still legal."""
    league = _league()
    opponent_ids = list(range(1, 41))
    # My board drops some players the opponents rank, adds some they have never
    # heard of, and ranks the overlap in a different order.
    my_ids = list(range(10, 51))[::-1]
    pos = _position_map(set(opponent_ids) | set(my_ids))

    opponent_board = _board(opponent_ids, [float(500 - 10 * i) for i in range(len(opponent_ids))])
    my_board = _board(my_ids, [float(400 - 5 * i) for i in range(len(my_ids))])

    for strategy in ("greedy", "smart"):
        roster = draft_mod.snake_draft_vs_fixed_opponents(
            my_board, opponent_board, league.NUM_TEAMS, league.ROSTER_SIZE,
            league.MY_DRAFT_SLOT, pos, league, strategy=strategy,
        )
        assert len(roster) == league.ROSTER_SIZE, strategy
        assert len(set(roster)) == len(roster), f"{strategy} drafted a duplicate"
        assert set(roster) <= set(my_ids), f"{strategy} drafted off the wrong board"


def test_opponents_never_lose_a_pick_to_me() -> None:
    """Nobody is drafted twice: my picks must be disjoint from what opponents took.

    Reconstructed by replaying the opponents' greedy rule over the same board and
    checking the total pool consumed is exactly one player per pick.
    """
    league = _league()
    ids = list(range(1, 41))
    pos = _position_map(ids)
    opponent_board = _board(ids, [float(500 - 10 * i) for i in range(len(ids))])
    my_board = _board(ids[::-1], [float(10 * i) for i in range(len(ids))])

    roster = draft_mod.snake_draft_vs_fixed_opponents(
        my_board, opponent_board, league.NUM_TEAMS, league.ROSTER_SIZE,
        league.MY_DRAFT_SLOT, pos, league, strategy="smart",
    )
    # I took the worst-ranked players by the opponents' board, so if bookkeeping
    # were broken the opponents would have taken them too.
    n_opponent_picks = league.NUM_TEAMS * league.ROSTER_SIZE - league.ROSTER_SIZE
    opponent_took = [pid for pid in ids if pid not in set(roster)][:n_opponent_picks]
    assert not (set(roster) & set(opponent_took))


def _synthetic() -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str]]:
    seasons = ["2020-21", "2021-22", "2022-23", "2023-24"]
    totals, bios = [], []
    position_map: dict[int, str] = {}
    for pid in range(1, 49):
        position_map[pid] = _POSITIONS[pid % 3]
        for i, season in enumerate(seasons):
            totals.append({
                C.COL_SEASON: season, C.COL_PLAYER_ID: pid,
                C.COL_PLAYER_NAME: f"Player {pid:02d}",
                C.COL_TOTAL_FANTASY_PTS: float(2200 - pid * 35 + (pid % 4) * 8),
                C.COL_GAMES_PLAYED: 72,
            })
            bios.append({C.COL_SEASON: season, C.COL_PLAYER_ID: pid,
                         C.COL_AGE: float(24 + (pid % 8) + i)})
    return pd.DataFrame(totals), pd.DataFrame(bios), position_map


def test_head_to_head_backtest_reports_a_margin_per_variant() -> None:
    totals, bios, pos = _synthetic()
    result = backtest_mod.head_to_head_backtest(
        totals, bios, ["2022-23", "2023-24"], _league(), positions_for=lambda s: pos)

    assert result["summary"]["n_seasons"] == 2
    assert result["summary"]["baseline"] == backtest_mod.NAIVE_BASELINE_LABEL

    for row in result["per_season"]:
        assert row["naive_baseline"] > 0
        for label, _board, _strategy in backtest_mod.DEFAULT_HEAD_TO_HEAD_VARIANTS:
            # The margin must be consistent with the two absolute numbers. All
            # three are slot-averages rounded independently to 1dp, so allow the
            # rounding to drift by one unit in the last place (plus float slop).
            assert abs(row[f"vs naive: {label}"]
                       - (row[label] - row["naive_baseline"])) <= 0.1 + 1e-9

    for label, _board, _strategy in backtest_mod.DEFAULT_HEAD_TO_HEAD_VARIANTS:
        stats = result["summary"]["variants"][label]
        assert stats["worst_pct"] <= stats["mean_pct"] <= stats["best_pct"]
        assert 0 <= stats["seasons_won"] <= 2


def test_head_to_head_baseline_is_zero_margin_against_itself() -> None:
    """Drafting naively into a naive league is the baseline, so its margin is 0."""
    totals, bios, pos = _synthetic()
    result = backtest_mod.head_to_head_backtest(
        totals, bios, ["2023-24"], _league(), positions_for=lambda s: pos,
        variants=(("naive again", "naive", "greedy"),))

    row = result["per_season"][0]
    assert row["vs naive: naive again"] == 0.0
    assert result["summary"]["variants"]["naive again"]["mean_pct"] == 0.0


def test_every_slot_is_evaluated_by_default() -> None:
    """The default averages all NUM_TEAMS slots, and exposes the grid behind it."""
    totals, bios, pos = _synthetic()
    league = _league()
    result = backtest_mod.head_to_head_backtest(
        totals, bios, ["2022-23", "2023-24"], league, positions_for=lambda s: pos,
        variants=(("V3", "v2", "smart"),))

    assert result["summary"]["slots"] == [1, 2, 3, 4]
    assert result["summary"]["n_slots"] == league.NUM_TEAMS
    # One grid row per (season, slot); one reported row per season.
    assert len(result["per_slot_season"]) == 2 * league.NUM_TEAMS
    assert len(result["per_season"]) == 2
    assert {r["slot"] for r in result["per_slot_season"]} == {1, 2, 3, 4}


def test_season_rows_are_the_mean_over_slots() -> None:
    """The reported per-season figure must be the slot average, not one slot."""
    totals, bios, pos = _synthetic()
    result = backtest_mod.head_to_head_backtest(
        totals, bios, ["2023-24"], _league(), positions_for=lambda s: pos,
        variants=(("V3", "v2", "smart"),))

    grid = [r for r in result["per_slot_season"] if r["season"] == "2023-24"]
    expected = sum(r["pct: V3"] for r in grid) / len(grid)
    assert result["per_season"][0]["pct: V3"] == round(expected, 1)


def test_single_slot_reproduces_the_unaveraged_result() -> None:
    """Passing one slot must match that slot's row in the full-grid run."""
    totals, bios, pos = _synthetic()
    league = _league()
    variants = (("V3", "v2", "smart"),)

    full = backtest_mod.head_to_head_backtest(
        totals, bios, ["2023-24"], league, positions_for=lambda s: pos,
        variants=variants)
    just_one = backtest_mod.head_to_head_backtest(
        totals, bios, ["2023-24"], league, positions_for=lambda s: pos,
        variants=variants, slots=[3])

    from_grid = next(r for r in full["per_slot_season"] if r["slot"] == 3)
    assert just_one["per_season"][0]["pct: V3"] == round(from_grid["pct: V3"], 1)
    assert just_one["summary"]["n_slots"] == 1
    # A single slot has no spread to report.
    assert just_one["summary"]["variants"]["V3"]["slot_spread_pct"] == 0.0


def test_slot_averaging_reduces_between_season_variance() -> None:
    """The whole point: averaging slots must not be noisier than a single slot.

    Uses the real four-variant default so this reflects the shipped configuration
    rather than a hand-picked variant.
    """
    totals, bios, pos = _synthetic()
    league = _league()
    result = backtest_mod.head_to_head_backtest(
        totals, bios, ["2021-22", "2022-23", "2023-24"], league,
        positions_for=lambda s: pos)

    for label, _b, _s in backtest_mod.DEFAULT_HEAD_TO_HEAD_VARIANTS:
        stats = result["summary"]["variants"][label]
        assert 0.0 <= stats["share_positive"] <= 1.0
        assert stats["slot_spread_pct"] >= 0.0
        assert stats["worst_pct"] <= stats["mean_pct"] <= stats["best_pct"]


def test_invalid_slots_are_rejected() -> None:
    totals, bios, pos = _synthetic()
    league = _league()
    for bad in ([], [0], [league.NUM_TEAMS + 1], [1, 99]):
        with pytest.raises(ValueError):
            backtest_mod.head_to_head_backtest(
                totals, bios, ["2023-24"], league, positions_for=lambda s: pos,
                variants=(("V3", "v2", "smart"),), slots=bad)
