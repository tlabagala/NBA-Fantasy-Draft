"""Snake-draft simulation.

V1 supports two strategies off a static board:
- ``"greedy"``: every team always takes the best player remaining on the board.
- ``"random"``: every team takes a uniformly random remaining player (the naive
  baseline the greedy draft is benchmarked against).

There is no opponent model in V1 - all teams use the same strategy. The backtest
documents that it simulates the full draft (every team drafts) rather than
replaying a real historical draft, because no true historical replay exists.
Marginal value, positional-run risk, and an optional Monte Carlo opponent model
are V3.
"""

from __future__ import annotations

import random
from collections import Counter

import pandas as pd

from src import columns as C
from src import lineup as lineup_mod
from src import positions as pos_mod


def snake_pick_order(n_teams: int, roster_size: int) -> list[int]:
    """Return the sequence of team slots (1-based) for a full snake draft.

    Round 1 runs 1..n_teams, round 2 runs n_teams..1, alternating. The result has
    ``n_teams * roster_size`` entries - one per overall pick.
    """
    order: list[int] = []
    for round_index in range(roster_size):
        teams = range(1, n_teams + 1)
        order.extend(teams if round_index % 2 == 0 else reversed(teams))
    return order


def snake_draft(
    board: pd.DataFrame,
    n_teams: int,
    roster_size: int,
    my_slot: int,
    strategy: str = "greedy",
    seed: int | None = None,
) -> dict[int, list[int]]:
    """Simulate a full snake draft and return every team's roster.

    Args:
        board: ranked board from :func:`src.board.build_board`; row order defines
            "best available" for the greedy strategy.
        n_teams: number of teams drafting.
        roster_size: players drafted per team.
        my_slot: your 1-based draft position (validated but not special-cased -
            every team drafts by the same ``strategy``).
        strategy: ``"greedy"`` or ``"random"``.
        seed: seed for the random strategy (ignored by greedy); makes results
            reproducible.

    Returns:
        ``{team_slot: [player_id, ...]}`` for every team slot ``1..n_teams``.
    """
    if my_slot < 1 or my_slot > n_teams:
        raise ValueError(f"my_slot {my_slot} must be within 1..{n_teams}.")
    if strategy not in {"greedy", "random"}:
        raise ValueError(f"Unknown strategy {strategy!r}; expected 'greedy' or 'random'.")

    order = snake_pick_order(n_teams, roster_size)
    rosters: dict[int, list[int]] = {slot: [] for slot in range(1, n_teams + 1)}

    if strategy == "greedy":
        # Board is pre-sorted best-first; pop from the front as players are taken.
        available: list[int] = board[C.COL_PLAYER_ID].tolist()
        for team_slot in order:
            if available:
                rosters[team_slot].append(available.pop(0))
    else:  # random baseline
        rng = random.Random(seed)
        available = board[C.COL_PLAYER_ID].tolist()
        rng.shuffle(available)
        for team_slot in order:
            if available:
                rosters[team_slot].append(available.pop())

    return rosters


# ===========================================================================
# V3: smarter engine - marginal lineup value + positional-run-risk.
# ===========================================================================
# Opponents draft greedily off the board (a simple bot). The engine's validity is
# bounded by this opponent model (master spec: the opponent-model quality is the
# binding constraint); the optional Monte Carlo upgrade, which needs ADP data, is
# deferred until the cheap heuristic is shown to leave value on the table.

def _gaps_to_next_my_pick(order: list[int], my_slot: int) -> dict[int, int]:
    """For each of my pick indices, how many opponent picks fall before my next pick.

    The last of my picks maps to a large gap (no next pick), so its run-risk term
    naturally vanishes.
    """
    my_indices = [i for i, team in enumerate(order) if team == my_slot]
    gaps: dict[int, int] = {}
    for k, idx in enumerate(my_indices):
        if k + 1 < len(my_indices):
            gaps[idx] = my_indices[k + 1] - idx - 1
        else:
            gaps[idx] = len(order)  # effectively no next pick
    return gaps


def _positional_run_risk(
    candidate_pos: str,
    available_ids: list[int],
    value_by_id: dict[int, float],
    position_by_id: dict[int, str],
    gap_to_next: int,
) -> float:
    """Expected value drop-off at a position before my next pick.

    Assumes greedy opponents take the top ``gap_to_next`` board players. The risk
    is how much the best available player at ``candidate_pos`` deteriorates between
    now and my next pick - independent of the candidate himself, so it boosts every
    candidate at a position that is about to run dry.
    """
    if candidate_pos == pos_mod.UNKNOWN_POSITION or gap_to_next <= 0:
        return 0.0
    predicted_taken = available_ids[:gap_to_next]
    n_taken_at_pos = sum(1 for pid in predicted_taken
                         if position_by_id.get(pid, pos_mod.UNKNOWN_POSITION) == candidate_pos)
    pos_values = [value_by_id.get(pid, 0.0) for pid in available_ids
                  if position_by_id.get(pid, pos_mod.UNKNOWN_POSITION) == candidate_pos]
    if not pos_values:
        return 0.0
    best_now = pos_values[0]  # available_ids are board-ordered (best first)
    expected_next = pos_values[n_taken_at_pos] if n_taken_at_pos < len(pos_values) else 0.0
    return max(0.0, best_now - expected_next)


def snake_draft_smart(
    board: pd.DataFrame,
    n_teams: int,
    roster_size: int,
    my_slot: int,
    position_map: dict[int, str],
    league,
    run_risk_weight: float | None = None,
) -> dict[int, list[int]]:
    """Simulate a snake draft where my picks use the V3 smart engine.

    At each of my picks I take the candidate maximising
    ``marginal_lineup_value + run_risk_weight x positional_run_risk``:

    - **Marginal lineup value** - how much the candidate improves my best legal
      starting lineup given the slots I have already filled. This stops me
      over-drafting a deep position, because a fourth guard adds little once my
      guard and flex slots are covered.
    - **Positional-run risk** - the expected drop-off at the candidate's position
      before my next pick, the cheap substitute for a full opponent simulation.

    Every other team drafts greedily off the board (top available). Board order is
    taken as given (V2 sorts it by value over replacement).

    Args:
        board: ranked board with ``PLAYER_ID`` and ``PROJECTED_PTS``.
        n_teams, roster_size, my_slot: league shape and my draft position.
        position_map: player id -> coarse position.
        league: the :mod:`src.config` module (roster slots, mappings).
        run_risk_weight: overrides ``config.RUN_RISK_WEIGHT`` when given.

    Returns:
        ``{team_slot: [player_id, ...]}`` for every team.
    """
    if my_slot < 1 or my_slot > n_teams:
        raise ValueError(f"my_slot {my_slot} must be within 1..{n_teams}.")
    weight = league.RUN_RISK_WEIGHT if run_risk_weight is None else run_risk_weight

    order = snake_pick_order(n_teams, roster_size)
    gaps = _gaps_to_next_my_pick(order, my_slot)
    rosters: dict[int, list[int]] = {slot: [] for slot in range(1, n_teams + 1)}

    value_by_id = dict(zip(board[C.COL_PLAYER_ID], board[C.COL_PROJECTED_PTS]))
    position_by_id = {pid: position_map.get(pid, pos_mod.UNKNOWN_POSITION)
                      for pid in board[C.COL_PLAYER_ID]}
    available: list[int] = board[C.COL_PLAYER_ID].tolist()  # board-ordered, best first

    for idx, team_slot in enumerate(order):
        if not available:
            break
        if team_slot != my_slot:
            rosters[team_slot].append(available.pop(0))  # greedy opponent: top available
            continue
        my_roster = rosters[my_slot]
        base = lineup_mod.optimal_lineup_value(my_roster, value_by_id, position_by_id, league)
        gap = gaps.get(idx, 0)
        best_id, best_score = available[0], float("-inf")
        for cand in available:
            marginal = (
                lineup_mod.optimal_lineup_value(my_roster + [cand], value_by_id,
                                                position_by_id, league)
                - base
            )
            # Positional-run risk only matters for candidates who actually improve my
            # lineup; it must never resurrect idle depth (a surplus guard I cannot
            # start), so it is gated on positive marginal value.
            if marginal > 0:
                run = _positional_run_risk(position_by_id[cand], available, value_by_id,
                                           position_by_id, gap)
                score = marginal + weight * run
            else:
                score = marginal
            if score > best_score:
                best_id, best_score = cand, score
        available.remove(best_id)
        rosters[my_slot].append(best_id)

    return rosters
