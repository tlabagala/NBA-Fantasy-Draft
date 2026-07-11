"""Starting-lineup value - the realistic H2H objective the V3 engine optimises.

In a head-to-head points league only your STARTERS score each week, so a roster is
worth its best legal starting lineup, not the sum of all drafted players. Depth at
a position therefore has diminishing returns (you can only start so many guards),
which is exactly what makes the V3 marginal-value and positional-run logic matter.

Positions are the coarse ``G/F/C`` from :mod:`src.positions`. A player with no
known position (``UNK``) can only fill a flexible (UTIL) slot, never a dedicated
G/F/C slot. Bench slots do not score (a documented simplification - bench value via
streaming and injury cover is not modelled until later, if at all).

Limitation: with only coarse positions, dedicated PG vs SG (etc.) cannot be told
apart, so guard slots are pooled as ``G``. True Yahoo multi-position eligibility
(V5) would refine this.
"""

from __future__ import annotations

from src import positions as pos_mod


def starting_requirements(league) -> tuple[dict[str, int], int]:
    """Derive the coarse starting requirements from the league roster slots.

    Returns:
        ``(dedicated, n_flex)`` where ``dedicated`` maps each coarse position
        ``{G, F, C}`` to its number of dedicated starting slots, and ``n_flex`` is
        the number of any-position (UTIL) starting slots. Bench/IL slots are
        excluded - they do not score.
    """
    dedicated = {p: 0 for p in pos_mod.COARSE_POSITIONS}
    n_flex = 0
    for slot, count in league.ROSTER_SLOTS.items():
        coarse = league.SLOT_TO_COARSE_POSITION.get(slot)
        if coarse in dedicated:
            dedicated[coarse] += count
        elif slot in league.FLEX_SLOT_NAMES:
            n_flex += count
        # else: bench/IL - not a scoring slot, ignored.
    return dedicated, n_flex


def optimal_lineup(
    roster: list[int],
    value_by_id: dict[int, float],
    position_by_id: dict[int, str],
    league,
) -> tuple[list[int], float]:
    """Return the best legal starting lineup as ``(starter_ids, total_value)``.

    Optimality for this slot structure is simple: each dedicated G/F/C slot must be
    filled by that position, so fill it with that position's best available
    players; then the flexible (UTIL) slots take the best of everyone left over.

    Args:
        roster: drafted player ids.
        value_by_id: player id -> value (projected points when drafting, actual
            points when scoring a backtest). Missing ids count as 0.
        position_by_id: player id -> coarse position; missing ids are treated as
            ``UNK`` (flex-eligible only).
        league: the :mod:`src.config` module (roster slots and mappings).

    Returns:
        ``(starter_ids, total_value)`` - the chosen starters and their summed value.
    """
    dedicated, n_flex = starting_requirements(league)

    buckets: dict[str, list[int]] = {p: [] for p in pos_mod.COARSE_POSITIONS}
    flex_only: list[int] = []  # unknown-position players: UTIL-eligible only
    for pid in roster:
        position = position_by_id.get(pid, pos_mod.UNKNOWN_POSITION)
        (buckets[position] if position in buckets else flex_only).append(pid)

    def _value(pid: int) -> float:
        return value_by_id.get(pid, 0.0)

    starters: list[int] = []
    leftovers: list[int] = list(flex_only)
    for position, capacity in dedicated.items():
        ranked = sorted(buckets[position], key=_value, reverse=True)
        starters.extend(ranked[:capacity])
        leftovers.extend(ranked[capacity:])

    leftovers.sort(key=_value, reverse=True)
    starters.extend(leftovers[:n_flex])
    return starters, float(sum(_value(pid) for pid in starters))


def optimal_lineup_value(
    roster: list[int],
    value_by_id: dict[int, float],
    position_by_id: dict[int, str],
    league,
) -> float:
    """Value of a roster's best legal starting lineup (see :func:`optimal_lineup`)."""
    return optimal_lineup(roster, value_by_id, position_by_id, league)[1]
