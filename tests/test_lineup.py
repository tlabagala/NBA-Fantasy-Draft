"""V3 starting-lineup objective tests."""

from __future__ import annotations

import types

from src import lineup as lineup_mod
from src import positions as pos_mod


def _league() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        ROSTER_SLOTS={"G": 2, "F": 2, "C": 1, "UTIL": 1, "BN": 2},
        SLOT_TO_COARSE_POSITION={"G": "G", "F": "F", "C": "C", "UTIL": None, "BN": None},
        FLEX_SLOT_NAMES=frozenset({"UTIL"}),
        BENCH_SLOT_NAMES=frozenset({"BN"}),
    )


def test_starting_requirements_excludes_bench() -> None:
    dedicated, n_flex = lineup_mod.starting_requirements(_league())
    assert dedicated == {"G": 2, "F": 2, "C": 1}
    assert n_flex == 1   # UTIL only; the 2 BN slots do not score


def test_lineup_picks_best_per_slot_and_flex() -> None:
    # 6 starting slots: 2G, 2F, 1C, 1 UTIL.
    values = {
        1: 100, 2: 90, 3: 80,        # guards
        4: 70, 5: 60,                # forwards
        6: 50,                       # centre
        7: 40,                       # bench-quality guard
    }
    pos = {1: "G", 2: "G", 3: "G", 4: "F", 5: "F", 6: "C", 7: "G"}
    starters, value = lineup_mod.optimal_lineup(list(values), values, pos, _league())
    # Best 2 G (100, 90) + 2 F (70, 60) + 1 C (50) + UTIL = best leftover (G 80).
    assert value == 100 + 90 + 70 + 60 + 50 + 80
    assert set(starters) == {1, 2, 4, 5, 6, 3}


def test_extra_depth_at_a_position_does_not_add_value() -> None:
    """A fifth guard adds nothing once guard + flex slots are full - the core V3 idea."""
    league = _league()
    pos = {i: "G" for i in range(1, 8)}
    pos[6] = "F"; pos[5] = "F"  # ensure forwards exist so the lineup is fillable
    values = {1: 100, 2: 95, 3: 90, 4: 85, 5: 80, 6: 75}
    base = lineup_mod.optimal_lineup_value([1, 2, 3, 5, 6], values, pos, league)
    # Adding guard 4 (85) - guard slots (2) + UTIL already covered by better guards.
    with_extra = lineup_mod.optimal_lineup_value([1, 2, 3, 4, 5, 6], values, pos, league)
    assert with_extra == base   # the extra guard is redundant -> zero marginal value


def test_unknown_position_only_fills_flex() -> None:
    league = _league()
    # One UNK star; he can only take the single UTIL slot, not a dedicated slot.
    values = {1: 100, 2: 90, 3: 80, 4: 70, 5: 60, 9: 999}
    pos = {1: "G", 2: "G", 3: "F", 4: "F", 5: "C"}  # 9 missing -> UNK
    starters, value = lineup_mod.optimal_lineup(list(values), values, pos, league)
    assert 9 in starters                       # UNK star takes UTIL
    assert value == 100 + 90 + 80 + 70 + 60 + 999
