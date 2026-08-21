"""V6 live draft assistant: draft-position selection, recording real picks, and the
engine re-ranking recommendations off the true remaining board (not a simulation).
"""

from __future__ import annotations

import json
import types

import pytest

from src import live_draft as live_mod


def _league() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        NUM_TEAMS=2,
        ROSTER_SIZE=4,
        MY_DRAFT_SLOT=1,
        ROSTER_SLOTS={"G": 1, "F": 1, "C": 1, "UTIL": 1},
        SLOT_TO_COARSE_POSITION={"G": "G", "F": "F", "C": "C", "UTIL": None},
        FLEX_SLOT_NAMES=frozenset({"UTIL"}),
        BENCH_SLOT_NAMES=frozenset(),
        RUN_RISK_WEIGHT=1.0,
    )


# Same shape as test_draft_smart: guards stacked at the top, one centre and one
# forward buried lower - a naive top-of-board pick wastes a redundant guard.
_VALUES = {1: 100.0, 2: 98.0, 3: 96.0, 4: 94.0, 5: 92.0, 6: 50.0, 7: 45.0, 8: 20.0, 9: 15.0}
_POS = {1: "G", 2: "G", 3: "G", 4: "G", 5: "G", 6: "C", 7: "F", 8: "C", 9: "F"}
_NAMES = {i: f"Player {i:02d}" for i in _VALUES}


def _state(my_slot: int = 1) -> live_mod.LiveDraftState:
    league = _league()
    order = [1] * 0  # placeholder, replaced below
    from src import draft as draft_mod
    order = draft_mod.snake_pick_order(league.NUM_TEAMS, league.ROSTER_SIZE)
    board_ids = sorted(_VALUES, key=lambda pid: -_VALUES[pid])
    return live_mod.LiveDraftState(
        order=order, my_slot=my_slot, board_ids=board_ids,
        value_by_id=dict(_VALUES), position_by_id=dict(_POS), name_by_id=dict(_NAMES),
        league=league,
    )


def test_draft_position_selection_drives_the_pick_clock() -> None:
    # 2 teams, 4-round snake: 1,2,2,1,1,2,2,1
    state = _state(my_slot=2)
    assert state.on_the_clock() == 1
    assert state.gap_to_my_next_pick() == 1  # one opponent (team 1) picks before I do


def test_recording_picks_for_every_team_shrinks_the_board_and_advances_the_clock() -> None:
    state = _state(my_slot=1)
    assert state.on_the_clock() == 1
    team = state.record_pick(1)  # I take player 1 (best available)
    assert team == 1
    assert 1 not in state.available_ids()
    assert state.my_roster() == [1]

    assert state.on_the_clock() == 2
    state.record_pick(2)  # opponent's real pick, recorded on my behalf
    assert state.roster_of(2) == [2]
    assert state.on_the_clock() == 2  # round 1 is a snake: team 2 picks twice in a row
    state.record_pick(3)
    assert state.on_the_clock() == 1


def test_cannot_draft_the_same_player_twice() -> None:
    state = _state()
    state.record_pick(1)
    with pytest.raises(ValueError):
        state.record_pick(1)


def test_cannot_draft_an_unknown_player() -> None:
    state = _state()
    with pytest.raises(ValueError):
        state.record_pick(999)


def test_out_of_turn_correction_via_explicit_team_slot() -> None:
    """The `team` override lets any real pick be attributed to the right team, even
    when the recorder was not first to hear about it - the pick CLOCK still just
    advances by one (a pick happened), independent of who it's attributed to."""
    state = _state()
    state.record_pick(5, team_slot=2)  # team 2 actually took player 5, out of clock order
    assert state.roster_of(2) == [5]
    assert state.on_the_clock() == 2  # clock advanced by one pick, as normal


def test_undo_restores_prior_state() -> None:
    state = _state()
    state.record_pick(1)
    state.record_pick(2)
    undone = state.undo_last_pick()
    assert undone == (2, 2, 2)
    assert 2 in state.available_ids()
    assert state.roster_of(2) == []
    assert state.on_the_clock() == 2  # back to team 2's turn


def test_undo_on_empty_log_is_a_noop() -> None:
    state = _state()
    assert state.undo_last_pick() is None


def test_engine_recommendation_adjusts_as_the_board_changes() -> None:
    """The live recommendation must track the ACTUAL remaining board, not a fixed
    pre-draft ranking: once the guards are gone, the top guard-value candidate must
    drop out of the recommendation even though it once looked best."""
    state = _state(my_slot=1)
    top_before = state.recommend(top_n=1)[0]["player_id"]

    # Opponent drains the guard position entirely.
    for guard_id in (1, 2, 3, 4, 5):
        if guard_id in state.available_ids():
            state.record_pick(guard_id, team_slot=2)

    top_after = state.recommend(top_n=1)[0]["player_id"]
    assert top_after in (6, 7, 8, 9)  # only centres/forwards remain
    assert top_after != top_before or top_before not in (1, 2, 3, 4, 5)


def test_recommendation_never_resurrects_idle_depth() -> None:
    """Once my starting slots are full, a redundant guard scores zero marginal
    value even if it is the best remaining player on paper."""
    state = _state(my_slot=1)
    state.record_pick(1)  # G starter
    state.record_pick(6, team_slot=1)  # C starter
    state.record_pick(7, team_slot=1)  # F starter
    state.record_pick(2, team_slot=1)  # UTIL starter (my 4th and final roster slot)

    ranked = state.recommend(top_n=5)
    # No more slots to fill; every remaining candidate should score 0 (roster_size
    # already met, but recommend() should still run without raising).
    assert all(row["marginal_value"] <= 0 for row in ranked) or not state.available_ids()


def test_find_player_matches_partial_and_case_insensitive() -> None:
    state = _state()
    assert 1 in state.find_player("player 01")
    assert 1 in state.find_player("PLAYER 01")
    assert state.find_player("nobody") == []


def test_save_and_resume_round_trip(tmp_path) -> None:
    state = _state(my_slot=1)
    state.record_pick(1)
    state.record_pick(2)
    path = state.save(tmp_path / "state.json")

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["my_slot"] == 1
    assert [tuple(row) for row in saved["pick_log"]] == state.pick_log

    fresh = _state(my_slot=1)
    fresh.load_picks(saved)
    assert fresh.pick_log == state.pick_log
    assert fresh.available_ids() == state.available_ids()
