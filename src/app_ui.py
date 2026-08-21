"""V7: a Streamlit web UI wrapped around the V6 live draft assistant.

V6 (``src/live_draft.py``) already works end to end as a terminal REPL - every
pick typed by hand, ambiguous names resolved with a follow-up prompt. That is
fine for a handful of picks but does not survive a real 12-team, 13-round draft,
where roughly 156 picks must be recorded against a clock and eleven of every
twelve are somebody else's. V7 changes nothing about the projection, the board or
the recommendation logic - it is purely an entry-speed and situational-awareness
layer over :class:`src.live_draft.LiveDraftState`.

Two things make a Streamlit app non-trivial here, and both are handled deliberately:

1. **Streamlit re-runs this whole script on every interaction.** The board build
   (ingest -> scoring -> V4 projection -> value-over-replacement board) is
   expensive and slot-independent, so it is wrapped in ``st.cache_data`` via
   :func:`_cached_board_bundle`, which calls
   :func:`src.live_draft.build_board_bundle` - the one engine change V7 makes
   (splitting the slot-independent half out of ``build_live_draft``, which is
   otherwise unchanged and still covered by ``tests/test_live_draft.py``).
2. **``LiveDraftState`` is mutable** (``record_pick`` mutates it in place), so it
   lives ONLY in ``st.session_state`` and is never passed through
   ``st.cache_resource`` (which does not copy its return value - a cached mutable
   object would corrupt across sessions) nor ``st.cache_data`` (which would copy
   it on every read, silently discarding pick history).

Run with ``streamlit run draft_app.py`` from the repo root (see that file for why
the entry point lives there rather than inside ``src/``).
"""

from __future__ import annotations

import json
import time
import types

import pandas as pd
import streamlit as st

from src import columns as C
from src import config
from src import draft as draft_mod
from src import lineup as lineup_mod
from src import live_draft as live_mod
from src import positions as pos_mod
from src import projection as proj_mod
from src import reconcile

# Fixed, known slot/stat vocabularies used to render the setup-screen configuration
# widgets. Kept separate from config.py's dicts (which the user overrides) so the
# UI always offers the full standard set even if config.py's defaults omit one.
_ROSTER_SLOT_NAMES: tuple[str, ...] = ("PG", "SG", "SF", "PF", "C", "G", "F", "UTIL", "BN")
_SCORABLE_STATS: tuple[tuple[str, str], ...] = (
    (C.COL_PTS, "PTS"), (C.COL_REB, "REB"), (C.COL_AST, "AST"), (C.COL_STL, "STL"),
    (C.COL_BLK, "BLK"), (C.COL_TOV, "TOV"), (C.COL_FGM, "FGM"), (C.COL_FGA, "FGA"),
    (C.COL_FTM, "FTM"), (C.COL_FTA, "FTA"), (C.COL_FG3M, "FG3M"), (C.COL_FG3A, "FG3A"),
)

# ===========================================================================
# Cached, slot-independent pipeline (Phase 2 - the caching spine).
# ===========================================================================

@st.cache_data(show_spinner="Building the board (ingest -> scoring -> projection -> VOR board) ...")
def _cached_board_bundle(
    scoring_key: tuple, roster_slots_key: tuple, num_teams: int, target_season: str,
    seasons_key: tuple, use_aging: bool, use_shrinkage: bool, _league,
) -> tuple[list[int], dict[int, float], dict[int, str], dict[int, str]]:
    """Cached wrapper around :func:`src.live_draft.build_board_bundle`.

    ``st.cache_data`` copies its return value on every read, so this is safe to
    call from anywhere in the script without risking a shared-mutable-state bug -
    unlike ``st.cache_resource``, which would hand back the SAME object every
    time. ``_league`` is prefixed with an underscore so Streamlit does not try to
    hash the (mutable-ish) namespace itself; everything that actually affects the
    pipeline's output is passed as an explicit hashable key argument instead, so
    changing a league setting on the setup screen correctly busts the cache
    rather than silently reusing a board built under the old settings.
    """
    return live_mod.build_board_bundle(_league)


def _league_cache_key(league) -> tuple:
    """The hashable subset of ``league`` that actually changes the board (see
    :func:`_cached_board_bundle`'s docstring for why bare `league` isn't enough)."""
    return (
        tuple(sorted(league.SCORING.items())),
        tuple(sorted(league.ROSTER_SLOTS.items())),
        league.NUM_TEAMS, league.TARGET_SEASON, tuple(league.SEASONS),
        league.V4_USE_AGING, league.V4_USE_SHRINKAGE,
    )


def _default_league() -> types.SimpleNamespace:
    """Every ALL_CAPS setting in config.py, copied into a mutable namespace.

    Everything not overridden on the setup screen (Yahoo credentials, the
    backtest-only knobs, the aging-curve internals) keeps its config.py value -
    the setup screen only overrides the settings that are both league-specific
    and actually threaded through ``league`` at the call sites that matter
    (confirmed by reading src/projection.py: project_v4 reads N_PROJECTION_SEASONS
    and GP_RECENCY_DECAY from config.py directly regardless of what `league` is
    passed to it, so those two are deliberately NOT exposed here - a control that
    silently does nothing would be worse than no control).
    """
    return types.SimpleNamespace(**{name: getattr(config, name) for name in dir(config)
                                    if name.isupper()})


@st.cache_data(show_spinner=False)
def _cached_recommend(
    _state: live_mod.LiveDraftState, pick_log_key: tuple, top_n: int, weight: float,
) -> list[dict]:
    """Cached wrapper around :meth:`LiveDraftState.recommend`.

    Keyed on the full pick log rather than bare ``pick_index``: an undo followed
    by a DIFFERENT pick can return to the same pick_index with a different roster
    and a different remaining board, and ``recommend()`` depends on both - keying
    on pick_index alone would silently serve a stale recommendation in that case.
    The pick log fully determines the roster and the available board, so it is
    the correct cache key (its length already encodes pick_index). ``_state`` is
    prefixed with an underscore so Streamlit does not try to hash the mutable
    object itself - only the declared key arguments matter for cache identity.
    """
    return _state.recommend(top_n=top_n, run_risk_weight=weight)


def _start_session(slot: int, league, resume_saved: dict | None) -> None:
    """Build (or resume) a draft into ``st.session_state``. Called only from a button."""
    if resume_saved is None and live_mod.STATE_PATH.exists():
        # Starting fresh must not silently destroy an un-resumed session - the
        # first recorded pick would otherwise overwrite the file unconditionally.
        backup_path = live_mod.backup_state_file()
        if backup_path:
            st.session_state.setup_message = f"Previous progress backed up to {backup_path}."

    t0 = time.time()
    board_ids, value_by_id, position_by_id, name_by_id = _cached_board_bundle(
        *_league_cache_key(league), league)
    build_seconds = time.time() - t0

    my_slot = resume_saved["my_slot"] if resume_saved else slot
    order = draft_mod.snake_pick_order(league.NUM_TEAMS, league.ROSTER_SIZE)
    state = live_mod.LiveDraftState(
        order=order, my_slot=my_slot, board_ids=board_ids, value_by_id=value_by_id,
        position_by_id=position_by_id, name_by_id=name_by_id, league=league,
    )
    if resume_saved:
        state.load_picks(resume_saved)

    st.session_state.draft_state = state
    st.session_state.last_recorded_pick_index = -1
    st.session_state.build_seconds = round(build_seconds, 2)
    st.session_state.pick_error = None


# ===========================================================================
# Setup screen (no draft in progress yet).
# ===========================================================================

def _render_setup_screen() -> None:
    st.subheader("Start a draft")
    if msg := st.session_state.pop("setup_message", None):
        st.info(msg)

    saved = None
    if live_mod.STATE_PATH.exists():
        saved = json.loads(live_mod.STATE_PATH.read_text(encoding="utf-8"))
        st.info(f"Saved progress found: {len(saved['pick_log'])} picks recorded, "
                f"slot {saved['my_slot']}. Resuming reuses config.py's league "
                f"settings as they were - the settings below apply only to a "
                f"fresh start.")
        if st.button("Resume saved draft", type="primary"):
            _start_session(saved["my_slot"], _default_league(), resume_saved=saved)
            st.rerun()
        st.divider()
        st.caption("Or configure and start a fresh draft below "
                   "(backs up the saved one first):")

    st.warning(
        "config.py ships with illustrative PLACEHOLDER league settings, not your "
        "real league. Fill in the values below from Yahoo -> League Settings "
        "before drafting live - see the warning at the top of src/config.py.",
        icon="⚠️",
    )

    with st.form("setup_form"):
        st.markdown("**League shape**")
        col_slot, col_teams, col_season = st.columns(3)
        slot = col_slot.number_input(
            "Your draft position", min_value=1, max_value=24,
            value=config.MY_DRAFT_SLOT, step=1,
            help="Assumes a snake draft; must not exceed the team count below.")
        num_teams = col_teams.number_input(
            "Number of teams", min_value=2, max_value=24, value=config.NUM_TEAMS, step=1)
        # Completed seasons already in config.SEASONS (for a backtest-style sanity
        # check against real results) PLUS the season right after the most recent
        # one - the actual upcoming draft, which has no results yet. The leakage
        # guard in src/projection.py only checks that no data FROM OR AFTER the
        # target was used, which trivially holds for a season not yet played.
        upcoming_season = proj_mod.next_season(config.SEASONS[-1])
        target_options = list(config.SEASONS[1:] or config.SEASONS) + [upcoming_season]
        target_season = col_season.selectbox(
            "Target season (drafting for)", options=target_options,
            index=len(target_options) - 1,  # default to the upcoming season - the live-draft case
            help=f"'{upcoming_season}' is the upcoming season (no actual results yet - "
                 f"the real live-draft case). Earlier seasons let you sanity-check "
                 f"against real results instead.")
        if target_season not in config.SEASONS:
            st.caption(
                f"⚠️ {target_season} has not been played - no rookies or players with "
                f"no prior NBA history will appear on the board (no signal to project "
                f"from). This is a known limitation, not new to this season choice.")

        st.markdown("**Roster slots** (per team - starting + bench composition)")
        roster_cols = st.columns(len(_ROSTER_SLOT_NAMES))
        roster_slots: dict[str, int] = {}
        for col, slot_name in zip(roster_cols, _ROSTER_SLOT_NAMES):
            roster_slots[slot_name] = col.number_input(
                slot_name, min_value=0, max_value=10,
                value=config.ROSTER_SLOTS.get(slot_name, 0), step=1,
                key=f"setup_slot_{slot_name}")

        st.markdown("**Scoring** (points per stat - Yahoo: League Settings -> Scoring)")
        scoring: dict[str, float] = {}
        scoring_cols = st.columns(4)
        for i, (stat_col, label) in enumerate(_SCORABLE_STATS):
            scoring[stat_col] = scoring_cols[i % 4].number_input(
                label, value=float(config.SCORING.get(stat_col, 0.0)), step=0.25,
                format="%.2f", key=f"setup_score_{stat_col}")

        with st.expander("Advanced projection settings"):
            use_aging = st.checkbox(
                "Use the aging-curve adjustment (V4)", value=config.V4_USE_AGING,
                help="The V5 backtest found this improves accuracy - on by default.")
            use_shrinkage = st.checkbox(
                "Use positional shrinkage (V4)", value=config.V4_USE_SHRINKAGE,
                help="The V5 backtest found this does NOT help on historical data "
                     "- off by default for a reason; flip on to re-measure.")

        submitted = st.form_submit_button("Build board and start", type="primary")

    if not submitted:
        return

    league = _default_league()
    league.NUM_TEAMS = int(num_teams)
    league.ROSTER_SLOTS = {name: int(count) for name, count in roster_slots.items() if count > 0}
    league.ROSTER_SIZE = sum(league.ROSTER_SLOTS.values())
    league.TARGET_SEASON = target_season
    league.SCORING = scoring
    league.V4_USE_AGING = use_aging
    league.V4_USE_SHRINKAGE = use_shrinkage

    if league.ROSTER_SIZE == 0:
        st.error("Roster slots must add up to at least one draftable slot.")
    elif int(slot) > league.NUM_TEAMS:
        st.error(f"Draft position {int(slot)} exceeds {league.NUM_TEAMS} teams.")
    else:
        _start_session(int(slot), league, resume_saved=None)
        st.rerun()


# ===========================================================================
# Callbacks. Every draft-state mutation happens in one of these, never inline
# in the script body - inline mutation plus Streamlit's rerun ordering would
# double-record a pick on the same interaction.
# ===========================================================================

def _apply_pick(pid: int, team_slot: int | None) -> None:
    """Shared mutation path for both callbacks below, with the double-submit guard."""
    state: live_mod.LiveDraftState = st.session_state.draft_state
    target_pick_index = state.pick_index  # the pick slot this submission is FOR
    if target_pick_index == st.session_state.get("last_recorded_pick_index", -1):
        return  # this pick slot was already filled by an earlier firing - ignore
    try:
        state.record_pick(pid, team_slot=team_slot)
        state.save()
        st.session_state.last_recorded_pick_index = target_pick_index
        st.session_state.pick_error = None
    except ValueError as error:
        st.session_state.pick_error = str(error)


def _opponent_pick_callback() -> None:
    label = st.session_state.get("pick_selectbox")
    if not label:
        return
    pid = st.session_state.get("label_to_id", {}).get(label)
    if pid is None:
        st.session_state.pick_error = f"Unrecognised selection: {label!r}"
    else:
        _apply_pick(pid, team_slot=None)  # None = whoever is on the clock
    st.session_state.pick_selectbox = None  # clear the box for the next entry


def _recommend_click_callback() -> None:
    state: live_mod.LiveDraftState = st.session_state.draft_state
    selection = st.session_state.get("rec_table")
    rows = selection.selection.rows if selection is not None and selection.selection else []
    if not rows:
        return
    row_ids: list[int] = st.session_state.get("rec_row_ids", [])
    if rows[0] >= len(row_ids):
        return
    # Recommendations are always mine - draft explicitly for my slot regardless
    # of who is nominally on the clock, so this control is only ever wired up
    # while it actually is my turn (see _render_recommendations).
    _apply_pick(row_ids[rows[0]], team_slot=state.my_slot)


def _undo_callback() -> None:
    state: live_mod.LiveDraftState = st.session_state.draft_state
    undone = state.undo_last_pick()
    if undone:
        state.save()  # re-save immediately so an undone pick does not survive on disk
        st.session_state.pick_error = None
        st.session_state.undo_message = f"Undid pick {undone}."


# ===========================================================================
# In-draft panes.
# ===========================================================================

def _render_sidebar(state: live_mod.LiveDraftState) -> None:
    with st.sidebar:
        st.header("Draft controls")
        st.caption(f"You are drafting from slot {state.my_slot} of {state.league.NUM_TEAMS}. "
                   f"Board built in {st.session_state.get('build_seconds', '?')}s "
                   f"({len(state.board_ids)} players).")
        st.session_state.setdefault("run_risk_weight", float(state.league.RUN_RISK_WEIGHT))
        st.session_state.run_risk_weight = st.slider(
            "Run-risk weight", 0.0, 3.0, st.session_state.run_risk_weight, 0.1,
            help="Weight on positional-run risk vs marginal lineup value "
                 "(src.draft.rank_candidates). 0 = pure marginal value.",
        )
        st.session_state.setdefault("top_n", 10)
        st.session_state.top_n = st.slider("Recommendations shown", 5, 25, st.session_state.top_n, 1)

        if st.button("Save now"):
            state.save()
            st.success("Saved.")
        st.button("Undo last pick", on_click=_undo_callback)
        if msg := st.session_state.pop("undo_message", None):
            st.info(msg)

        st.divider()
        if st.button("End session (saved progress is kept on disk)"):
            del st.session_state["draft_state"]
            st.rerun()


def _render_status_strip(state: live_mod.LiveDraftState) -> None:
    dedicated, n_flex = lineup_mod.starting_requirements(state.league)
    filled = {position: 0 for position in dedicated}
    for pid in state.my_roster():
        position = state.position_by_id.get(pid, pos_mod.UNKNOWN_POSITION)
        if position in filled:
            filled[position] += 1
    unfilled = {position: max(0, need - filled[position]) for position, need in dedicated.items()}
    unfilled_label = ", ".join(f"{position} x{count}" for position, count in unfilled.items() if count) or "none"

    round_no = state.pick_index // state.league.NUM_TEAMS + 1
    on_clock = None if state.is_complete() else state.on_the_clock()

    cols = st.columns(5)
    cols[0].metric("Overall pick", f"{min(state.pick_index + 1, len(state.order))}/{len(state.order)}")
    cols[1].metric("Round", round_no)
    cols[2].metric("On the clock", f"Team {on_clock}" if on_clock else "complete")
    cols[3].metric("Picks until your turn", state.gap_to_my_next_pick())
    cols[4].metric("Unfilled starters", f"{unfilled_label} (+{n_flex} flex)")


def _render_opponent_fast_path(state: live_mod.LiveDraftState) -> None:
    on_clock = state.on_the_clock()
    available = state.available_ids()
    labels = [state.player_label(pid) for pid in available]
    st.session_state.label_to_id = dict(zip(labels, available))
    st.selectbox(
        f"Record the pick for team {on_clock} (type to search)",
        options=labels, index=None, placeholder="Start typing a player's name...",
        key="pick_selectbox", on_change=_opponent_pick_callback,
    )
    if error := st.session_state.get("pick_error"):
        st.error(error)


def _render_recommendations(state: live_mod.LiveDraftState) -> None:
    rows = _cached_recommend(state, tuple(state.pick_log), st.session_state.top_n,
                             st.session_state.run_risk_weight)
    st.session_state.rec_row_ids = [row["player_id"] for row in rows]
    if not rows:
        st.caption("No available player currently improves your lineup.")
        return

    df = pd.DataFrame(rows).rename(columns={
        "name": "Player", "position": "Pos", "value": "Value",
        "marginal_value": "Marginal", "run_risk": "Run risk", "score": "Score",
    })
    my_turn = not state.is_complete() and state.on_the_clock() == state.my_slot
    if my_turn:
        st.caption("Click a row to draft that player for your team.")
        st.dataframe(df, hide_index=True, key="rec_table",
                     on_select=_recommend_click_callback, selection_mode="single-row",
                     column_order=["Player", "Pos", "Value", "Marginal", "Run risk", "Score"])
    else:
        st.caption("Read-only until you are on the clock.")
        st.dataframe(df, hide_index=True, key="rec_table",
                     column_order=["Player", "Pos", "Value", "Marginal", "Run risk", "Score"])


def _render_board(state: live_mod.LiveDraftState) -> None:
    position_filter = st.radio(
        "Position", ["ALL", *pos_mod.COARSE_POSITIONS], horizontal=True, key="board_pos_filter")
    available = state.available_ids()
    if position_filter != "ALL":
        available = [pid for pid in available if state.position_by_id.get(pid) == position_filter]
    rows = [
        {"Player": state.name_by_id.get(pid, ""), "Pos": state.position_by_id.get(pid, ""),
         "Value": round(state.value_by_id.get(pid, 0.0), 1)}
        for pid in available[:50]
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, key="board_table")


def _render_search(state: live_mod.LiveDraftState) -> None:
    """Search ALL players (available or drafted), unlike LiveDraftState.find_player
    (available only) - a taken player should read as TAKEN, not as a typo."""
    query = st.text_input("Search any player (available or drafted)", key="search_query")
    if not query.strip():
        return
    needle = reconcile.normalise_name(query)
    matches = [pid for pid in state.board_ids
              if needle in reconcile.normalise_name(state.name_by_id.get(pid, ""))]
    if not matches:
        st.caption("No matches.")
        return
    pick_no_by_id = {pid: pick_no for pick_no, _team, pid in state.pick_log}
    rows = []
    for pid in matches[:25]:
        if pid in state.drafted:
            status = f"TAKEN - team {state.drafted[pid]}, pick {pick_no_by_id.get(pid, '?')}"
        else:
            status = "available"
        rows.append({"Player": state.player_label(pid), "Status": status})
    st.dataframe(pd.DataFrame(rows), hide_index=True, key="search_table")


def _render_roster_and_lineup(state: live_mod.LiveDraftState) -> None:
    st.subheader("Your roster")
    roster = state.my_roster()
    if not roster:
        st.caption("No picks yet.")
    else:
        st.dataframe(pd.DataFrame({"Player": [state.player_label(pid) for pid in roster]}),
                     hide_index=True, key="my_roster_table")

    starters, value = state.my_lineup()
    st.metric("Best legal starting lineup", f"{value:,.1f} pts")
    if starters:
        st.dataframe(pd.DataFrame({"Starter": [state.player_label(pid) for pid in starters]}),
                     hide_index=True, key="my_lineup_table")


def _render_recent_picks(state: live_mod.LiveDraftState) -> None:
    st.subheader("Recent picks")
    recent = list(reversed(state.pick_log[-8:]))
    if not recent:
        st.caption("No picks yet.")
        return
    rows = [{"Pick": pick_no, "Team": team, "Player": state.player_label(pid)}
            for pick_no, team, pid in recent]
    st.dataframe(pd.DataFrame(rows), hide_index=True, key="recent_picks_table")


def _render_complete(state: live_mod.LiveDraftState) -> None:
    st.success("Draft complete.")
    _render_roster_and_lineup(state)


# ===========================================================================
# Entry point.
# ===========================================================================

def main() -> None:
    st.set_page_config(page_title="NBA Draft Assistant", layout="wide")
    st.title("NBA Fantasy Draft Optimiser - live draft assistant")

    if "draft_state" not in st.session_state:
        _render_setup_screen()
        return

    state: live_mod.LiveDraftState = st.session_state.draft_state
    _render_sidebar(state)

    if state.is_complete():
        _render_status_strip(state)
        _render_complete(state)
        return

    _render_status_strip(state)
    _render_opponent_fast_path(state)

    left, right = st.columns([2, 1])
    with left:
        tab_rec, tab_board, tab_search = st.tabs(["Recommendations", "Board", "Search"])
        with tab_rec:
            _render_recommendations(state)
        with tab_board:
            _render_board(state)
        with tab_search:
            _render_search(state)
    with right:
        _render_roster_and_lineup(state)
        _render_recent_picks(state)


if __name__ == "__main__":
    main()
