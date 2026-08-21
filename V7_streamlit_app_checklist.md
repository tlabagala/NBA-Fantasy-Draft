# V7 - Streamlit draft-day UI: build checklist

Companion to `V7_streamlit_app_prompt.md`. Work top to bottom. Each phase ends in a
**gate** - do not start the next phase until the gate passes.

Rough effort: Phases 1 to 4 are the bulk (an afternoon). Phases 5 and 6 are another couple
of hours and they are the ones people skip and later regret.

---

## Phase 0 - before any code

- [ ] Read `nba_draft_optimiser_masterdoc.md` Sections 3, 7 and 9.
- [ ] Read `src/live_draft.py` in full.
- [ ] Confirm `pytest` passes on the current tree. Record the result - this is the baseline
      you must not break.
- [ ] Commit or stash any work in progress, so V7 starts from a clean tree.
- [ ] Time the current CLI: record 10 opponent picks with `python -m src.live_draft` and
      note the seconds per pick. **This is the benchmark V7 must beat.**

**Gate:** tests green, CLI timing written down.

---

## Phase 1 - skeleton

- [ ] Activate the existing venv: `.venv\Scripts\activate`
- [ ] `pip install streamlit`
- [ ] Add to `requirements.txt` under a new `# --- V7 (active) ---` heading, matching the
      file's existing comment style, with a minimum-version pin.
- [ ] Create `src/app_ui.py` - all UI logic lives here, per masterdoc Section 4.
- [ ] Create `draft_app.py` at the repo **root** as a three-line shim that imports and calls
      `src.app_ui`. Root placement is deliberate: `streamlit run` puts the script's own
      directory on `sys.path`, so an app inside `src/` cannot resolve `from src import ...`.
- [ ] Minimum viable page: sidebar slot selector, a build button, and `st.write` of the
      board size plus the cold-build duration.
- [ ] Run it: `streamlit run draft_app.py`

**Gate:** the page loads, imports resolve, the board builds, and you know how long a cold
build takes.

---

## Phase 2 - caching spine

The phase that decides whether the app is usable. Streamlit re-runs the entire script on
every interaction.

- [ ] In `src/live_draft.py`, extract `build_board_bundle(league=config)` containing the
      slot-independent pipeline (ingest → scoring → `project_v4` → `build_board_vor`),
      returning `(board_ids, value_by_id, position_by_id, name_by_id)`.
- [ ] Rewrite `build_live_draft` to call it. **Signature, return type and behaviour must not
      change.**
- [ ] Run `pytest`. `tests/test_live_draft.py` is your proof the refactor was safe.
- [ ] Wrap `build_board_bundle` in `@st.cache_data` in the app layer.
- [ ] Hold `LiveDraftState` in `st.session_state` and nowhere else. It is mutated by
      `record_pick`, so it must never be cached.
- [ ] Do **not** use `st.cache_resource` for anything you mutate - it does not copy its
      return value, so mutations corrupt the cache.
- [ ] Memoise `recommend()` keyed on `state.pick_index`.

**Gate:** clicking around the app never re-triggers the build spinner, and `pytest` is
still green.

---

## Phase 3 - opponent fast path

Build this before the pretty parts. Eleven of every twelve picks are somebody else's.

- [ ] Searchable `st.selectbox` over available player labels (Streamlit filters as you type).
- [ ] `on_change` callback records the pick for whoever is on the clock.
- [ ] All state mutation happens **inside callbacks**, never inline in the script body -
      inline mutation plus rerun ordering produces double-recorded picks.
- [ ] Double-submit guard: store the `pick_index` a submission was recorded against and
      ignore repeats.
- [ ] Surface `ValueError` from `record_pick` as a readable `st.error`, not a traceback.
- [ ] Undo button.

**Gate:** ten consecutive opponent picks recorded without touching the mouse, no
duplicates, no crashes.

---

## Phase 4 - recommendations, board and layout

- [ ] Recommendations table from `pd.DataFrame(state.recommend(n))` - it already returns
      records with the right keys.
- [ ] Click-to-draft via `st.dataframe(..., on_select="rerun", selection_mode="single-row")`,
      reading `event.selection.rows`. **Verify the installed Streamlit supports this**; if
      not, fall back to `st.button` per row for the top 10 to 15 and note it in a comment.
- [ ] Status strip: round and overall pick, team on the clock, picks until your next turn
      via `gap_to_my_next_pick()`, and unfilled starter slots via
      `lineup.starting_requirements()`.
- [ ] Sidebar: draft slot, `RUN_RISK_WEIGHT` slider, top-n, save, reset.
- [ ] Board tab with a position filter (G / F / C).
- [ ] Right pane: your roster, best legal lineup with its projected total from
      `my_lineup()`, and the recent pick log.
- [ ] Extend search to show drafted players with their status (`TAKEN - team 4, pick 22`)
      rather than returning nothing. `find_player` currently filters to available only, so
      a taken player reads like a typo.

**Gate:** a full round of picks - yours and opponents' - recorded entirely through the UI.

---

## Phase 5 - persistence and crash safety

`st.session_state` survives reruns but **not** a browser refresh or a server restart.

- [ ] Autosave via `state.save()` on every recorded pick, exactly as the CLI does.
- [ ] Explicit "resume saved draft" path on launch, using `load_picks()`.
- [ ] **Bug fix:** `undo` does not re-save. `record_pick` calls `state.save()` but the
      `undo` branch of `run_cli` does not, so an undone pick survives in the saved JSON.
      Fix it in both the CLI and the app.
- [ ] **Bug fix:** declining the resume prompt silently destroys the old session, because
      the first new pick overwrites `outputs/live_draft_state.json`. Back it up to a
      timestamped copy before starting fresh.

**Gate:** kill the server mid-draft, restart, resume - every pick intact.

---

## Phase 6 - rehearsal and verification

- [ ] `pytest` passes with **no test modified** to accommodate the UI.
- [ ] `python -m src.live_draft` still runs and behaves as it did before V7.
- [ ] Script a full 156-pick draft through the app's own callback functions; confirm it
      completes without error and the final state is complete.
- [ ] Timed rehearsal: record an opponent pick in **under five seconds**.
- [ ] Compare against the Phase 0 CLI timing on the same set of picks. Write both numbers
      down.
- [ ] Sanity-check the recommendations against the CLI for an identical pick log - the same
      board and pick history must produce the same top-10. If they differ, the UI has
      changed engine behaviour and something is wrong.
- [ ] Add a V7 section to `README.md`: how to run it, what it does not do, and the
      limitation that it is a local single-user app with no opponent model beyond the V3
      heuristic already documented.

**Gate:** V7 measurably beats the CLI on seconds per pick. If it does not, say so plainly
and keep the CLI as the primary interface - that is "keep only what helps" applied to V7.

---

## Phase 7 - draft-day hardening

- [ ] Warm the board before the draft starts, so the cold build is not running while you
      are on the clock.
- [ ] `pip freeze > requirements.txt` to lock exact versions, as the file's own header
      instructs.
- [ ] Confirm the CLI fallback runs from a second terminal.
- [ ] Check the real league settings in `src/config.py` - `SCORING`, `NUM_TEAMS`,
      `ROSTER_SLOTS` and `MY_DRAFT_SLOT` are all flagged as placeholders in that file's own
      header. Do not draft live off illustrative defaults.
- [ ] Freeze the code. No changes on draft day.

**Gate:** you can go from cold laptop to a working assistant in under a minute.
