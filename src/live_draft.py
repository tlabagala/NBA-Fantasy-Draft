"""V6: an interactive draft assistant usable during a REAL live draft.

Everything before this module is offline: it builds a board and then simulates
every team's pick under a fixed strategy. That is enough to validate the engine
(V1-V5), but it cannot sit next to you during an actual draft, where:

- your draft position is whatever you were assigned, not ``config.MY_DRAFT_SLOT``;
- every OTHER team's pick is a human decision, not a greedy bot; and
- you need a recommendation for THIS pick, given exactly what has already been
  taken - not a full pre-simulated roster.

:class:`LiveDraftState` tracks a real draft pick by pick. Each pick - mine or an
opponent's - is recorded as it happens; the board shrinks accordingly, and my next
recommendation (:meth:`LiveDraftState.recommend`) is re-computed from
:func:`src.draft.rank_candidates` - the same marginal-lineup-value + positional-
run-risk logic the V3 backtest validated, now driven by the true remaining board
and the true gap to my next pick, not a simulated opponent.

Run ``python -m src.live_draft`` for the interactive CLI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src import board as board_mod
from src import columns as C
from src import config
from src import draft as draft_mod
from src import ingest, positions, projection, scoring
from src import lineup as lineup_mod
from src import positions as pos_mod
from src import reconcile

STATE_PATH = Path(__file__).resolve().parent.parent / "outputs" / "live_draft_state.json"


@dataclass
class LiveDraftState:
    """Mutable state for one real, in-progress draft.

    Args:
        order: the full pick order (team slot per overall pick) from
            :func:`src.draft.snake_pick_order`.
        my_slot: my 1-based draft position.
        board_ids: every draftable player id, best-first (the reference ranking;
            drafted players are filtered out on read, never removed from this list,
            so :meth:`gap_to_my_next_pick`'s "best available" reasoning stays valid).
        value_by_id, position_by_id, name_by_id: per-player lookups.
        league: the :mod:`src.config` module (or a compatible namespace).
        drafted: ``{player_id: team_slot}`` for every pick made so far.
        pick_log: ``[(overall_pick_no, team_slot, player_id), ...]`` in order, for
            undo and for resuming a saved session.
    """

    order: list[int]
    my_slot: int
    board_ids: list[int]
    value_by_id: dict[int, float]
    position_by_id: dict[int, str]
    name_by_id: dict[int, str]
    league: object
    drafted: dict[int, int] = field(default_factory=dict)
    pick_log: list[tuple[int, int, int]] = field(default_factory=list)

    # --- draft-clock bookkeeping -------------------------------------------------

    @property
    def pick_index(self) -> int:
        """0-based index of the NEXT pick to be made (== picks made so far)."""
        return len(self.pick_log)

    def is_complete(self) -> bool:
        return self.pick_index >= len(self.order)

    def on_the_clock(self) -> int:
        """Team slot whose turn it is right now."""
        if self.is_complete():
            raise ValueError("Draft is complete - no team is on the clock.")
        return self.order[self.pick_index]

    def gap_to_my_next_pick(self) -> int:
        """How many other teams pick between now and my next turn (0 if I'm up)."""
        for j in range(self.pick_index, len(self.order)):
            if self.order[j] == self.my_slot:
                return j - self.pick_index
        return len(self.order)  # no more picks left for me

    # --- board / roster reads ----------------------------------------------------

    def available_ids(self) -> list[int]:
        """Undrafted player ids, still board-ordered (best first)."""
        return [pid for pid in self.board_ids if pid not in self.drafted]

    def roster_of(self, team_slot: int) -> list[int]:
        """Player ids drafted by ``team_slot``, in the order they were picked."""
        return [pid for _pick_no, team, pid in self.pick_log if team == team_slot]

    def my_roster(self) -> list[int]:
        return self.roster_of(self.my_slot)

    def my_lineup(self) -> tuple[list[int], float]:
        """My best legal starting lineup right now (see :mod:`src.lineup`)."""
        return lineup_mod.optimal_lineup(self.my_roster(), self.value_by_id,
                                         self.position_by_id, self.league)

    def find_player(self, query: str) -> list[int]:
        """Available player ids whose name contains ``query`` (case/accent-insensitive)."""
        needle = reconcile.normalise_name(query)
        if not needle:
            return []
        return [pid for pid in self.available_ids()
                if needle in reconcile.normalise_name(self.name_by_id.get(pid, ""))]

    def player_label(self, pid: int) -> str:
        name = self.name_by_id.get(pid, f"<unknown id {pid}>")
        pos = self.position_by_id.get(pid, pos_mod.UNKNOWN_POSITION)
        value = self.value_by_id.get(pid, 0.0)
        return f"{name} ({pos}, {value:,.0f} pts)"

    # --- mutation ------------------------------------------------------------

    def record_pick(self, player_id: int, team_slot: int | None = None) -> int:
        """Record a real pick. Defaults ``team_slot`` to whoever is on the clock.

        Returns:
            The team slot the pick was recorded against.

        Raises:
            ValueError: draft already complete, player already drafted, unknown
                player id, or an out-of-range ``team_slot``.
        """
        if self.is_complete():
            raise ValueError("Draft is already complete.")
        if player_id in self.drafted:
            raise ValueError(
                f"{self.player_label(player_id)} was already drafted by team "
                f"{self.drafted[player_id]}.")
        if player_id not in self.value_by_id:
            raise ValueError(f"Unknown player id {player_id}.")
        team = self.on_the_clock() if team_slot is None else team_slot
        if team < 1 or team > self.league.NUM_TEAMS:
            raise ValueError(f"team_slot {team} must be within 1..{self.league.NUM_TEAMS}.")

        self.drafted[player_id] = team
        self.pick_log.append((self.pick_index + 1, team, player_id))
        return team

    def undo_last_pick(self) -> tuple[int, int, int] | None:
        """Undo the most recent pick; returns ``(pick_no, team_slot, player_id)`` or None."""
        if not self.pick_log:
            return None
        last = self.pick_log.pop()
        del self.drafted[last[2]]
        return last

    # --- recommendation --------------------------------------------------------

    def recommend(self, top_n: int = 10,
                  run_risk_weight: float | None = None) -> list[dict]:
        """Rank the best available players for MY next pick right now.

        Uses :func:`src.draft.rank_candidates` (the V3 marginal-lineup-value +
        positional-run-risk rule) against the true remaining board and the true
        number of opponent picks before my next turn - not a simulation.
        """
        available = self.available_ids()
        gap = self.gap_to_my_next_pick()
        ranked = draft_mod.rank_candidates(
            available, self.my_roster(), self.value_by_id, self.position_by_id,
            self.league, gap, run_risk_weight,
        )
        return [
            {
                "player_id": pid,
                "name": self.name_by_id.get(pid, f"<unknown id {pid}>"),
                "position": self.position_by_id.get(pid, pos_mod.UNKNOWN_POSITION),
                "value": round(self.value_by_id.get(pid, 0.0), 1),
                "marginal_value": round(marginal, 1),
                "run_risk": round(run, 1),
                "score": round(score, 1),
            }
            for pid, marginal, run, score in ranked[:top_n]
        ]

    # --- persistence (so a real draft survives a closed terminal) --------------

    def to_json(self) -> dict:
        return {
            "n_teams": self.league.NUM_TEAMS,
            "roster_size": self.league.ROSTER_SIZE,
            "my_slot": self.my_slot,
            "pick_log": self.pick_log,
        }

    def save(self, path: Path | None = None) -> Path:
        # Resolved at call time, not bind time, so reassigning the module-level
        # STATE_PATH (e.g. in tests) actually takes effect.
        path = path if path is not None else STATE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")
        return path

    def load_picks(self, saved: dict) -> None:
        """Replay a saved ``pick_log`` on top of a freshly built board.

        Board contents (projections) can legitimately change between sessions;
        only the picks themselves are restored, in order, so state re-derives
        cleanly rather than trusting a stale snapshot.
        """
        for _pick_no, team, pid in saved["pick_log"]:
            self.record_pick(pid, team_slot=team)


def build_live_draft(my_slot: int | None = None, league=config) -> LiveDraftState:
    """Build a fresh :class:`LiveDraftState` from the same pipeline as ``main.py``.

    Runs ingestion -> scoring -> V4 projection (aging, no shrinkage per the
    measured V4 default) -> coarse positions -> value-over-replacement board, then
    hands back an empty draft ready for :meth:`LiveDraftState.record_pick`.

    Args:
        my_slot: my 1-based draft position; defaults to ``league.MY_DRAFT_SLOT``.
        league: the :mod:`src.config` module (or a compatible namespace/test double).
    """
    slot = league.MY_DRAFT_SLOT if my_slot is None else my_slot
    if slot < 1 or slot > league.NUM_TEAMS:
        raise ValueError(f"my_slot {slot} must be within 1..{league.NUM_TEAMS}.")

    scoring_weights = league.SCORING
    logs = ingest.load_player_logs(league.SEASONS)
    totals = scoring.season_totals(scoring.apply_scoring(logs, scoring_weights))

    window = projection.recent_prior_seasons(totals, league.TARGET_SEASON,
                                             league.N_PROJECTION_SEASONS)
    rosters_by_season = [(season, ingest.load_team_rosters(season)) for season in window]
    position_map = positions.coarse_position_map_over_seasons(rosters_by_season)

    bios = ingest.load_player_bios(league.SEASONS)
    proj = projection.project_v4(
        totals, league.TARGET_SEASON, bios, position_map, league,
        use_aging=league.V4_USE_AGING, use_shrinkage=league.V4_USE_SHRINKAGE,
    )
    board = board_mod.build_board_vor(proj, position_map, league)

    board_ids = board[C.COL_PLAYER_ID].tolist()
    value_by_id = dict(zip(board[C.COL_PLAYER_ID], board[C.COL_PROJECTED_PTS]))
    position_by_id = dict(zip(board[C.COL_PLAYER_ID], board[C.COL_COARSE_POSITION]))
    name_by_id = dict(zip(board[C.COL_PLAYER_ID], board[C.COL_PLAYER_NAME]))

    order = draft_mod.snake_pick_order(league.NUM_TEAMS, league.ROSTER_SIZE)
    return LiveDraftState(
        order=order, my_slot=slot, board_ids=board_ids, value_by_id=value_by_id,
        position_by_id=position_by_id, name_by_id=name_by_id, league=league,
    )


# ===========================================================================
# Interactive CLI - the thing you actually run during a draft.
# ===========================================================================

_HELP = """\
Commands (draft position and picks):
  pick <name or #id> [team]   record a pick (default team = whoever is on the clock)
  me <name or #id>             shortcut for "pick ... <my slot>"
  undo                         undo the most recent pick
  board [n]                    show the top n available players (default 15)
  rec [n]                      show my top n recommended picks right now (default 10)
  roster [team]                show a team's drafted players (default: mine)
  lineup                       show my best legal starting lineup right now
  find <text>                  search available players by (partial) name
  save                         save progress to outputs/live_draft_state.json
  help                         show this message
  quit                         save and exit
"""


def _match_player(state: LiveDraftState, token: str) -> int:
    """Resolve a CLI token to a player id: ``#123`` is an id, else a name search."""
    token = token.strip()
    if token.startswith("#"):
        try:
            pid = int(token[1:])
        except ValueError as error:
            raise ValueError(f"Not a valid id: {token!r}") from error
        if pid not in state.value_by_id:
            raise ValueError(f"Unknown player id {pid}.")
        return pid

    matches = state.find_player(token)
    if not matches:
        raise ValueError(f"No available player matches {token!r}.")
    if len(matches) > 1:
        options = "; ".join(f"#{pid} {state.player_label(pid)}" for pid in matches[:8])
        raise ValueError(
            f"{token!r} is ambiguous ({len(matches)} matches) - use #id. {options}")
    return matches[0]


def _print_recommendations(state: LiveDraftState, rows: list[dict]) -> None:
    if not rows:
        print("  (no available players improve your lineup)")
        return
    for row in rows:
        print(f"  #{row['player_id']:<8} {row['name']:<24} {row['position']:<4} "
              f"value {row['value']:>8,.1f}  marginal {row['marginal_value']:>7,.1f}  "
              f"run-risk {row['run_risk']:>6,.1f}  score {row['score']:>8,.1f}")


def _print_board(state: LiveDraftState, n: int) -> None:
    for pid in state.available_ids()[:n]:
        print(f"  #{pid:<8} {state.player_label(pid)}")


def _print_roster(state: LiveDraftState, team: int) -> None:
    roster = state.roster_of(team)
    label = "you" if team == state.my_slot else f"team {team}"
    if not roster:
        print(f"  {label}: (no picks yet)")
        return
    print(f"  {label}:")
    for pid in roster:
        print(f"    {state.player_label(pid)}")


def run_cli() -> None:
    print("=" * 72)
    print("NBA Fantasy Draft Optimiser - LIVE draft assistant")
    print("=" * 72)

    resume = None
    if STATE_PATH.exists():
        answer = input(f"Found saved progress at {STATE_PATH}. Resume it? [y/N] ").strip().lower()
        if answer == "y":
            resume = json.loads(STATE_PATH.read_text(encoding="utf-8"))

    if resume:
        my_slot = resume["my_slot"]
        print(f"Resuming: {len(resume['pick_log'])} picks already recorded, slot {my_slot}.")
    else:
        raw = input(f"Your draft position (1-{config.NUM_TEAMS}) "
                    f"[default {config.MY_DRAFT_SLOT}]: ").strip()
        my_slot = int(raw) if raw else config.MY_DRAFT_SLOT

    print("Building the board (first run pulls nba_api and caches; later runs are fast)...")
    state = build_live_draft(my_slot=my_slot)
    if resume:
        state.load_picks(resume)

    print(f"Board ready: {len(state.board_ids)} players, {config.NUM_TEAMS} teams, "
          f"you draft from slot {state.my_slot}.")
    print(_HELP)

    while not state.is_complete():
        on_clock = state.on_the_clock()
        marker = "  <-- YOU ARE ON THE CLOCK" if on_clock == state.my_slot else ""
        print(f"\nPick {state.pick_index + 1}/{len(state.order)}: team {on_clock} is up.{marker}")
        if on_clock == state.my_slot:
            print("Top recommendations for you:")
            _print_recommendations(state, state.recommend(top_n=10))

        try:
            raw = input("> ").strip()
        except EOFError:
            break
        if not raw:
            continue
        parts = raw.split()
        cmd, args = parts[0].lower(), parts[1:]

        try:
            if cmd in {"quit", "exit"}:
                state.save()
                print(f"Saved to {STATE_PATH}. Bye.")
                return
            elif cmd == "help":
                print(_HELP)
            elif cmd == "save":
                path = state.save()
                print(f"Saved to {path}.")
            elif cmd == "undo":
                undone = state.undo_last_pick()
                print(f"Undid pick {undone}" if undone else "Nothing to undo.")
            elif cmd == "board":
                n = int(args[0]) if args else 15
                _print_board(state, n)
            elif cmd == "rec":
                n = int(args[0]) if args else 10
                _print_recommendations(state, state.recommend(top_n=n))
            elif cmd == "roster":
                team = int(args[0]) if args else state.my_slot
                _print_roster(state, team)
            elif cmd == "lineup":
                starters, value = state.my_lineup()
                print(f"  best legal lineup ({value:,.1f} projected pts):")
                for pid in starters:
                    print(f"    {state.player_label(pid)}")
            elif cmd == "find":
                if not args:
                    print("  usage: find <text>")
                else:
                    matches = state.find_player(" ".join(args))
                    if not matches:
                        print("  no matches.")
                    for pid in matches[:15]:
                        print(f"  #{pid:<8} {state.player_label(pid)}")
            elif cmd in {"pick", "me"}:
                if not args:
                    print(f"  usage: {cmd} <name or #id> [team]")
                    continue
                team_override = None
                name_tokens = args
                if cmd == "me":
                    team_override = state.my_slot
                elif args[-1].isdigit():
                    team_override = int(args[-1])
                    name_tokens = args[:-1]
                pid = _match_player(state, " ".join(name_tokens))
                team = state.record_pick(pid, team_slot=team_override)
                print(f"  recorded: team {team} took {state.player_label(pid)}")
                state.save()
            else:
                print(f"  unknown command {cmd!r}. Type 'help' for the command list.")
        except ValueError as error:
            print(f"  error: {error}")

    print("\nDraft complete.")
    _print_roster(state, state.my_slot)
    starters, value = state.my_lineup()
    print(f"Your best legal starting lineup projects to {value:,.1f} points.")


if __name__ == "__main__":
    run_cli()
