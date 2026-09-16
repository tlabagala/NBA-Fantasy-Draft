"""V1 configuration: scoring, league shape, and which seasons to use.

============================  ACTION REQUIRED  ============================
Four blocks below are PLACEHOLDERS. They are filled with runnable, illustrative
"standard Yahoo-style" defaults so the pipeline runs out of the box, but they are
NOT your league's real settings. Replace them with the values from your Yahoo
league before trusting any output:

  1. SCORING        - the point value of each stat (League Settings -> Scoring)
  2. NUM_TEAMS      - number of teams in your league
  3. ROSTER_SLOTS   - your roster's slot composition (PG, SG, ... BN, IL)
  4. MY_DRAFT_SLOT  - your pick position in round 1 (1 = first overall)

Until you confirm these, treat the backtest number as a mechanics demo, not a
read on your actual league. (In V5 these come live from the Yahoo API and the
placeholders disappear.)
==========================================================================
"""

from __future__ import annotations

import os

from src import columns as C

# Load a git-ignored .env if present so Yahoo credentials never live in the repo
# (master spec Section 3, rule 6). Safe no-op when python-dotenv or .env is absent.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv not installed yet (pre-V5)
    pass

# --------------------------------------------------------------------------
# 1. SCORING  --  PLACEHOLDER: replace with your league's exact point values.
# --------------------------------------------------------------------------
# Maps an nba_api stat column to its fantasy point weight. Keys MUST be real
# nba_api columns (validated at runtime against columns.SCORABLE_STAT_COLUMNS);
# a typo raises rather than silently scoring zero.
#
# The defaults below are a coherent, commonly-seen simple points formula. They do
# NOT double-count (we score total PTS, not each made shot). Your league almost
# certainly differs - e.g. many Yahoo leagues score made FG/FT/3PT individually
# and apply turnover/foul penalties. Overwrite this whole dict.
SCORING: dict[str, float] = {
    C.COL_FGM: 1.0,     # made field goal
    C.COL_FGA: -0.25,   # field goal attempt (penalty)
    C.COL_FTM: 1.0,     # made free throw
    C.COL_FTA: -0.25,   # free throw attempt (penalty)
    C.COL_FG3M: 1.5,    # made three-pointer (bonus on top of PTS)
    C.COL_FG3A: -0.25,  # three-point attempt (penalty)
    C.COL_PTS: 1.0,     # points
    C.COL_REB: 1.25,    # total rebounds
    C.COL_AST: 1.5,     # assists
    C.COL_STL: 2.0,     # steals
    C.COL_BLK: 1.5,     # blocks
    C.COL_TOV: -1.0,    # turnovers (penalty)
    # TODO: Double-double (DD, 2.5) and triple-double (TD, 3.0) are part of the
    # Yahoo formula but are DERIVED, not raw nba_api columns. They are omitted
    # here because _validate_scoring_keys would reject them. To add them:
    #   1. register COL_DD / COL_TD in columns.py (+ allow as scorable),
    #   2. compute the 0/1 flags before apply_scoring (add_double_triple_doubles),
    #   3. add C.COL_DD: 2.5 and C.COL_TD: 3.0 here.
}

# --------------------------------------------------------------------------
# 2-4. LEAGUE shape  --  PLACEHOLDERS where marked.
# --------------------------------------------------------------------------
# NUM_TEAMS     PLACEHOLDER - your league's team count (Yahoo default is 12).
# ROSTER_SLOTS  PLACEHOLDER - your starting + bench composition. V1 drafts a flat
#               roster of size sum(ROSTER_SLOTS.values()); positional eligibility
#               and value-over-replacement arrive in V2. Slot names are free-form
#               here because V1 does not enforce positions - only the TOTAL count
#               matters in V1.
# MY_DRAFT_SLOT PLACEHOLDER - your first-round pick number, 1..NUM_TEAMS.
NUM_TEAMS: int = 12

ROSTER_SLOTS: dict[str, int] = {
    "PG": 1, "SG": 1, "SF": 1, "PF": 1, "C": 1,
    "G": 1, "F": 1, "UTIL": 3,   # 10 starters
    "BN": 3,                       # 3 bench
    # "IL": 0,  # injured-list slots are not drafted; leave out of the draftable count
}

MY_DRAFT_SLOT: int = 1   # 1 = first overall pick

# Total number of players each team drafts in V1 (flat roster, no positions yet).
ROSTER_SIZE: int = sum(ROSTER_SLOTS.values())


# --------------------------------------------------------------------------
# Seasons and backtest target  --  sensible defaults; adjust freely.
# --------------------------------------------------------------------------
# These are NOT league-specific, so they are not placeholders. Each season is one
# cached nba_api call. TARGET_SEASON must be a COMPLETED season (so actual totals
# exist to score against) and must NOT be the earliest season (we need a prior
# season to project from). The projection for TARGET_SEASON uses only seasons
# strictly before it - enforced by the leakage guard in projection.py.
SEASONS: list[str] = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
TARGET_SEASON: str = "2025-26"   # projected from prior seasons; scored on actual 2025-26

# For a REAL live draft of an upcoming season (no actual data yet - e.g. "2026-27"
# while SEASONS ends at "2025-26"), pass that season string directly as TARGET_SEASON
# instead: the leakage guard only checks that no data FROM OR AFTER it was used, which
# trivially holds when the season has not been played yet. src/app_ui.py's setup
# screen offers this automatically via src.projection.next_season(SEASONS[-1]).

# Regular season only (no playoffs) for both projection and scoring.
SEASON_TYPE: str = "Regular Season"


# --------------------------------------------------------------------------
# V2 projection parameters.
# --------------------------------------------------------------------------
# Number of most-recent prior seasons the V2 projection may use (the "2-3 seasons"
# recency window). With fewer prior seasons available it simply uses what it has;
# with a 1-season window V2 collapses toward V1. Adding earlier seasons to SEASONS
# strengthens this window.
N_PROJECTION_SEASONS: int = 3

# Recency decay for the projected games-played estimate: a season N years older
# than the most recent prior season gets weight GP_RECENCY_DECAY ** N. 1.0 = no
# decay (equal-weight mean); lower = trust recent durability more. Note the PPG
# estimate is games-played-weighted (total points / total games), so a season with
# few games already counts less there.
GP_RECENCY_DECAY: float = 0.75

# Maps each ROSTER_SLOTS key to a coarse NBA position {G, F, C} for the V2 value-
# over-replacement baseline. Flexible slots (UTIL, BN, IL) map to None and are
# spread evenly across the three positions. This coarse mapping is the V2 stand-in
# for true Yahoo multi-position eligibility, which arrives in V5.
SLOT_TO_COARSE_POSITION: dict[str, str | None] = {
    "PG": "G", "SG": "G", "G": "G",
    "SF": "F", "PF": "F", "F": "F",
    "C": "C",
    "UTIL": None, "BN": None, "IL": None,
}


# --------------------------------------------------------------------------
# V3 draft engine: lineup scoring and positional-run risk.
# --------------------------------------------------------------------------
# Which roster slots are flexible STARTERS (score every week, any position) versus
# BENCH/injured-list (do not score in the V3 lineup objective). In H2H points only
# your starters score, so the V3 engine and backtest value a roster by its best
# legal starting lineup, not by the sum of all 13 players. Bench value (streaming
# and injury insurance) is real but not modelled here - a documented limitation.
FLEX_SLOT_NAMES: frozenset[str] = frozenset({"UTIL"})
BENCH_SLOT_NAMES: frozenset[str] = frozenset({"BN", "IL"})

# Weight on the positional-run-risk term relative to marginal lineup value. 0 turns
# the run adjustment off (pure marginal value); higher prioritises grabbing
# positions that are about to be depleted before your next pick.
RUN_RISK_WEIGHT: float = 1.0


# --------------------------------------------------------------------------
# V4 sophisticated projection: aging curve and empirical-Bayes shrinkage.
# --------------------------------------------------------------------------
# Shrinkage pseudo-count, in games. A player's per-game rate is regressed toward a
# role (position) prior with weight k/(games + k): a 70-game season is trusted
# heavily, a 10-game season is pulled most of the way to the prior. Larger k =
# more regression. Roughly a third of a season is a sensible default.
SHRINKAGE_K: float = 25.0

# Ages outside this band are clamped onto the estimated aging curve (sparse data at
# the extremes makes raw multipliers unreliable).
AGE_CURVE_MIN: int = 19
AGE_CURVE_MAX: int = 38

# Smoothing half-width (in years) for the aging curve: each age's multiplier is a
# rolling average over +/- this many years, to damp small-sample noise.
AGE_CURVE_SMOOTH: int = 1

# Which V4 components to use by default. The backtest found the aging adjustment
# improves time-ordered accuracy versus V2, but shrinkage toward the positional
# prior did NOT (it drags genuinely-high players toward the mean), so per the spec
# rule "keep only what helps" shrinkage is off by default. Flip it on to re-measure
# on your own seasons.
V4_USE_AGING: bool = True
V4_USE_SHRINKAGE: bool = False


# --------------------------------------------------------------------------
# Determinism, the naive baseline, and the random floor.
# --------------------------------------------------------------------------
# THE BASELINE IS THE NAIVE MODEL: a greedy snake draft off the prior-season-totals
# board ("take whoever scored most last season"). It needs no config of its own -
# it is derived from SEASONS/TARGET_SEASON and the league shape above, and lives in
# src/backtest.py::naive_baseline_roster. Every engine's headline margin is measured
# against it, because that is the bar a competent human sets.
#
# The RANDOM draft is kept only as a sanity floor: it shows the board and pick order
# are wired up, not that the engine is any good (nobody drafts randomly). A single
# random draft is high variance, so the floor averages RANDOM_BASELINE_TRIALS seeded
# trials. All randomness is seeded.
SEED: int = 42
RANDOM_BASELINE_TRIALS: int = 200   # trials for the random FLOOR, not the baseline


# --------------------------------------------------------------------------
# nba_api politeness (rate-limit handling). Tune only if you see throttling.
# --------------------------------------------------------------------------
REQUEST_TIMEOUT_SECONDS: int = 60   # per-call HTTP timeout
SLEEP_BETWEEN_CALLS_SECONDS: float = 0.6  # short delay between live pulls
MAX_RETRIES: int = 4                # retry-with-backoff attempts per call
BACKOFF_BASE_SECONDS: float = 1.5   # exponential backoff base


# --------------------------------------------------------------------------
# V5 Yahoo integration. Credentials come from a git-ignored .env (never hardcode).
# See .env.example for setup. USE_YAHOO turns on automatically once all three are
# present; until then the pipeline runs on the placeholder SCORING above.
# --------------------------------------------------------------------------
YAHOO_LEAGUE_ID: str = os.getenv("YAHOO_LEAGUE_ID", "")
YAHOO_GAME_CODE: str = os.getenv("YAHOO_GAME_CODE", "nba")
YAHOO_CONSUMER_KEY: str = os.getenv("YAHOO_CONSUMER_KEY", "")
YAHOO_CONSUMER_SECRET: str = os.getenv("YAHOO_CONSUMER_SECRET", "")
USE_YAHOO: bool = bool(YAHOO_LEAGUE_ID and YAHOO_CONSUMER_KEY and YAHOO_CONSUMER_SECRET)


# --------------------------------------------------------------------------
# V5 multi-season distribution backtest. A single season is high variance, so the
# trustworthy verdict runs across several target seasons and reports a distribution.
#
# History runs from 2010-11. A target needs N_PROJECTION_SEASONS (3) prior seasons,
# so the first usable target is 2013-14, giving 13 target seasons. Why 2010-11 and
# not earlier: the aging curve is the only component that reads deep history, and
# its estimates converge with diminishing returns - measured against a 29-season
# curve, mean absolute deviation over ages 22-33 falls 0.0181 (3 seasons) -> 0.0088
# (10) -> 0.0067 (15) -> 0.0020 (25). Fifteen seasons captures most of the
# available gain while keeping the era reasonably comparable; nba_api can go back
# to 1996-97 (earlier seasons lack the SEASON_YEAR column, and the three-point line
# only exists from 1979-80) if a longer curve is ever wanted.
#
# Cost of extending further: logs and bios are one call per season, but positions
# need 30 CommonTeamRoster calls per season, which is the real expense.
# --------------------------------------------------------------------------
MULTI_SEASON_SEASONS: list[str] = [
    f"{year}-{str(year + 1)[-2:]}" for year in range(2010, 2026)
]   # 2010-11 .. 2025-26
MULTI_SEASON_TARGETS: list[str] = [
    f"{year}-{str(year + 1)[-2:]}" for year in range(2013, 2026)
]   # 2013-14 .. 2025-26 (each has a full 3-season prior window)


def league_summary() -> str:
    """Return a one-line human summary of the league shape (for the CLI banner)."""
    return (
        f"{NUM_TEAMS} teams, {ROSTER_SIZE}-player rosters, "
        f"drafting from slot {MY_DRAFT_SLOT}; target season {TARGET_SEASON}"
    )
