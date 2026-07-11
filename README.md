# NBA Fantasy Draft Optimiser

A draft assistant for a Yahoo head-to-head (H2H) points-format fantasy NBA league
using a snake draft. It projects each player's season value, builds a ranked draft
board, simulates the snake draft, and validates the whole thing on past seasons
without data leakage.

This repository is built in strict version order. **V1-V4 are built.** V5 (live
Yahoo integration, multi-season distribution, visualisation) is specified in
[`nba_draft_optimiser_masterdoc.md`](nba_draft_optimiser_masterdoc.md) and is not
built yet.

## Honest framing

Drafting is only a modest lever on H2H success - in-season management (waivers,
schedule-based streaming, trades) usually matters more. The value of this project
is as a demonstration of a multi-source pipeline, a defensible projection, a
constrained decision engine, a simulation, and a leakage-free backtest. It is
**not** a guaranteed league-winning edge, and the outputs do not claim to be.

## What V1 does

1. **Ingest** - pulls per-game player logs from `nba_api` (one league-wide call
   per season) and caches each season to `data/` as Parquet. Cached seasons are
   never re-pulled. Live calls are rate-limited (short sleep, retry-with-backoff,
   per-call timeout).
2. **Score** - applies the league scoring formula to each game and aggregates to a
   season total per player. Because the projection is a prior-season total,
   availability (games missed) is baked in automatically.
3. **Project** - V1's projection for season N is each player's **prior-season
   total** ("next season = last season"). This is the naive, defensible baseline.
4. **Board** - ranks players by projected total, descending. In V1 the ranking is
   the board (no value-over-replacement yet).
5. **Draft** - simulates a full snake draft where every team drafts greedily (best
   available). Returns each team's roster.
6. **Backtest** - scores the greedy roster on the **actual** target-season totals
   and compares it against random drafts, reporting the margin.

## Method, in one line

Projection = prior-season fantasy-point total. Board = sort by projection. Draft =
greedy snake. Backtest = greedy roster's actual points minus the mean of many
seeded random rosters, on a held-out (later) season.

## Time-ordering and no leakage

The single most important property: **a projection for season N uses only data
from seasons strictly before N.** This is enforced by an explicit assertion in
[`src/projection.py`](src/projection.py), not merely by convention, and covered by
[`tests/test_leakage.py`](tests/test_leakage.py). Train/test is always
time-ordered - earlier seasons in, a later season scored - never a random split
across time.

## Benchmarking

No model output is reported in isolation. The greedy draft is benchmarked against
a random draft. A single random draft is high variance, so the baseline is the
**mean over many seeded random drafts** (`RANDOM_BASELINE_TRIALS` in config), and
the result is reported as a **margin** plus the share of trials the greedy roster
beats.

The backtest simulates the **full draft** - every team drafts under the same
strategy - rather than replaying a real historical draft, because no faithful
replay of a past draft exists. This choice is documented in
[`src/backtest.py`](src/backtest.py).

## V2 - realistic projection and a value-over-replacement board

V2 adds the two upgrades from the spec and **measures whether each one helps**,
keeping only what does:

1. **Recency-weighted points-per-game x projected games-played**
   ([`project_v2`](src/projection.py)). Points-per-game is games-played-weighted
   over the most recent 2-3 seasons (so a low-minutes season counts for less), and
   the total is that rate times a separate **projected games-played** estimate.
   Durability is modelled directly: a season a player *missed entirely* counts as
   zero games in the games-played estimate (the availability signal) - but only
   from his first appearance onward, so a sophomore is not penalised for seasons
   before he was in the league. The same leakage guard applies as V1.
2. **Value over replacement** ([`build_board_vor`](src/board.py)). Each player's
   value is his projection minus a per-position replacement level, where the
   replacement rank is `round(NUM_TEAMS x starters_per_team[position])` derived
   from league size and roster slots. Positions are a coarse `G/F/C` taken from
   nba_api rosters ([`src/positions.py`](src/positions.py)); true Yahoo
   multi-position eligibility is V5. Positions are sourced across the whole
   (leakage-safe) projection window so players who missed a season are still
   placed. Players with no found position are scored against an overall
   replacement level and **counted, never dropped**.

### V2 result (honest finding)

On this data, **V1's prior-season total is a hard-to-beat baseline and V2's
projection does not improve it**:

| Projection | Pooled MAE (3-season window, targets 2021-22/22-23/23-24) |
|---|---|
| V1 - prior-season total | **466.7** |
| V2 - recency PPG x GP | 480.4 |

V2 wins individual seasons occasionally (e.g. 2021-22) but loses on balance. This
matches a well-known result: last season alone is a strong naive predictor, and
multi-season blending dampens genuine recent change while the games-played
estimate adds variance (COVID-shortened seasons do not help). The single-season
**draft** comparison is high variance and inconclusive (V2 was within ~2% of V1 on
2023-24, sometimes ahead, sometimes behind, depending on the season).

**Decision, per the spec ("keep the simpler version if the complex one does not
beat it"): V1's prior-season-total projection remains the projection of record.**
The V2 code is retained because (a) it is the apples-to-apples benchmark, (b) the
value-over-replacement board is independent of the projection and is the more
promising half, and (c) a trustworthy verdict on the draft needs the multi-season
distribution backtest, which is V5. Run `python src/main.py` to see the V1-vs-V2
numbers for your own configured seasons and scoring.

## V3 - smarter draft engine

V3 replaces greedy picking with a constrained decision engine, and introduces the
realistic scoring objective that makes roster construction matter.

**The starting-lineup objective.** In H2H points only your starters score each
week, so a roster is worth its **best legal starting lineup**, not the sum of all
13 players ([`src/lineup.py`](src/lineup.py)). Depth at a position therefore has
diminishing returns. This is the objective V3 optimises and is scored on; the
whole-roster sum is still reported for continuity. (This is the documented choice
the spec asks for in Section 7 - bench value via streaming and injury cover is
real but not modelled.)

**The engine** ([`snake_draft_smart`](src/draft.py)) values each candidate by:

- **Marginal lineup value** - how much he improves my best starting lineup given
  the slots I have already filled. A fourth guard adds nothing once my guard and
  flex slots are covered, so the engine stops over-drafting deep positions.
- **Positional-run risk** - the expected value drop-off at his position before my
  next pick (assuming greedy opponents), the cheap substitute for a full opponent
  simulation. It is **gated on positive marginal value** so it never resurrects
  idle depth - a subtle but important fix.

Every other team drafts greedily; **the engine is only as valid as that opponent
model** (the binding constraint the spec flags).

### V3 result

The smart engine beats the V2 greedy roster on the starting-lineup objective, and
- unlike the V2 projection - the win is **robust across every season tested**:

| Target season | V2 greedy lineup | V3 smart lineup | Margin |
|---|---|---|---|
| 2021-22 | 24,365 | 26,128 | **+1,763** |
| 2022-23 | 22,277 | 25,055 | **+2,778** |
| 2023-24 | 27,780 | 30,567 | **+2,787** |

The win is structural: greedy ignores lineup construction and wastes picks on depth
it cannot start, while the engine balances the roster. **Decision: V3 is kept.**

**Monte Carlo opponent model: deferred (not built).** The spec makes it optional -
build it only if the cheap heuristic is shown to leave value on the table, and it
needs an ADP data source. The heuristic already beats greedy convincingly, so a
full opponent simulation is not justified yet; it would also introduce a new data
source better placed alongside the Yahoo work in V5. This is recorded as a
deliberate choice, not an omission.

## V4 - sophisticated projection (aging + shrinkage)

V4 adds two refinements to the V2 points-per-game estimate, each measured
separately and kept only if it improves time-ordered accuracy versus V2.

1. **Aging curve** ([`build_aging_curve`](src/projection.py)). From every
   consecutive-season pair a player completed in prior seasons, we take the ratio
   of the second year's per-game rate to the first and summarise it by age. The
   curve rises into the mid-20s, sits near 1.0 around age 26, then declines - and
   a player's projection is multiplied by the chained factor from his most recent
   age forward to the target season. Age comes from one league-wide
   `leaguedashplayerbiostats` call per season ([`load_player_bios`](src/ingest.py)).
2. **Empirical-Bayes shrinkage** ([`project_v4`](src/projection.py)). Each player's
   rate is regressed toward his position's mean with weight `k / (games + k)`, so
   extreme small-sample seasons are trusted less.

### Known flaw - survivorship bias (documented, as the spec requires)

The aging curve uses only players who appeared in **both** seasons of a pair.
Players who decline and leave the league are excluded, so late-career decline is
**understated**. This is visible in the fitted curve: the multiplier falls through
the early 30s (~0.89 -> 0.83) but then *ticks back up* at 36-38, because only the
strongest old players survive to be measured. The curve is kept with this caveat
flagged in code and here; a fuller fix (treating league exits as a decline signal)
needs careful injury-versus-decline handling and is left as future work.

### V4 result

MAE versus the V2 baseline (the spec's reference), scored on a common player set:

| Variant | 2023-24 MAE | 2022-23 MAE |
|---|---|---|
| V2 - PPG x GP | 470.3 | 501.3 |
| **V4 aging only** | **465.1** | **491.1** |
| V4 shrinkage only | 488.0 | 516.5 |
| V4 aging + shrinkage | 486.4 | 509.4 |

**Aging improves on V2 on both metrics in every season tested; shrinkage does
not** (regressing toward a coarse positional mean drags genuinely-high players
down). **Decision: keep aging, drop shrinkage** (`V4_USE_SHRINKAGE = False` in
config; flip it on to re-measure). Honest caveat: even the aging-improved
projection still does **not** beat the dead-simple V1 prior-season total on MAE -
across V2, V3 and V4, no "sophisticated" projection has beaten that baseline on
this data, exactly as the spec warns. V1 remains the projection of record; aging
is a genuine improvement *within* the V2 family.

## Setup

```bash
py -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

## Configure your league (required before trusting output)

[`src/config.py`](src/config.py) ships with **placeholder** league settings so the
pipeline runs out of the box. They are illustrative "standard Yahoo-style"
defaults, **not** your league. Before reading anything into the result, replace:

| Setting | What to fill in | Where to find it |
|---|---|---|
| `SCORING` | Each stat's point value | Yahoo: League Settings -> Scoring |
| `NUM_TEAMS` | Number of teams | League Settings |
| `ROSTER_SLOTS` | Your starting + bench slot composition | League Settings -> Roster Positions |
| `MY_DRAFT_SLOT` | Your first-round pick number (1 = first overall) | Your draft position |

`SEASONS` and `TARGET_SEASON` are not league-specific and have sensible defaults
you can adjust. (In V5 the scoring and roster come live from the Yahoo API and
these placeholders disappear.)

## Run

```bash
python src/main.py
```

Prints the V1/V2/V4 projection accuracy (with the aging and shrinkage variants);
the actual-season totals for the random baseline, V1 greedy and V2 greedy; and the
V3 smart roster with its starting-lineup margin over V2 greedy.

## Test

```bash
.venv/Scripts/python -m pytest
```

- `test_leakage.py` - the V1 projection for season N uses no season-N data, and
  the guard trips under a forced regression.
- `test_baseline.py` - the greedy draft beats a random draft on the backtest
  season.
- `test_projection_v2.py` - V2 leakage guard, PPG x GP mechanics, and V2 beating
  V1 on the injury-rebound case it is designed for.
- `test_vor.py` - scarcer positions out-rank equal projections; unknown-position
  players are scored not dropped; multi-season position merge prefers the recent.
- `test_lineup.py` - starting-lineup value excludes bench, fills slots optimally,
  and gives zero marginal value to redundant depth (the core V3 idea).
- `test_draft_smart.py` - the smart engine beats greedy on lineup value when depth
  is wasted, and produces a complete, duplicate-free draft.
- `test_aging.py` - the aging curve rises young and declines old, is leakage-safe
  (target rows do not change it), and shrinkage regresses small samples hardest.

## Known limitations

- **Prior-season totals (V1) penalise players returning from injury.** V2's
  points-per-game x games-played split softens this for players who were *present
  but limited*; a player who missed a full season is still correctly discounted by
  the durability model.
- **No rookie signal.** Players with no NBA history have no projection and are
  absent from the board (both versions).
- **Availability (games played) is the largest, least predictable source of
  variance.** V2 models it explicitly via projected games-played, but it remains
  the dominant noise term and is the main reason a single-season backtest is
  unreliable.
- **V2 positions are coarse (`G/F/C`) and single-valued.** True Yahoo
  multi-position eligibility, which materially changes value-over-replacement, is
  V5. Some players have no position in the nba_api rosters and fall back to an
  overall replacement level (the count is surfaced each run).
- **The aging curve has survivorship bias** (V4): players who age out and leave the
  league are not in the consecutive-season pairs, so late-career decline is
  understated and the curve even ticks up at 36-38. Flagged in code and above.
- **No "sophisticated" projection has beaten the V1 prior-season total on MAE**
  (V2, V3, V4 all tested). V1 remains the projection of record; V4's aging is an
  improvement only relative to V2.
- **The draft engine is only as valid as its opponent model.** V3 opponents draft
  greedily (a simple bot); the engine's edge would shrink against smarter
  opponents. A full Monte Carlo opponent model is deferred (see the V3 section).
- **V3 bench value is not modelled.** Only starters score in the lineup objective,
  so streaming and injury-cover value from the bench is ignored - a real lever in
  H2H that drafting alone does not capture.
- **A single-season backtest is high variance.** The trustworthy multi-season
  distribution is V5. The V3 win was checked across three seasons to guard against
  a fluke, but the V2 draft comparison remains inconclusive on one season.

## Repository layout (V1 - V4)

```
src/
  config.py      scoring dict, league config, seasons, V2/V3/V4 params (PLACEHOLDERS marked)
  columns.py     named constants for the verified nba_api column names
  ingest.py      nba_api logs + rosters + bios -> Parquet cache (rate-limited, retried)
  scoring.py     apply scoring, aggregate season totals
  projection.py  V1 prior totals; V2 recency PPG x GP; V4 aging + shrinkage; leakage guards
  positions.py   V2: coarse G/F/C positions from rosters
  board.py       V1 ranking; V2 value-over-replacement
  lineup.py      V3: best legal starting-lineup value (the H2H objective)
  draft.py       greedy / random snake draft; V3 smart engine
  backtest.py    V1 greedy-vs-random; V2/V4 projection MAE/RMSE; V3 smart-vs-greedy
  main.py        CLI entry point (runs V1-V4, reports the comparisons)
tests/
  test_leakage.py        V1 leakage guard
  test_baseline.py       greedy beats random
  test_projection_v2.py  V2 leakage, PPGxGP, accuracy vs V1
  test_vor.py            value-over-replacement behaviour
  test_lineup.py         starting-lineup objective
  test_draft_smart.py    V3 smart engine beats greedy
  test_aging.py          V4 aging curve, leakage-safety, shrinkage
data/              Parquet cache (git-ignored, disposable)
```

Files for later versions (`yahoo.py`, `reconcile.py`, `viz.py`, `.env`) do not
exist yet by design.
