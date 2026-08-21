# NBA Fantasy Draft Optimiser

A draft assistant for a Yahoo head-to-head (H2H) points-format fantasy NBA league
using a snake draft. It projects each player's season value, builds a ranked draft
board, simulates the snake draft, and validates the whole thing on past seasons
without data leakage.

This repository is built in strict version order, per
[`nba_draft_optimiser_masterdoc.md`](nba_draft_optimiser_masterdoc.md). **All five
versions are built (V1-V5).** The one step that remains is yours: the one-time
Yahoo OAuth authorisation that switches the pipeline from placeholder league
settings to your live ones (see [V5](#v5---yahoo-integration-multi-season-distribution-and-presentation)).

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

## V5 - Yahoo integration, multi-season distribution, and presentation

V5 closes the loop: live league settings, a real cross-source join, a trustworthy
multi-season verdict, and charts.

### Live Yahoo league settings ([`src/yahoo.py`](src/yahoo.py))

Pulls the real scoring formula, positional eligibility, and draft results via
`yfpy`, replacing the placeholder `SCORING`. Credentials live only in a
git-ignored `.env` (see [`.env.example`](.env.example)) - never hardcoded, never
committed. **The one-time browser OAuth is yours to perform**; the code sets
everything up but cannot authorise on your behalf. Until you do, the pipeline
prints that Yahoo is not configured and falls back to the placeholder scoring, so
everything else still runs. Yahoo stat abbreviations are mapped to real nba_api
columns, and anything unmappable (double-doubles, shooting percentages - which have
no raw nba_api column) is **surfaced for review rather than guessed**.

> This module could not be executed during the build (no credentials), so treat it
> as carefully-structured scaffolding and verify the field names on your first
> authorised run. Its deterministic helpers are unit tested.

### Cross-source reconciliation ([`src/reconcile.py`](src/reconcile.py))

`nba_api` uses NBA.com ids, Yahoo uses its own - this is the single place the two
are bridged. It prefers a public id crosswalk when supplied, otherwise matches on
**normalised names** (accent-, punctuation- and suffix-stripped, so `Nikola Jokić`
matches `Nikola Jokic` and `Jaren Jackson Jr.` matches `Jaren Jackson`). Names that
are ambiguous on either side are **left unmatched and surfaced for manual review** -
the code never mis-joins on a guess and never silently drops a player.

### Multi-season distribution backtest

A single season is high variance, so the trustworthy verdict runs the engine across
several target seasons and reports a **distribution against named benchmarks**, all
scored on the starting-lineup objective with actual points:

| Season | Smart engine | Prior totals | ADP proxy | Random |
|---|---|---|---|---|
| 2021-22 | 26,128.5 | 23,608.5 | 19,055.8 | 11,341.2 |
| 2022-23 | 25,054.8 | 25,653.8 | 26,199.8 | 9,726.9 |
| 2023-24 | 30,567.0 | 29,991.8 | 30,350.5 | 9,637.3 |

| Benchmark | Mean margin | Mean % | Seasons won |
|---|---|---|---|
| Random draft | +17,014.9 | **+168.4%** | 3/3 |
| Prior-season totals | +832.1 | **+3.4%** | 2/3 |
| ADP proxy | +2,048.1 | **+11.1%** | 2/3 |

**Read this honestly.** Beating a random draft by 168% is a low bar. Against a
competent opponent who simply drafts by last season's totals the edge is about
**3%**, and it *loses* one season in three (2022-23: -2.3% vs prior totals, -4.4%
vs ADP). That spread is the whole reason the spec demanded a distribution rather
than a single number.

The ADP benchmark uses a **proxy** by default - the aging-adjusted V4 ranking, a
reasonable stand-in for market consensus - because real ADP needs a live source.
Pass `adp_board_for` to `multi_season_backtest` (or authorise Yahoo and use
`get_draft_results`) to benchmark against genuine draft positions.

### Charts ([`src/viz.py`](src/viz.py))

Three `plotnine` charts render to `outputs/` (git-ignored, re-renderable):
`aging_curve.png`, `value_by_position.png`, `backtest_distribution.png`. They use
only nba_api-derived data, so they run without any Yahoo credentials.

### Optional DuckDB layer

Not built. The spec marks it optional and not required for the project to
function; Parquet plus pandas is sufficient at this scale. The dependency is left
commented in `requirements.txt` for anyone who wants a SQL story.

## V6 - live draft assistant ([`src/live_draft.py`](src/live_draft.py))

Everything above is offline: it builds a board once, then simulates every team's
pick under a fixed strategy to validate the engine. V6 is the tool you actually
open **during a real draft**, where your draft slot is whatever you were assigned
and every pick - yours and everyone else's - is a live human decision, not a bot.

```bash
python -m src.live_draft
```

- **Pick your draft position.** Prompted at startup (defaults to
  `config.MY_DRAFT_SLOT`); the snake pick order is generated from it, so the tool
  knows exactly whose turn it is at every point in the draft.
- **Record every pick as it happens**, for any team - `pick <name> [team]` (or the
  `me <name>` shortcut for your own picks). The board shrinks and the pick clock
  advances after every entry, whether the pick was yours or an opponent's.
- **The engine adjusts accordingly.** After each recorded pick, `rec` re-ranks the
  true remaining board with the same marginal-lineup-value + positional-run-risk
  rule the V3 backtest validated (`src/draft.py::rank_candidates`, extracted from
  `snake_draft_smart` so both share one scoring path) - driven by what has actually
  been taken and the real number of opponent picks before your next turn, not a
  simulated one.
- **Other commands:** `board` (top available), `roster [team]`, `lineup` (your best
  legal starting lineup right now), `find <text>` (name search), `undo` (fix a
  mis-entered pick), `save`/auto-save-on-pick, and `quit`. Progress is written to
  `outputs/live_draft_state.json` (git-ignored) after every pick, and the CLI
  offers to resume it on the next launch - only the pick log is replayed onto a
  freshly built board, so projections can be refreshed between sessions without
  invalidating the save.

Known limitation: recommendations still assume *my* remaining picks are the only
ones being optimised - there is still no model of what an opponent will do next
(the same V3 limitation, now live instead of simulated).

## Project overview notebook

[`project_overview.ipynb`](project_overview.ipynb) is a high-level walkthrough -
objective, data wrangling, methodology, results, discussion, conclusion - with the
charts and representative code. Start there for the narrative; this README is the
reference.

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

Runs every version end to end and prints: the V1/V2/V4 projection accuracy (with the
aging and shrinkage variants); the actual-season totals for the random baseline, V1
greedy and V2 greedy; the V3 smart roster with its starting-lineup margin; the V5
multi-season distribution against all three benchmarks; and the paths of the three
rendered charts.

First run pulls from `nba_api` and caches to `data/`; later runs read the cache.

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
- `test_reconcile.py` - name normalisation across accents/suffixes/punctuation; the
  crosswalk surfaces unmatched players and refuses to mis-join ambiguous names.
- `test_yahoo.py` - Yahoo position and stat mappings (pure helpers, no network).
- `test_multi_season.py` - the multi-season backtest returns a distribution against
  every named benchmark.
- `test_live_draft.py` - draft-position selection drives the pick clock; recording
  real picks (for any team) shrinks the board and advances the clock; undo and
  out-of-turn corrections; recommendations track the true remaining board and never
  resurrect idle depth; name search; save/resume round-trips.

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
- **A single-season backtest is high variance.** The V5 multi-season distribution
  is the trustworthy read, and it shows the edge over a competent baseline is small
  (~3% vs prior-season totals) and loses one season in three.
- **The live Yahoo path is unexecuted scaffolding** until you complete the OAuth.
  Field names in `yahoo.py` should be verified on the first authorised run.
- **The ADP benchmark is a proxy** (aging-adjusted ranking), not real draft-position
  data, until a live ADP source is supplied.
- **Only three seasons** feed the multi-season distribution by default. More
  history would tighten every estimate, especially the aging curve.

## Repository layout (V1 - V6)

```
src/
  config.py      scoring, league, seasons, V2-V5 params (PLACEHOLDERS marked)
  columns.py     named constants for the verified nba_api column names
  ingest.py      nba_api logs + rosters + bios -> Parquet cache (rate-limited, retried)
  scoring.py     apply scoring, aggregate season totals
  projection.py  V1 prior totals; V2 recency PPG x GP; V4 aging + shrinkage; leakage guards
  positions.py   V2: coarse G/F/C positions from rosters
  board.py       V1 ranking; V2 value-over-replacement
  lineup.py      V3: best legal starting-lineup value (the H2H objective)
  draft.py       greedy / random snake draft; V3 smart engine
  backtest.py    V1 greedy-vs-random; V2/V4 accuracy; V3 smart-vs-greedy; V5 multi-season
  yahoo.py       V5: live yfpy league settings, eligibility, draft results
  reconcile.py   V5: nba_api <-> Yahoo player join, unmatched surfaced
  viz.py         V5: plotnine charts -> outputs/
  live_draft.py  V6: interactive draft assistant for a REAL live draft (CLI)
  main.py        CLI entry point (runs V1-V5, reports every comparison)
tests/           13 test modules, 43 tests (see the Test section)
project_overview.ipynb   high-level walkthrough with charts
.env.example     Yahoo credential template (copy to git-ignored .env)
data/            Parquet cache (git-ignored, disposable)
outputs/         rendered charts + live_draft_state.json (git-ignored, re-renderable)
```

## Headline results

| Question | Answer |
|---|---|
| Does the draft beat random? | Yes - **+168%** starting-lineup points on average, 3/3 seasons |
| Does it beat a competent baseline (prior-season totals)? | Marginally - **+3.4%**, 2/3 seasons |
| Did a more sophisticated projection help? | **No** - V1's prior-season total was never beaten on MAE |
| Did a smarter draft engine help? | **Yes** - V3 beat greedy on the lineup objective in every season |
| Did the aging curve help? | **Yes** vs V2 (kept). Shrinkage did **not** (dropped) |

The pattern worth taking away: **the decision engine paid off, the fancier
projections did not.** Roster construction beat forecasting on this problem.
