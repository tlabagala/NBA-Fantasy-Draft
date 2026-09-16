# NBA Fantasy Draft Optimiser

A draft assistant for a Yahoo head-to-head (H2H) points-format fantasy NBA league
using a snake draft. It projects each player's season value, builds a ranked draft
board, simulates the snake draft, and validates the whole thing on past seasons
without data leakage.

This repository is built in strict version order, per
[`nba_draft_optimiser_masterdoc.md`](nba_draft_optimiser_masterdoc.md), plus two
usability layers beyond the original spec: a live draft-day CLI
([V6](#v6---live-draft-assistant-srclive_draftpy)) and a Streamlit web UI over it
([V7](#v7---draft-day-web-ui-srcapp_uipy-draft_apppy)). **V1-V7 are all built.**
The one step that remains is yours: the one-time Yahoo OAuth authorisation that
switches the pipeline from placeholder league settings to your live ones (see
[V5](#v5---yahoo-integration-multi-season-distribution-and-presentation)).

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
6. **Backtest** - scores the greedy roster on the **actual** target-season totals.
   This roster *is* the naive baseline every later version is measured against
   (see [Benchmarking](#benchmarking)); a random draft is scored beneath it only
   as a sanity floor.

## Method, in one line

Projection = prior-season fantasy-point total. Board = sort by projection. Draft =
greedy snake. Backtest = each engine's actual points minus the **naive
prior-season-totals draft's** actual points, on a held-out (later) season.

## Time-ordering and no leakage

The single most important property: **a projection for season N uses only data
from seasons strictly before N.** This is enforced by an explicit assertion in
[`src/projection.py`](src/projection.py), not merely by convention, and covered by
[`tests/test_leakage.py`](tests/test_leakage.py). Train/test is always
time-ordered - earlier seasons in, a later season scored - never a random split
across time.

## Benchmarking

No model output is reported in isolation, and **the baseline is a naive model, not
a random draft.**

**The baseline: the naive prior-season-totals draft.** A greedy snake draft off the
V1 board - "take whoever scored most last season". It is the model a competent but
unsophisticated human approximates with a printed cheat sheet, so it is the bar
worth clearing, and every engine's headline margin is measured against it
([`naive_baseline_roster`](src/backtest.py)). It is held to the same leakage guard
as everything else, so the comparison is fair.

**The random draft is a sanity floor, not a baseline.** It is still computed - the
mean over `RANDOM_BASELINE_TRIALS` seeded drafts - but it shows only that the board
and pick order are wired up. Nobody drafts randomly, so beating it is evidence of
nothing. It is reported last and labelled a floor everywhere it appears.

One consequence worth stating plainly: **V1 *is* the naive baseline** (its
projection is the prior-season total), so there is no "V1 versus baseline" margin
to report - it is the reference, and the meaningful margins start at V2.

> **Deviation from the spec, recorded.** The masterdoc (Section 3, rule 2) names
> *random* as V1's naive draft and prior-season-totals only "later". Random was
> kept as V1's baseline long enough to show it is uninformative - a +153% margin
> that means nothing - so the reference was moved down to the naive model the same
> rule names for later versions. The spec's actual requirement, "always benchmark
> against a naive baseline", is better served this way.

The backtest simulates the **full draft** - every team drafts under the same
strategy - rather than replaying a real historical draft, because no faithful
replay of a past draft exists. This choice is documented in
[`src/backtest.py`](src/backtest.py).

## Evaluating a projection: why MAE is not enough

Every projection result in this README used to be a single MAE number. That is a
weak test for this problem, for two reasons:

1. **A draft board is a ranking.** Nothing consumes the projected point value - the
   draft consumes the *order*. A projection that doubles every player's total and
   adds 500 has a catastrophic MAE and produces an **identical board**. (There is a
   test asserting exactly that: `test_rank_metrics_ignore_a_monotone_rescaling`.)
2. **Only the top of the board matters.** In a 12-team, 13-round league exactly 156
   players are drafted. Being wrong about the 400th-ranked player costs nothing, yet
   MAE weights him the same as the first pick.

[`projection_accuracy`](src/backtest.py) now reports three families:

| Family | Metrics | What it catches that MAE misses |
|---|---|---|
| Error size | `mae`, `rmse`, `medae`, `p90_ae`, `bias` | `medae` is robust to injury blow-ups; `p90_ae` is the tail itself; **`bias` is signed**, so systematic over-prediction is visible |
| Association | `pearson`, `r2`, `spearman` | `spearman` is rank correlation - the closest single number to "is this a good board" |
| Top-of-board | `precision_at_k`, `ndcg_at_k`, `spearman_at_k`, `precision_at_round1` | judges only the players actually drafted; `ndcg` weights the early picks most |

### Result: the metrics disagree, and no variant dominates

Pooled over the 13 target seasons (k = 156, round 1 = 12):

| Variant | MAE | RMSE | MedAE | bias | Spearman | R² | P@156 | NDCG | Spear@156 | P@rd1 |
|---|---|---|---|---|---|---|---|---|---|---|
| V1 prior total | 508.9 | 688.9 | **384.3** | +34.9 | **0.731** | 0.515 | **0.735** | 0.884 | **0.576** | 0.532 |
| V2 PPG x GP | 521.1 | 683.4 | 410.8 | +13.3 | 0.712 | 0.526 | 0.714 | 0.878 | 0.548 | 0.558 |
| **V4 aging** | **501.8** | 659.3 | 401.2 | **+9.1** | **0.731** | 0.559 | 0.731 | **0.888** | 0.558 | **0.577** |
| V4 shrinkage | 522.1 | 674.0 | 423.8 | +29.4 | 0.711 | 0.540 | 0.713 | 0.877 | 0.539 | 0.545 |
| V4 aging+shrink | 504.9 | **652.2** | 408.6 | +28.5 | 0.730 | **0.570** | 0.726 | 0.886 | 0.565 | **0.577** |

**V4's aging curve wins the error magnitude, the tail, the bias and the
rank-weighted top of the board** (NDCG, first-round precision). **V1 wins the median
error and the ordering within the drafted pool** (`precision_at_k`,
`spearman_at_k`); Spearman overall is a dead tie at 0.731.

Read together that is a coherent story rather than a contradiction: aging mainly
fixes the *large* misses - ageing players whose prior-season total badly overstates
them - which is why it wins MAE, RMSE, p90 and bias while slightly losing the
median. Shrinkage loses on every family, confirming the decision to leave it off.

### Two things only the new metrics could show

**V1 is systematically optimistic.** Its bias is **+34.9 points per player**, against
+9.1 for V4 aging. Prior-season totals over-predict, because the players who
finished highest last season are disproportionately those who were both good *and*
lucky with health, and because nobody in the pool is getting younger. MAE cannot
reveal a systematic direction; `bias` does, and it is a genuine point in V4's
favour that eight versions of this README never recorded.

**The board is far weaker at the top than it looks.** Overall Spearman is a
respectable ~0.73, but **`spearman_at_k` is only ~0.55-0.58** - ordering the players
you actually draft is much harder than separating good players from bad ones. And
**`precision_at_round1` is ~0.53-0.58**: barely more than half of your projected
first-rounders actually finish in the real top twelve. That number belongs in the
honest-framing conversation more than any MAE does.

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

| Projection | Pooled MAE (13 targets, 2013-14 .. 2025-26) |
|---|---|
| V1 - prior-season total | **508.9** |
| V2 - recency PPG x GP | 521.1 |

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

**This result was revised when history was extended from 3 target seasons to 13.**
It originally read "robust across every season tested" (3/3). Across 2013-14 to
2025-26 the same-board comparison is much weaker:

| Comparison (same board for all 12 teams) | Result over 13 targets |
|---|---|
| V3 smart beats V2 greedy | 7/13 seasons, mean **+846** lineup points |
| V3 smart beats the naive baseline | 8/13 seasons, mean **-0.2%** |

So on the measurement the V3 gate was originally written against, the engine is
close to a coin flip. That is not a reason to drop it - it is a reason to distrust
the measurement. When the same engine is measured against a league that did *not*
also change boards, it is decisively better ([head-to-head](#head-to-head-holding-the-league-fixed):
**+7.8%, 12/13 seasons, p = 0.0004**).

The mechanism is unchanged and still the right idea: greedy ignores lineup
construction and wastes picks on depth it cannot start, while the engine balances
the roster. **Decision: V3 is kept** - on the head-to-head evidence, not this table.

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

MAE on a common player set, across all 13 target seasons (2013-14 .. 2025-26). MAE
is only one of three metric families - see
[Evaluating a projection](#evaluating-a-projection-why-mae-is-not-enough) for the
ranking and bias picture, which is more favourable to aging than MAE alone
suggests:

| Variant | Pooled MAE | Seasons won (lowest MAE) |
|---|---|---|
| **V4 aging only** | **501.8** | 2/13 |
| V4 aging + shrinkage | 504.9 | 6/13 |
| V1 - prior-season total | 508.9 | 5/13 |
| V2 - PPG x GP | 521.1 | 0/13 |
| V4 shrinkage only | 522.1 | 0/13 |

**Aging beats V2 in every season, and now beats V1 pooled** (501.8 vs 508.9), with
some V4 variant winning 8 of 13 seasons outright. Shrinkage alone remains the worst
variant and never wins a season. **Decision: keep aging, drop shrinkage**
(`V4_USE_SHRINKAGE = False` in config; flip it on to re-measure).

**Updated finding: the V1 baseline is no longer undefeated on accuracy.** This
README previously recorded "no sophisticated projection has beaten V1 on MAE",
which was true of the three seasons it was then measured on. Over 13 seasons V4's
aging curve beats it. V1 remains the *baseline* by definition; it is no longer the
most accurate projection.

Note the interaction with history depth: the aging curve reads *all* seasons before
the target, so extending history to 2010-11 gave later targets a curve fitted on up
to 15 seasons instead of 4. The sparse 35-38 age buckets that produce the
documented survivorship "tick up" are far better populated as a result (47-118
consecutive-season pairs per age on deep history, against 7-19 on a 5-season
window).

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
**13 target seasons** (2013-14 .. 2025-26) and reports a distribution against named
benchmarks, all scored on the starting-lineup objective with actual points.

| Benchmark | Mean margin | Mean % | Seasons won |
|---|---|---|---|
| **Naive baseline (prior-season totals)** | **-74.2** | **-0.2%** | **8/13** |
| ADP proxy | +477.9 | +2.1% | 7/13 |
| Random draft (floor, not a result) | +15,085.9 | +153.0% | 13/13 |

**Read this honestly - and note what changed.** On three target seasons this table
showed the engine beating the naive baseline by +4.8%. On thirteen it shows
**-0.2% (p = 0.96, 95% CI [-7.8%, +7.5%])**: no effect whatsoever. Per-season
results swing from -20.3% to +19.9%. The apparent edge was noise, and three seasons
was nowhere near enough to see that.

**This does not mean the engine is worthless - it means this experiment cannot
measure it.** Every team here drafts off the same board, so changing the board
changes the whole league at once and my slot's advantage largely cancels. The
[head-to-head](#head-to-head-holding-the-league-fixed) below, which holds the
opposing league fixed, finds a large and highly significant edge on exactly the
same 13 seasons. Where the two disagree, the head-to-head is the one measuring what
a draft assistant is for.

The random floor is in the table only to show the floor is where you would expect;
+153% is not a result, which is exactly why random is no longer the baseline.

The ADP benchmark uses a **proxy** by default - the aging-adjusted V4 ranking, a
reasonable stand-in for market consensus - because real ADP needs a live source.
Pass `adp_board_for` to `multi_season_backtest` (or authorise Yahoo and use
`get_draft_results`) to benchmark against genuine draft positions.

### Head-to-head: holding the league fixed

Every backtest above hands the *same* board to all twelve teams, so two runs differ
by an entire league's worth of information. If everybody drafts off the aging
board, the draft as a whole improves and my slot inherits some of that - whether or
not I did anything smarter than my opponents. That measures "what if the whole
league upgraded?", which is not the question a draft assistant exists to answer,
and on 13 seasons it turns out to measure essentially nothing at all.

[`head_to_head_backtest`](src/backtest.py) asks the narrower question instead: my
eleven opponents all draft naively (best available by last season's totals), and
only **my** board and strategy change. The margin over drafting naively myself,
into that same room, is the edge a version actually buys me
([`snake_draft_vs_fixed_opponents`](src/draft.py) is the primitive).

**Every draft slot is averaged, not just slot 1.** Within a single season the spread
across the twelve slots is ~12 percentage points - as large as the variation between
seasons - so a single-slot verdict is mostly reading draft position. Averaging gives
**12 slots x 13 seasons = 156 drafts per variant**.

| My version | Mean margin | Mean % | 95% CI | p | Seasons won | Slot-seasons positive |
|---|---|---|---|---|---|---|
| V2 VOR board, greedy | -3,051.0 | **-11.8%** | [-21.9, -1.7] | 0.026 | 2/13 | 28% |
| V4 aging board, greedy | -627.6 | **-1.4%** | [-11.2, +8.3] | 0.755 | 6/13 | 48% |
| V3 engine on V2 board | +1,626.8 | **+7.8%** | [+4.3, +11.3] | 0.0004 | 12/13 | 72% |
| **V4 board + V3 engine** | **+3,182.0** | **+14.6%** | **[+9.0, +20.3]** | **0.0001** | **13/13** | **85%** |

**The shipped stack wins every season tested**, with a worst case of +1.0% and a
mean of +14.6% - and unlike everything else in this README, it is statistically
unambiguous (p = 0.0001).

**The engine is what buys the edge; boards drafted greedily do not.** Greedy off
the VOR board is significantly *worse* than naive (-11.8%, p = 0.026). The
mechanism is visible in the replacement levels: on 2023-24 they are G 1,808 /
F 1,552 / **C 826**, so value over replacement massively inflates centres.
Greedy-off-VOR opens Jokic, Embiid, Vucevic - taking Embiid (2,830 actual) and
Vucevic (3,239) while Doncic (5,138) and Gilgeous-Alexander (4,613) sit on the
board. The VOR top 13 averages 3,591 actual points against the naive top 13's
3,976. You reach for scarcity against a market that is not pricing it, and pay in
absolute points. Greedy off the *aging* board is merely neutral (-1.4%, p = 0.76).

The V3 engine fixes exactly this: it values a candidate by what he adds to *my*
starting lineup and uses run-risk to time positional scarcity, rather than treating
the board as a draft order. **So a better board is useful *inside* the engine, not
*as* a ranking** - which is how [V6](#v6---live-draft-assistant-srclive_draftpy)
and [V7](#v7---draft-day-web-ui-srcapp_uipy-draft_apppy) already use it. And the
better projection genuinely compounds: V4's board nearly doubles the engine's edge
(+14.6% vs +7.8%).

Caveat: opponents are non-reactive - they never adjust to what I take. This is a
better opponent model than "everyone runs my strategy", not a good one, and it is
the remaining reason to treat +14.6% as an upper bound rather than a forecast.

### What sets the starting lineup (and why it barely matters)

Scoring a roster means picking its best legal starting lineup and summing those
players' **actual** points - but "best" needs something to sort by, and that choice
is an assumption, not a detail. Both options are now computed and reported
(`lineup_value_source` on `backtest_v3`, `multi_season_backtest` and
`head_to_head_backtest`; see [`score_lineup` / `lineup_bounds`](src/backtest.py)):

- **`"hindsight"`** (default) - starters picked using the actual season totals. The
  best lineup the roster *could* have produced: an **upper bound**. This is what
  every backtest in this project did before the parameter existed.
- **`"projection"`** - starters picked from the board the roster was drafted off,
  then scored on actual points. The lineup you would have set in October and never
  touched: a **lower bound**.

| Head-to-head edge over naive | Hindsight | Projection-picked |
|---|---|---|
| V2 VOR board, greedy | -11.8% | -14.4% |
| V4 aging board, greedy | -1.4% | -2.0% |
| V3 engine on V2 board | +7.8% | +6.7% |
| **V4 board + V3 engine** | **+14.6%** | **+15.0%** |

**The headline is robust to the assumption.** Hindsight inflates every roster's
absolute points, but it inflates the naive baseline at least as much as the engine,
so the *margin* is near-identical - and for the shipped stack the pessimistic bound
is fractionally **higher**. Real managers set lineups weekly using in-season
information that is legitimately available (unlike at the draft), so the truth sits
between these two, and it makes almost no difference which end you take.

To be explicit about what this is and is not: **no draft decision anywhere sees
target-season data** - the leakage guards in [`src/projection.py`](src/projection.py)
enforce that against the target season *and every season after it*. This is purely a
question of how generously the resulting roster is scored.

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

## V7 - draft-day web UI ([`src/app_ui.py`](src/app_ui.py), [`draft_app.py`](draft_app.py))

V6 works, but a real 12-team, 13-round draft is roughly 156 picks against a clock,
eleven of every twelve somebody else's, and typing each one by hand under time
pressure is the actual bottleneck - not the model. V7 changes nothing about the
projection, board or recommendation logic; it is an entry-speed and situational-
awareness layer over the exact same `LiveDraftState` V6 already validates.

**V7 is a local, single-user tool with no authentication, no database and no
deployment** - it is meant to run on your own laptop during your own draft, not
to be shared or hosted. It carries the same opponent-model limitation as V3/V6:
recommendations optimise for me alone, with a cheap positional-run heuristic
standing in for what opponents will actually do next, not a full simulation.

### Run it

```bash
streamlit run draft_app.py
```

The entry point is deliberately at the repo root rather than inside `src/`:
`streamlit run` puts the *script's own directory* on `sys.path`, and a script
inside `src/` cannot then resolve `from src import ...`. `draft_app.py` is a
three-line shim; all real UI logic lives in `src/app_ui.py`.

### League configuration, on the setup screen

`config.py` ships with four settings explicitly marked as PLACEHOLDERS in its own
header - `SCORING`, `NUM_TEAMS`, `ROSTER_SLOTS` and `MY_DRAFT_SLOT` - and editing
Python to fill them in was the main friction point before drafting live. The
setup screen now exposes all four directly, plus `TARGET_SEASON` and the V4
aging/shrinkage toggles, as an `st.form` (one rerun on submit, not one per
keystroke across ~24 fields):

- **Draft position, team count, target season** - the season selector offers every
  completed season in `config.SEASONS` (for a backtest-style sanity check against
  real results) plus the season right after the most recent one
  (`src.projection.next_season(SEASONS[-1])`), selected by default: the actual
  upcoming, not-yet-played season, which is the real live-draft case. As of this
  writing `config.SEASONS` runs through `"2025-26"` (now complete), so the
  default target is `"2026-27"`. The leakage guard only checks that no data FROM
  OR AFTER the target season was used, which trivially holds for a season with no
  games yet - no engine change was needed for this. The one real limitation: a
  season with zero games has no rookie signal at all (an existing, documented V1
  limitation, just more visible here since every player must come from prior-
  season data).
- **Roster slots** - a count per standard Yahoo slot (PG/SG/SF/PF/C/G/F/UTIL/BN),
  pre-filled from `config.ROSTER_SLOTS`.
- **Scoring** - a point value per stat (PTS/REB/AST/STL/BLK/TOV/FGM/FGA/FTM/FTA/
  FG3M/FG3A), pre-filled from `config.SCORING`.
- **Advanced (collapsed by default):** the V4 aging-curve and shrinkage toggles.

None of this is an engine change - `build_board_bundle` and `LiveDraftState`
already took `league` as a duck-typed parameter (every test in
`tests/test_draft_smart.py` and `tests/test_live_draft.py` passes a
`types.SimpleNamespace` stand-in, not the real `config` module), so the setup
screen just builds one from the submitted form instead of importing `config`
directly. **Two settings are deliberately NOT exposed**: `N_PROJECTION_SEASONS`
and `GP_RECENCY_DECAY`. Reading `src/projection.py` confirms `project_v4` calls
`project_v2(totals, target_season)` with no override, so those two are always
read from the literal `config` module regardless of what `league` object is
passed in - a slider that silently did nothing would be worse than no slider.

Because the board depends on these settings now, not just on the draft slot, the
cache key had to grow to match: `_cached_board_bundle` is keyed on the scoring
dict, roster slots, team count, target season, and the two V4 flags (all
hashable), with the `league` object itself passed unhashed - changing any
setting on the setup screen correctly rebuilds the board rather than serving a
stale one from a previous session's settings (verified directly: two different
`SCORING["PTS"]` values produce two different top-of-board values in the same
process, and re-requesting the same settings is a cache hit).

Resuming a saved draft intentionally does NOT restore custom league settings -
the saved JSON only ever stored `n_teams`, `roster_size`, `my_slot` and the pick
log (unchanged by this addition), so a resumed session rebuilds the board from
whatever `config.py` says today. The setup screen says so explicitly.

### What it adds over the CLI

- **Type-to-search, click-to-record** for every pick, mine or an opponent's -
  Streamlit's fuzzy filter narrowed 711 players to a single match on a five-
  character partial name (`"Jokic"` -> Nikola Jokić) in real-browser testing, and
  because only players Streamlit itself lists are selectable, a typo can no
  longer silently record the wrong player - a real failure mode in the CLI, where
  a partial name that happens to substring-match an unintended player would be
  drafted without complaint. This is the concrete, verifiable way V7 beats the
  CLI: not a claimed speed number, but a structurally eliminated error class.
- **Click-to-draft recommendations** - the same top-10 table V6 prints, now a
  clickable `st.dataframe` (row-select, wired via `on_select`) that drafts the
  clicked player for me, but only while I am actually on the clock - otherwise
  it renders read-only so a click can never mis-attribute a pick to whichever
  team happens to be up.
- **A status strip**: round and overall pick, who is on the clock, picks until my
  next turn (`gap_to_my_next_pick()` - already computed by V6 but never
  surfaced there), and unfilled starter slots by position.
- **Board and Search tabs** - the board with a G/F/C filter, and a name search
  that (unlike `LiveDraftState.find_player`, available players only) also shows
  drafted players with their status (`TAKEN - team 4, pick 22`) rather than
  reading like a typo.
- **The `RUN_RISK_WEIGHT` and top-n as live sliders**, so the positional-run
  aggressiveness can be tuned mid-draft rather than only in `config.py`.
- **Autosave on every pick plus an explicit resume prompt on launch.**
  `st.session_state` does not survive a browser refresh or a killed server -
  only the JSON file does. Declining to resume no longer silently destroys the
  old session either: the previous file is backed up to a timestamped
  `.bak-<timestamp>.json` before a fresh session's first pick can overwrite it
  (the same fix applied to the CLI). A second bug fixed in both interfaces:
  `undo` previously did not re-save, so an undone pick could survive on disk if
  the process died before the next pick's autosave.

### Engine change (the only one V7 makes)

`build_live_draft` was split into a slot-independent
[`build_board_bundle`](src/live_draft.py) (ingest -> scoring -> V4 projection ->
value-over-replacement board) plus the cheap per-slot `LiveDraftState`
construction. `build_live_draft`'s signature and behaviour are unchanged and
still covered by `tests/test_live_draft.py`. The Streamlit layer wraps
`build_board_bundle` in `st.cache_data` (which copies its return value on every
read, so nothing mutable ever leaks across cache reads) and holds
`LiveDraftState` only in `st.session_state`, never in `st.cache_resource` -
that cache does NOT copy its return value, and `record_pick` mutates the state
in place, so caching it that way would corrupt it across sessions.
`recommend()` is memoised keyed on the full pick log (not bare `pick_index`
alone, as an undo followed by a different pick can revisit the same pick_index
with a different roster - the pick log is what actually determines the result).

### Verification method (and its limits)

Real end-to-end behaviour was confirmed in an actual Chrome browser against a
running server: the setup screen, cold board build (0.21s from the Parquet
cache), click-to-draft, and type-to-search-and-select all work as described
above, screenshotted at each step. Streamlit's own `AppTest` harness confirmed
the caching spine (no rebuild on a slider interaction) and ten consecutive real
opponent picks with no crashes or duplicates - but `AppTest` driving a full
156-pick draft through actual page reruns turned out to carry a large fixed
per-rerun cost in this environment unrelated to this app's logic, so the full
156-pick, duplicate-free, both-interfaces-agree verification below instead
calls `src/app_ui.py`'s own callback functions directly (`_opponent_pick_callback`,
`_recommend_click_callback`, `_undo_callback`) against a lightweight
`st.session_state` double - the same functions Streamlit wires to the widgets,
just without paying to re-render the whole page 156 times:

- A full 156-pick draft via these callbacks completes with zero errors, no
  duplicate or missing players, and every team ends with exactly 13.
- The autosaved file matches the in-memory pick log after every single pick.
- A stale/duplicate callback firing for an already-recorded pick is a no-op
  (the double-submit guard).
- `undo` removes the pick from memory AND the saved file.
- A brand-new session built only from the saved file resumes with every pick
  intact (simulating a killed server and a fresh browser tab).
- Given an identical pick log, the CLI's `LiveDraftState.recommend()` and the
  UI's cached `_cached_recommend()` return an *identical* top-10 - the UI has
  not changed engine behaviour.

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
aging and shrinkage variants); the actual-season totals for the **naive baseline**
and V2 greedy with the margin between them (and the random floor beneath, marked as
a sanity check); the V3 smart roster with its starting-lineup margin over both the
V2 greedy engine and the naive baseline; the V5 multi-season distribution, headed by
the margin over the naive baseline; the head-to-head table (opponents held naive,
only my version changing); and the paths of the rendered charts.

First run pulls from `nba_api` and caches to `data/`; later runs read the cache.
With history from 2010-11 that first pull is substantial - 16 seasons of logs and
bios (one call each) plus **30 `CommonTeamRoster` calls per season** for positions,
so budget roughly 10 minutes once. The populated cache is about 36 MB. A full run
on the warm cache takes a few minutes, most of it the 156 drafts per variant in the
head-to-head.

## Test

```bash
.venv/Scripts/python -m pytest
```

- `test_leakage.py` - the V1 projection for season N uses no season-N data, and
  the guard trips under a forced regression.
- `test_baseline.py` - the naive baseline is exactly the prior-season-totals
  draft (not something that drifted), V2 and V3 report their margins against it,
  and it clears the random sanity floor.
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
  every named benchmark, with the naive baseline leading the reported columns and
  the random floor last.
- `test_head_to_head.py` - the fixed-opponent draft reduces exactly to the ordinary
  greedy snake draft when both sides share a board (the correctness anchor), stays
  legal when the two boards disagree on the player pool, and the head-to-head
  backtest reports a margin per variant that is consistent with its own totals.
- `test_lineup_bounds.py` - the hindsight bound reproduces the pre-parameter
  behaviour exactly, hindsight can never score below projection-picked (they really
  are bounds), a perfect projection collapses them onto each other, and the bounds
  carried in one head-to-head run match a dedicated run with that source selected.
- `test_projection_metrics.py` - the ranking metrics are invariant to a monotone
  rescaling that wrecks MAE (the reason they exist), bias is signed where MAE is
  not, median and p90 separate typical error from injury blow-ups, top-of-board
  metrics ignore the undraftable tail, and degenerate input returns `nan` rather
  than raising.
- `test_live_draft.py` - draft-position selection drives the pick clock; recording
  real picks (for any team) shrinks the board and advances the clock; undo and
  out-of-turn corrections; recommendations track the true remaining board and never
  resurrect idle depth; name search; save/resume round-trips. Unchanged by V7 -
  its whole point is proving the engine underneath the UI still behaves.

V7 (`src/app_ui.py`) deliberately has no `pytest` suite of its own - see
[V7's verification method](#verification-method-and-its-limits) for how it was
checked instead (a real browser session plus a direct-callback harness), rather
than adding Streamlit-specific tests that would need their own maintenance.

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
- **V4's aging projection beats V1 on pooled MAE** (501.8 vs 508.9 over 13
  targets). The "V1 is undefeated" finding this README used to record was true only
  of the three seasons it was measured on. V2 and shrinkage still lose.
- **A better projection helps the draft only through the engine.** The V4 aging
  board nearly doubles the engine's edge (+14.6% vs +7.8%), but drafted greedily it
  is worth nothing at all (-1.4%, p = 0.76).
- **Same-board results did not survive more data.** Two findings this project once
  reported as solid on three seasons - the engine beating the naive baseline by
  +4.8%, and V3 beating V2 greedy in every season - collapse to -0.2% and 7/13 over
  thirteen. Anything measured with all twelve teams on one board should be treated
  as provisional.
- **Board quality and draft order are not the same thing.** Drafting greedily off
  the value-over-replacement board is *worse than naive* (-30%) against a league
  that prices players by last season's totals. VOR belongs inside the engine, not
  as a ranking - see the head-to-head section.
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
- **Thirteen target seasons** (2013-14 .. 2025-26) feed the distributions, on
  history from 2010-11. That is enough to make the head-to-head unambiguous
  (p = 0.0001) but it is what *overturned* two earlier three-season findings, so
  treat any result here as provisional until it has been re-run on more data.
  `nba_api` supports back to 1996-97 if more is wanted; the cost is 30
  `CommonTeamRoster` calls per additional season.
- **2019-20 and 2020-21 are COVID-shortened** and are both target seasons, which
  adds avoidable noise to the games-played model and the distributions.
- **Scoring assumes a perfect season-long lineup by default.** Starters are picked
  knowing the actual outcome (`lineup_value_source="hindsight"`), which is an upper
  bound on what a roster is worth. The lower bound (`"projection"`) is reported
  alongside it and the two differ by well under a percentage point on the headline,
  so the conclusions do not depend on it - but the absolute point totals are
  optimistic. Both bounds also fix one starting lineup for the whole season, while
  real H2H sets lineups weekly; neither models that.

## Repository layout (V1 - V7)

```
draft_app.py     V7: root shim for `streamlit run draft_app.py` (3 lines)
src/
  config.py      scoring, league, seasons, V2-V5 params (PLACEHOLDERS marked)
  columns.py     named constants for the verified nba_api column names
  ingest.py      nba_api logs + rosters + bios -> Parquet cache (rate-limited, retried)
  scoring.py     apply scoring, aggregate season totals
  projection.py  V1 prior totals; V2 recency PPG x GP; V4 aging + shrinkage; leakage guards
  positions.py   V2: coarse G/F/C positions from rosters
  board.py       V1 ranking; V2 value-over-replacement
  lineup.py      V3: best legal starting-lineup value (the H2H objective)
  draft.py       greedy / random snake draft; V3 smart engine;
                 snake_draft_vs_fixed_opponents (league held fixed, only I change)
  backtest.py    naive baseline (the reference) + random floor;
                 projection_accuracy (error / association / ranking metrics);
                 V3 smart-vs-greedy-vs-naive; V5 multi-season distribution;
                 head_to_head_backtest (the honest per-version verdict);
                 score_lineup / lineup_bounds (who is allowed to set the lineup)
  yahoo.py       V5: live yfpy league settings, eligibility, draft results
  reconcile.py   V5: nba_api <-> Yahoo player join, unmatched surfaced
  viz.py         V5: plotnine charts -> outputs/
  live_draft.py  V6: interactive draft assistant for a REAL live draft (CLI);
                 V7 split out build_board_bundle() (slot-independent, cacheable)
  app_ui.py      V7: Streamlit UI logic - all of it lives here, not in draft_app.py
  main.py        CLI entry point (runs V1-V5, reports every comparison)
tests/           16 test modules, 72 tests (see the Test section)
project_overview.ipynb   high-level walkthrough with charts
.env.example     Yahoo credential template (copy to git-ignored .env)
data/            Parquet cache (git-ignored, disposable)
outputs/         rendered charts + live_draft_state.json (git-ignored, re-renderable)
```

## Headline results

| Question | Answer |
|---|---|
| Does it beat the naive baseline, league held fixed? | **Yes, +14.6%**, 13/13 seasons, 85% of slot-seasons, p = 0.0001 - the number to trust |
| Does it beat naive when the whole league upgrades too? | **No - -0.2%**, p = 0.96. That experiment measures nothing |
| Does it clear the random floor? | Yes - +153%, 13/13 seasons, but that is a wiring check, not a result |
| Did a more sophisticated projection help *accuracy*? | **Yes** - V4 aging beats V1 pooled (501.8 vs 508.9 over 13 targets) |
| Did that better projection help the *draft*? | **Yes, but only via the engine** - +14.6% with it, +7.8% on the V2 board |
| Did a smarter draft engine help? | **Yes** - the only component that beats naive at all |
| Is the VOR board a draft order? | **No** - greedy off VOR is **-30%**, worse than naive |
| Did the aging curve help? | **Yes** vs V2 (kept). Shrinkage did **not** (dropped) |

The pattern worth taking away: **the decision engine paid off, the fancier
projections did not.** Roster construction beat forecasting on this problem - and
the head-to-head sharpens it further: the projection and scarcity work only pays
off when the *engine* consumes it. Used directly as a draft order, it loses badly to
"take whoever scored most last season".
