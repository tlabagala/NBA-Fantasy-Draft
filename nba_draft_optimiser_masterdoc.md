# NBA Fantasy Draft Optimiser - Master Build Specification

> **Purpose of this document.** This is the single source of truth for building this project in Claude Code. It encodes the locked decisions, the build order, the data contracts, and the rules the agent must follow. Build in strict version order (V1 first, fully working, before any V2 work). Do not skip ahead, and do not add complexity that the current version does not call for.

---

## 0. How to use this document

- Build **V1 in full first**. It is small and must run end to end before anything else starts.
- Each later version adds exactly one layer. A layer is kept only if it beats the previous version's backtest or serves a stated purpose; if it does not beat the baseline, keep the simpler version and record the result.
- When a decision in this doc conflicts with an instinct to "do it properly", follow this doc. The simplicity is deliberate.
- Where this doc says to verify something at runtime (e.g. exact API column names), verify it - do not assume a schema.

---

## 1. Project summary and honest framing

A draft assistant for a **Yahoo H2H points-format** fantasy NBA league using a **snake draft**. It projects each player's season value, builds a ranked draft board, simulates the snake draft, and validates the whole thing on past seasons without data leakage.

**Frame the project honestly in code comments and the README.** Drafting is only a modest part of H2H success; in-season management (waivers, schedule-based streaming, trades) usually matters more. The value of this project is as a demonstration of a multi-source pipeline, a defensible projection model, a constrained decision engine, a simulation, and a leakage-free backtest - not as a guaranteed league-winning edge. Do not overstate predictive power anywhere in the outputs.

---

## 2. Locked decisions (do not relitigate)

| Decision | Choice | Rationale |
|---|---|---|
| Language | Python (3.10+) | Required by `nba_api`; user's target language for this build |
| Fantasy format | H2H points | Makes expected score additive and the optimisation tractable |
| Draft type | Snake | The target league format |
| NBA stats source | `nba_api` | Standard maintained NBA.com client; granular per-game data |
| League data source | Yahoo Fantasy API via `yfpy` | Real scoring, positional eligibility, draft results - **deferred to V5** |
| V1 storage | Parquet + pandas | No database needed for a few seasons, single user |
| Optional later storage | DuckDB over Parquet | SQL query layer without a server; only if a SQL story is wanted |
| Visualisation | `plotnine` | ggplot grammar in Python, matching user preference |
| Projection (V1) | Prior-season total fantasy points | Simplest defensible baseline; bakes in availability |
| Draft engine (V1) | Greedy off a static board | Strong baseline; no opponent model needed to start |

**Do NOT in V1:** use the Yahoo API, set up any database, add a second data source, build value-over-replacement, build a games-played model, build aging/shrinkage, or build a Monte Carlo opponent model. All of these arrive in later versions.

---

## 3. Rules the agent must follow (non-negotiable)

1. **Time-ordering / no leakage.** Any projection for season N may use only data from seasons before N. No information from the target season may touch the projection or the draft decision. Enforce this with an explicit guard or assertion, not just by convention.
2. **Always benchmark against a naive baseline.** No model output is reported in isolation. The projection is compared against "next season = last season"; the draft is compared against a naive draft (random in V1, prior-season-totals and ADP later).
3. **Never fabricate.** Do not invent stats, column names, API fields, or player data. If a value is unknown or a column name is uncertain, verify it at runtime or surface it - do not guess silently.
4. **Match complexity to need.** Prefer the simplest method that works. Use empirical-Bayes-style shrinkage over full Bayesian unless explicitly justified. Use the cheap positional-run heuristic before a full Monte Carlo.
5. **Flag assumptions and limitations in code and README.** Where a method assumes something (stationarity of player roles, normality, independence) or has a known flaw (survivorship bias in the aging curve), say so in a comment and in the write-up.
6. **Secrets out of the repo.** Any Yahoo credentials or tokens live in a `.env` file that is git-ignored. Never commit them, never hardcode them.
7. **Prose and docs in Australian English. Use hyphens, not em dashes.**
8. **Clean, typed, chainable pandas.** Type-hint public functions. Keep transformations readable (method chaining over deeply nested intermediate variables).

---

## 4. Target repository structure

Built incrementally - V1 creates only the files it needs; later versions add the rest.

```
nba-draft-optimiser/
  src/
    config.py          # V1: scoring dict, league config, seasons
    ingest.py          # V1: nba_api pull -> Parquet cache
    scoring.py         # V1: apply scoring, aggregate season totals
    projection.py      # V1: prior-season-totals; V2+: weighted, GP split; V4: aging, shrinkage
    board.py           # V1: ranking; V2+: value over replacement
    draft.py           # V1: greedy snake; V3: marginal value, positional-run, optional MC
    backtest.py        # V1: single-season vs random; V5: multi-season distribution
    main.py            # CLI entry point, runs the current version end to end
    yahoo.py           # V5: yfpy league pull
    reconcile.py       # V5: nba_api <-> Yahoo player-ID join
    viz.py             # V5: plotnine charts
  data/                # Parquet cache (git-ignored)
  tests/               # leakage guard, baseline checks
  .env                 # V5: Yahoo creds (git-ignored)
  .gitignore
  README.md            # method, assumptions, limitations
  requirements.txt
```

---

## 5. Build plan by version

### V1 - Thin vertical slice (build completely first)

**Goal:** historical NBA data in, ranked board out, snake draft simulated, scored against a random draft, with time-ordering enforced.

**Modules and contracts:**

- `config.py`
  - `SCORING: dict[str, float]` - the league's point values per stat, hardcoded from the league settings page. Comment that this is the V1 stand-in for the Yahoo API.
  - `LEAGUE` config - number of teams, roster size and slot composition, your draft slot, list of seasons to use, target backtest season.

- `ingest.py`
  - `load_player_logs(seasons: list[str]) -> pd.DataFrame` - pull per-game logs via `nba_api` (use a season-level game-log endpoint to minimise calls). Add a short sleep between calls and retry-with-backoff on failure. Cache each season to `data/` as Parquet and read from cache if present. **Verify the actual returned column names at runtime; do not assume them.**

- `scoring.py`
  - `apply_scoring(logs: pd.DataFrame, scoring: dict) -> pd.DataFrame` - add a per-game fantasy-points column.
  - `season_totals(logs: pd.DataFrame) -> pd.DataFrame` - aggregate to one row per player-season with total fantasy points, games played, and identifying fields.

- `projection.py`
  - `project_v1(totals: pd.DataFrame, target_season: str) -> pd.DataFrame` - return each player's **prior-season total** as the projection for the target season. This is the naive baseline. Include a leakage guard: assert no target-season rows are used as input.

- `board.py`
  - `build_board(projection: pd.DataFrame) -> pd.DataFrame` - rank players by projected total, descending. In V1 the ranking is the board (no value over replacement).

- `draft.py`
  - `snake_draft(board, n_teams, roster_size, my_slot, strategy="greedy") -> dict[int, list]` - generate the snake pick order, and at each pick take the top available player (greedy). Return each team's roster.

- `backtest.py`
  - `backtest_v1(totals, target_season, league) -> dict` - project the target season from prior data, run the greedy snake draft, sum each roster's **actual** target-season total points, and compare your roster against a randomly drafted roster. Return both totals and the margin.

- `main.py` - run the above end to end for the configured target season and print the drafted roster and the backtest result.

- `tests/`
  - A test asserting the projection for season N contains no season-N input (leakage guard).
  - A test asserting the greedy draft beats a random draft on the backtest season.

**V1 definition of done:**
- Runs end to end from a single command.
- Prints a drafted roster with its actual-season points total.
- Beats a random draft on at least the configured backtest season.
- Leakage guard test passes.
- README documents the V1 method and its known limitation (prior-season totals penalise players who were injured last season but are healthy now).

---

### V2 - Realistic projection and board

**Adds:**
- Recency-weighted average across 2-3 seasons (weight each season by games played).
- Split projection into projected points-per-game multiplied by projected games-played, so durability is modelled directly rather than relying on a stale total.
- Value over replacement in `board.py`: subtract a per-position replacement-level baseline (tied to league size and roster slots) so the board reflects positional scarcity.

**Why:** these are the highest-value, lowest-complexity upgrades and they fix V1's two biggest weaknesses (stale totals, no scarcity).

**Definition of done:** projection beats the V1 baseline on time-ordered MAE/RMSE; the V2 drafted roster is compared against V1's. Keep any component only if it helps.

---

### V3 - Smarter draft engine

**Adds (in this order):**
- Marginal-roster-value: value a candidate by what he adds given the slots you have left, to stop over-drafting deep positions.
- Positional-run-risk adjustment: adjust each candidate's value by the expected drop-off at his position before your next pick. This is the cheap substitute for a full opponent simulation.
- **Optional** full Monte Carlo opponent model (bots draft by ADP with noise; simulate the draft forward many times; pick the candidate maximising expected final-roster value). Build this only if the backtest shows the cheap heuristic is leaving value on the table. A full opponent model needs draft-position data, so introduce an ADP source (Yahoo draft results or a public source) here.

**Why:** greedy is already strong; complexity in the engine must prove itself. The opponent-model quality is the binding constraint on the engine's validity - document that.

**Definition of done:** the smarter engine beats V2's greedy roster in the backtest. If it does not, keep greedy and report the finding.

---

### V4 - Sophisticated projection

**Adds:**
- Aging curve (estimate year-over-year change by age; production peaks mid-20s then declines).
- Shrinkage toward a role-based prior (empirical-Bayes style), so extreme seasons regress to the mean and are trusted less the fewer games they have.

**Flag the known flaw:** the aging curve estimated only from players appearing in consecutive seasons has **survivorship bias** - players who age badly leave the league, so late-career decline is understated. Document this in a comment and the README; partially mitigate by including players who left the league as a decline signal if feasible.

**Definition of done:** each addition improves time-ordered projection accuracy versus V2. Keep only what helps.

---

### V5 - Yahoo integration and presentation

**Adds:**
- `yahoo.py` using `yfpy`: pull the real scoring formula, positional eligibility, and draft results. Replace the hardcoded `SCORING` with the live values. **Credentials in `.env`, git-ignored. Never commit secrets. The agent must not perform the OAuth authorisation itself - it sets up the code and instructs the user to complete the one-time browser authorisation.**
- `reconcile.py`: join `nba_api` players to Yahoo players. Prefer an existing public ID crosswalk; otherwise match on normalised names and surface the unmatched stragglers for manual review. Do not silently drop or mis-join players.
- Multi-season backtest in `backtest.py`: run across several seasons and report a **distribution** of outcomes, not a single number (a one-season result is high-variance). Benchmark against prior-season-totals and ADP-based drafts.
- `viz.py` with `plotnine`: aging curve, value-by-position, and backtest results.
- README finalised with method, assumptions, and limitations.
- **Optional** DuckDB-over-Parquet query layer if a SQL story is wanted; not required for the project to function.

**Definition of done:** the pipeline reads live league settings, the two sources join cleanly with unmatched players surfaced rather than dropped, and the backtest reports a multi-season distribution against named benchmarks.

---

## 6. Data contracts and schema notes

- `nba_api` returns pandas DataFrames whose exact column names must be confirmed at runtime. Do not hardcode assumed column names - read them from the first successful pull and reference them via named constants.
- Player identity differs across sources: `nba_api` uses NBA.com IDs, Yahoo uses Yahoo IDs. The V5 reconciliation join is the only place these are bridged. Until V5 there is one source and no join.
- Cache layer: raw API pulls are written to `data/` as Parquet and never re-pulled unless missing. Treat `data/` as disposable and git-ignored.

---

## 7. Validation and anti-leakage (applies to every version)

- Train/test splits are always **time-ordered**: train on earlier seasons, test on a later one. Never a random split across time.
- Every projection has a leakage guard asserting no target-season data is used.
- Every modelled output is reported against a baseline. Report margins, not just absolute numbers.
- The draft backtest must state whether it simulates the full draft (bots and you) or holds opponents fixed, and why. There is no perfect replay of a historical draft; document the choice.

---

## 8. Known limitations to document (in README)

- Drafting is a limited lever on H2H success relative to in-season management.
- Prior-season-based projections misjudge players returning from injury and have no signal for rookies (no NBA history).
- Availability (games played) is the largest and least predictable source of variance.
- The aging curve has survivorship bias.
- The draft engine is only as valid as its opponent model.
- A single-season backtest is high-variance; the multi-season distribution (V5) is the trustworthy version.

---

## 9. Coding standards

- Python 3.10+, type hints on public functions, docstrings stating purpose, inputs, outputs, and any assumptions.
- Readable pandas (method chaining; avoid sprawling intermediate variables).
- Deterministic where possible; seed any randomness (random draft, Monte Carlo) and expose the seed.
- Small, testable functions. The leakage guard and the baseline comparison are first-class tests, not afterthoughts.
- `requirements.txt` pinned. Secrets only in git-ignored `.env`.
- Comments and prose in Australian English; hyphens, not em dashes.

---

## 10. First action for the agent

Scaffold the repo structure, then implement **V1 only**, end to end, until `main.py` runs and the V1 definition of done is met. Stop at the end of V1 and report the backtest result before beginning V2.
