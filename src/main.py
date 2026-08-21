"""CLI entry point - runs the pipeline end to end (V1 through V5).

    python src/main.py

Honest framing (master spec Section 1): drafting is only a modest lever on H2H
success; in-season management usually matters more. This tool demonstrates a
leakage-free multi-step pipeline - ingestion, scoring, defensible projections, a
constrained draft, and benchmarked backtests - not a guaranteed edge.

Each version's addition is reported against the version below it and kept only if it
helps. V5 reads live league settings from Yahoo when configured (otherwise the
placeholder scoring), and reports a multi-season distribution against named
benchmarks instead of a single high-variance number.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Player names contain non-ASCII characters (e.g. Jokic, Doncic). Force UTF-8 on
# stdout so printing the roster does not crash on a Windows cp1252 console.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Allow `python src/main.py` to work without installing the package: put the repo
# root on sys.path so `import src...` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import backtest, config, ingest, positions, projection, scoring, viz  # noqa: E402


def _resolve_scoring() -> dict[str, float]:
    """Use live Yahoo scoring when configured, else the placeholder config.SCORING.

    The agent never performs the OAuth; if Yahoo is configured but not yet
    authorised, this explains what to do and falls back to the placeholder so the
    rest of the pipeline still runs.
    """
    if not config.USE_YAHOO:
        print("Yahoo not configured - using placeholder SCORING. "
              "See .env.example to enable live league settings.")
        return config.SCORING
    try:
        from src import yahoo
        live, unmapped = yahoo.get_live_scoring()
        if not live:
            raise RuntimeError("Yahoo returned no scoring modifiers.")
        print(f"Loaded live Yahoo scoring: {live}")
        if unmapped:
            print(f"  unmapped Yahoo stats (review - no raw nba_api column): {unmapped}")
        return live
    except Exception as error:  # noqa: BLE001 - surface any auth/API issue clearly
        print(f"Yahoo configured but live pull failed: {error}")
        print("  Complete the one-time browser OAuth, then re-run. "
              "Falling back to placeholder SCORING for now.")
        return config.SCORING


def _positions_for(totals, target_season):
    """Coarse position map for a target season, sourced across its (prior) window."""
    window = projection.recent_prior_seasons(totals, target_season,
                                             config.N_PROJECTION_SEASONS)
    rbs = [(s, ingest.load_team_rosters(s)) for s in window]
    return positions.coarse_position_map_over_seasons(rbs)


def main() -> None:
    print("=" * 72)
    print("NBA Fantasy Draft Optimiser - V1 baseline vs V2 (recency PPGxGP + VOR)")
    print("=" * 72)
    print(f"League:   {config.league_summary()}")
    print(f"Seasons:  {config.SEASONS}  (projecting {config.TARGET_SEASON} "
          f"from prior data only)")
    scoring_weights = _resolve_scoring()
    print(f"Scoring:  {scoring_weights}")
    print("-" * 72)

    print("1. Ingesting per-game logs (cache -> Parquet) ...")
    logs = ingest.load_player_logs(config.SEASONS)

    print("2. Applying scoring and aggregating to season totals ...")
    totals = scoring.season_totals(scoring.apply_scoring(logs, scoring_weights))

    # --- Projection accuracy: V1 vs V2, time-ordered MAE/RMSE -------------
    print("3. Evaluating projection accuracy (time-ordered MAE/RMSE) ...")
    target_cmp = backtest.compare_projection_accuracy(totals, config.TARGET_SEASON, config)
    print(f"   Target {target_cmp['target_season']} "
          f"(n={target_cmp['n_common']} players scored on both):")
    print(f"     V1 (prior total)   MAE {target_cmp['v1']['mae']:>7,.1f}  "
          f"RMSE {target_cmp['v1']['rmse']:>7,.1f}")
    print(f"     V2 (PPG x GP)      MAE {target_cmp['v2']['mae']:>7,.1f}  "
          f"RMSE {target_cmp['v2']['rmse']:>7,.1f}")
    pooled = backtest.evaluate_projection_seasons(totals, config)
    if len(pooled) > 1:
        v1_mae = sum(r["v1"]["mae"] * r["n_common"] for r in pooled) / sum(r["n_common"] for r in pooled)
        v2_mae = sum(r["v2"]["mae"] * r["n_common"] for r in pooled) / sum(r["n_common"] for r in pooled)
        targets = ", ".join(r["target_season"] for r in pooled)
        print(f"   Pooled across {targets}:  V1 MAE {v1_mae:,.1f}  vs  V2 MAE {v2_mae:,.1f}")

    # --- Positions for the V2 value-over-replacement board ----------------
    # Source positions across the whole projection window (all leakage-safe, prior
    # seasons), so players who missed a season are still placed. Most recent wins.
    window = projection.recent_prior_seasons(totals, config.TARGET_SEASON,
                                             config.N_PROJECTION_SEASONS)
    print(f"4. Loading {window} rosters for coarse positions (VOR; leakage-safe) ...")
    rosters_by_season = [(season, ingest.load_team_rosters(season)) for season in window]
    position_map = positions.coarse_position_map_over_seasons(rosters_by_season)

    # --- V4 projection accuracy: aging curve + shrinkage vs V1/V2 ----------
    print("4b. Loading player ages and evaluating V4 projection (aging + shrinkage) ...")
    bios = ingest.load_player_bios(config.SEASONS)
    cmp4 = backtest.compare_projection_accuracy_v4(
        totals, config.TARGET_SEASON, bios, position_map, config)
    print(f"    Target {cmp4['target_season']} (n={cmp4['n_common']}, "
          f"{cmp4['n_unknown_age']} players without an age):")
    for label, key in [("V1 prior total", "v1"), ("V2 PPG x GP", "v2"),
                       ("V4 aging only", "v4_aging_only"),
                       ("V4 shrink only", "v4_shrink_only"),
                       ("V4 aging+shrink", "v4")]:
        print(f"      {label:<16} MAE {cmp4[key]['mae']:>7,.1f}  "
              f"RMSE {cmp4[key]['rmse']:>7,.1f}")

    # --- Drafts: V1 board, V2 VOR board, random baseline ------------------
    print(f"5. Backtesting {config.TARGET_SEASON}: V1 greedy, V2 greedy, and "
          f"{config.RANDOM_BASELINE_TRIALS} random drafts ...")
    res_v1 = backtest.backtest_v1(totals, config.TARGET_SEASON, config)
    res_v2 = backtest.backtest_v2(totals, config.TARGET_SEASON, config, position_map)

    print("-" * 72)
    print(f"V2 roster (drafting from slot {config.MY_DRAFT_SLOT}, "
          f"actual {config.TARGET_SEASON} points):")
    print(res_v2["my_roster"].to_string(index=False))
    if res_v2["n_unknown_pos"]:
        print(f"  (note: {res_v2['n_unknown_pos']} board players had no position in "
              f"{window} rosters; scored against an overall replacement level, not dropped)")

    print("-" * 72)
    print("Actual-season roster totals (higher is better):")
    print(f"  Random baseline (mean) : {res_v1['random_mean']:>10,.1f}  "
          f"(std {res_v1['random_std']:,.1f}, n={res_v1['n_trials']})")
    print(f"  V1 greedy (prior total): {res_v1['greedy_total']:>10,.1f}  "
          f"(margin vs random {res_v1['greedy_total'] - res_v1['random_mean']:+,.1f})")
    print(f"  V2 greedy (PPGxGP+VOR) : {res_v2['greedy_total']:>10,.1f}  "
          f"(margin vs random {res_v2['greedy_total'] - res_v1['random_mean']:+,.1f})")
    v2_vs_v1 = res_v2["greedy_total"] - res_v1["greedy_total"]
    print(f"  V2 - V1 margin         : {v2_vs_v1:>+10,.1f}")

    # --- V3: smart engine vs V2 greedy on the starting-lineup objective ----
    print("-" * 72)
    print("6. V3 smart engine (marginal lineup value + positional-run risk) vs "
          "V2 greedy,")
    print(f"   scored on the best legal STARTING LINEUP (only starters score; "
          f"run-risk weight {config.RUN_RISK_WEIGHT}) ...")
    res_v3 = backtest.backtest_v3(totals, config.TARGET_SEASON, config, position_map)
    print(f"   V3 smart roster ('*' = makes the actual starting lineup):")
    print(res_v3["smart_roster"].to_string(index=False))
    print("   Starting-lineup actual points (the realistic H2H objective):")
    print(f"     V2 greedy engine : {res_v3['greedy_lineup']:>10,.1f}")
    print(f"     V3 smart engine  : {res_v3['smart_lineup']:>10,.1f}  "
          f"(margin {res_v3['lineup_margin']:+,.1f})")
    print(f"     (whole-roster sums for continuity: V2 {res_v3['greedy_total']:,.1f}  "
          f"V3 {res_v3['smart_total']:,.1f})")

    # --- V5: multi-season distribution against named benchmarks + charts ---
    print("-" * 72)
    print(f"7. V5 multi-season distribution backtest over {config.MULTI_SEASON_TARGETS} "
          f"(smart engine vs prior-totals and ADP-proxy drafts) ...")
    ms_totals = scoring.season_totals(scoring.apply_scoring(
        ingest.load_player_logs(config.MULTI_SEASON_SEASONS), scoring_weights))
    ms_bios = ingest.load_player_bios(config.MULTI_SEASON_SEASONS)
    ms = backtest.multi_season_backtest(
        ms_totals, ms_bios, config.MULTI_SEASON_TARGETS, config,
        positions_for=lambda s: _positions_for(ms_totals, s))
    import pandas as pd  # local: only needed to pretty-print the table
    print(pd.DataFrame(ms["per_season"]).to_string(index=False))
    s = ms["summary"]
    print(f"   Smart engine vs random       : mean {s['vs_random_mean']:+,.1f} "
          f"({s['pct_vs_random_mean']:+.1f}%, range {s['pct_vs_random_min']:+.1f}%.."
          f"{s['pct_vs_random_max']:+.1f}%); wins {s['wins_vs_random']}/{s['n_seasons']} seasons")
    print(f"   Smart engine vs prior-totals : mean {s['vs_prior_mean']:+,.1f} "
          f"({s['pct_vs_prior_mean']:+.1f}%, std {s['vs_prior_std']:,.1f}); "
          f"wins {s['wins_vs_prior']}/{s['n_seasons']} seasons")
    print(f"   Smart engine vs ADP proxy    : mean {s['vs_adp_mean']:+,.1f} "
          f"({s['pct_vs_adp_mean']:+.1f}%, std {s['vs_adp_std']:,.1f}); "
          f"wins {s['wins_vs_adp']}/{s['n_seasons']} seasons")

    print("8. Rendering plotnine charts to outputs/ ...")
    curve = projection.build_aging_curve(totals, bios, config.TARGET_SEASON, config)
    v4_board = backtest.board_mod.build_board_vor(
        projection.project_v4(totals, config.TARGET_SEASON, bios, position_map, config),
        position_map, config)
    chart_paths = [
        viz.plot_aging_curve(curve, config.TARGET_SEASON),
        viz.plot_value_by_position(v4_board, config.TARGET_SEASON),
        viz.plot_projection_accuracy(cmp4),
        viz.plot_backtest_distribution(ms),
    ]
    for path in chart_paths:
        print(f"   wrote {path.relative_to(Path(__file__).resolve().parent.parent)}")
    print("=" * 72)

    proj_verdict = "BEATS" if target_cmp["v2"]["mae"] < target_cmp["v1"]["mae"] else "DOES NOT BEAT"
    draft_verdict = "BEATS" if v2_vs_v1 > 0 else "DOES NOT BEAT"
    v3_verdict = "BEATS" if res_v3["lineup_margin"] > 0 else "DOES NOT BEAT"
    v4_aging_verdict = "BEATS" if cmp4["v4_aging_only"]["mae"] < cmp4["v2"]["mae"] else "DOES NOT BEAT"
    v4_shrink_verdict = "BEATS" if cmp4["v4_shrink_only"]["mae"] < cmp4["v2"]["mae"] else "DOES NOT BEAT"
    print(f"V2 verdict: projection {proj_verdict} V1 on MAE; "
          f"V2 draft {draft_verdict} V1 draft (whole-roster) on {config.TARGET_SEASON}.")
    print(f"V3 verdict: smart engine {v3_verdict} V2 greedy on the starting-lineup "
          f"objective ({config.TARGET_SEASON}).")
    print(f"V4 verdict: aging {v4_aging_verdict} V2 on MAE (kept); "
          f"shrinkage {v4_shrink_verdict} V2 on MAE (dropped by default).")
    v5_prior = "BEATS" if s["vs_prior_mean"] > 0 else "DOES NOT BEAT"
    v5_adp = "BEATS" if s["vs_adp_mean"] > 0 else "DOES NOT BEAT"
    yahoo_state = "live Yahoo settings" if config.USE_YAHOO else "placeholder settings (Yahoo not configured)"
    print(f"V5 verdict: across {s['n_seasons']} seasons the smart engine {v5_prior} the "
          f"prior-totals draft and {v5_adp} the ADP proxy on average; ran on {yahoo_state}.")
    print("Per the spec, keep each component only where it helps - see the numbers "
          "above and the README. The multi-season distribution above is the "
          "trustworthy read; a single season is high variance.")


if __name__ == "__main__":
    main()
