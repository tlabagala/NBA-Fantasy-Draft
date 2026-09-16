"""Projection metrics beyond MAE: ranking, association and error shape.

MAE answers "how far off are the point totals". A draft board never consumes point
totals - it consumes the ORDER, and only for the ~156 players who actually get
drafted. These tests pin down the properties that make the extra metrics worth
having, above all the one that matters most: a projection can be arbitrarily wrong
on the numbers and still be a perfect board.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src import backtest as backtest_mod
from src import columns as C


def _projection(ids: list[int], values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({C.COL_PLAYER_ID: ids, C.COL_PROJECTED_PTS: values})


def _truth(ids: list[int], values: list[float]) -> dict[int, float]:
    return dict(zip(ids, values))


IDS = list(range(1, 21))
ACTUAL = _truth(IDS, [float(2000 - 80 * i) for i in range(len(IDS))])


def test_a_perfect_projection_scores_perfectly() -> None:
    perfect = _projection(IDS, [ACTUAL[i] for i in IDS])
    m = backtest_mod.projection_accuracy(perfect, ACTUAL, draftable_k=10, round1_k=3)

    assert m["mae"] == 0.0
    assert m["bias"] == 0.0
    assert m["r2"] == 1.0
    assert m["spearman"] == 1.0
    assert m["precision_at_k"] == 1.0
    assert m["ndcg_at_k"] == 1.0
    assert m["precision_at_round1"] == 1.0


def test_rank_metrics_ignore_a_monotone_rescaling_but_mae_does_not() -> None:
    """The headline property: the board is a ranking, so scale and offset are free.

    A projection that doubles every value and adds 500 is catastrophically wrong on
    MAE and badly biased, yet it produces the IDENTICAL draft board. This is why
    MAE alone is the wrong test for this problem.
    """
    base = _projection(IDS, [ACTUAL[i] for i in IDS])
    rescaled = _projection(IDS, [ACTUAL[i] * 2 + 500 for i in IDS])

    a = backtest_mod.projection_accuracy(base, ACTUAL, draftable_k=10, round1_k=3)
    b = backtest_mod.projection_accuracy(rescaled, ACTUAL, draftable_k=10, round1_k=3)

    # Ranking and association are untouched...
    for metric in ("spearman", "precision_at_k", "ndcg_at_k", "spearman_at_k",
                   "precision_at_round1", "pearson"):
        assert a[metric] == b[metric], metric
    # ...while error size and bias blow up.
    assert b["mae"] > a["mae"]
    assert b["bias"] > 500


def test_a_reversed_projection_is_maximally_wrong_on_rank() -> None:
    reversed_proj = _projection(IDS, [ACTUAL[i] for i in reversed(IDS)])
    m = backtest_mod.projection_accuracy(reversed_proj, ACTUAL,
                                         draftable_k=10, round1_k=3)
    assert m["spearman"] == -1.0
    assert m["precision_at_k"] == 0.0
    assert m["precision_at_round1"] == 0.0
    assert m["ndcg_at_k"] < 0.7   # heavily penalised by the log2 rank discount


def test_bias_is_signed_where_mae_is_not() -> None:
    """Systematic over-prediction must be visible, which |error| cannot show."""
    over = _projection(IDS, [ACTUAL[i] + 300 for i in IDS])
    under = _projection(IDS, [ACTUAL[i] - 300 for i in IDS])

    m_over = backtest_mod.projection_accuracy(over, ACTUAL)
    m_under = backtest_mod.projection_accuracy(under, ACTUAL)

    assert m_over["bias"] == pytest.approx(300, abs=0.1)
    assert m_under["bias"] == pytest.approx(-300, abs=0.1)
    # MAE is blind to the direction - identical for both.
    assert m_over["mae"] == m_under["mae"]


def test_median_and_tail_separate_typical_error_from_blowups() -> None:
    """One catastrophic miss should move MAE and p90 far more than the median.

    This is the injury case: a projection can be right about almost everyone and
    still carry a large MAE because a handful of players missed the season.
    """
    values = [ACTUAL[i] for i in IDS]
    for i in range(3):                       # 3 of 20 = 15%, so p90 catches them
        values[i] = ACTUAL[IDS[i]] + 8000
    blown = _projection(IDS, values)

    m = backtest_mod.projection_accuracy(blown, ACTUAL)
    assert m["medae"] == 0.0            # the typical player is still projected exactly
    assert m["mae"] > 300               # but the mean is dragged up
    assert m["p90_ae"] > m["mae"]       # and the tail further still


def test_top_of_board_metrics_only_judge_the_drafted_pool() -> None:
    """Getting the undraftable tail wrong must not affect precision@k."""
    good_top = [ACTUAL[i] for i in IDS]
    # Scramble only players ranked below the draftable cutoff of 10.
    scrambled = good_top[:10] + list(reversed(good_top[10:]))
    m_clean = backtest_mod.projection_accuracy(
        _projection(IDS, good_top), ACTUAL, draftable_k=10, round1_k=3)
    m_tail = backtest_mod.projection_accuracy(
        _projection(IDS, scrambled), ACTUAL, draftable_k=10, round1_k=3)

    assert m_tail["precision_at_k"] == m_clean["precision_at_k"] == 1.0
    assert m_tail["precision_at_round1"] == m_clean["precision_at_round1"] == 1.0
    # ...but a whole-board metric does notice.
    assert m_tail["spearman"] < m_clean["spearman"]


def test_metrics_are_bounded_and_degenerate_input_is_safe() -> None:
    m = backtest_mod.projection_accuracy(
        _projection(IDS, [float(i) for i in IDS]), ACTUAL,
        draftable_k=10, round1_k=3)
    assert 0.0 <= m["precision_at_k"] <= 1.0
    assert 0.0 <= m["ndcg_at_k"] <= 1.0
    assert -1.0 <= m["spearman"] <= 1.0

    empty = backtest_mod.projection_accuracy(_projection([], []), ACTUAL,
                                             draftable_k=10)
    assert empty["n"] == 0
    assert all(isinstance(empty[k], float) and math.isnan(empty[k])
               for k in ("mae", "spearman", "precision_at_k", "ndcg_at_k"))


def test_top_of_board_metrics_absent_without_a_cutoff() -> None:
    """Callers with no league to hand still get the error and association metrics."""
    m = backtest_mod.projection_accuracy(_projection(IDS, [ACTUAL[i] for i in IDS]),
                                         ACTUAL)
    assert m["spearman"] == 1.0
    assert math.isnan(m["precision_at_k"])
    assert math.isnan(m["ndcg_at_k"])
