"""Tests for uncertainty controls, concentration matching, and decision rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bayesian_lora_portfolio_risk.uncertainty.evaluation import (
    allocation_metrics,
    confidence_uncertainty,
    define_sparse_history_companies,
    define_underrepresented_sectors,
    define_unseen_companies,
    effective_n,
    empirical_cdf,
    fit_incremental_meta_model,
    match_effective_n,
    normalize_same_gross,
)


def test_confidence_uncertainty_bounds_and_symmetry() -> None:
    p = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    u = confidence_uncertainty(p)
    np.testing.assert_allclose(u, np.array([0.0, 0.5, 1.0, 0.5, 0.0]))
    assert np.all((u >= 0.0) & (u <= 1.0))
    # Scalar input
    assert float(confidence_uncertainty(0.5)) == 1.0
    assert float(confidence_uncertainty(1.0)) == 0.0


def test_empirical_cdf_invariants() -> None:
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    vals = np.array([0.5, 1.0, 2.5, 4.0, 5.0])
    pct = empirical_cdf(ref, vals)
    np.testing.assert_allclose(pct, np.array([0.0, 0.25, 0.5, 1.0, 1.0]))

    with pytest.raises(ValueError, match="empty"):
        empirical_cdf(np.array([]), np.array([1.0]))


def test_effective_n_extreme_cases() -> None:
    # Equal weights on N assets: effective N == N
    w_equal = np.ones(10)
    assert effective_n(w_equal) == pytest.approx(10.0)

    # All weight on 1 asset: effective N == 1
    w_single = np.zeros(10)
    w_single[0] = 5.0
    assert effective_n(w_single) == pytest.approx(1.0)

    # Invariant to constant scaling
    assert effective_n(w_equal * 100.0) == pytest.approx(10.0)

    with pytest.raises(ValueError):
        effective_n(np.zeros(5))


def test_normalize_same_gross_mean_is_one() -> None:
    raw = np.array([1.0, 2.0, 3.0, 4.0])
    norm = normalize_same_gross(raw)
    assert len(norm) == 4
    assert np.sum(norm) == pytest.approx(4.0)
    assert np.mean(norm) == pytest.approx(1.0)

    # Negative values clipped to 0
    raw_neg = np.array([-1.0, 1.0, 2.0])
    norm_neg = normalize_same_gross(raw_neg)
    assert norm_neg[0] == 0.0
    assert np.sum(norm_neg) == pytest.approx(3.0)


def test_match_effective_n_bisection() -> None:
    # Non-uniform raw scores
    raw = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    neff_raw = effective_n(raw)
    assert 1.0 < neff_raw < 5.0

    # Match to lower target (more concentrated, gamma > 1)
    target_low = 2.5
    matched_low, gamma_low = match_effective_n(raw, target_low)
    assert effective_n(matched_low) == pytest.approx(target_low, abs=1e-3)
    assert gamma_low > 1.0
    assert np.mean(matched_low) == pytest.approx(1.0)
    # Order must be preserved
    assert np.all(np.diff(matched_low) > 0)

    # Match to higher target (flatter, gamma < 1)
    target_high = 4.5
    matched_high, gamma_high = match_effective_n(raw, target_high)
    assert effective_n(matched_high) == pytest.approx(target_high, abs=1e-3)
    assert gamma_high < 1.0
    assert np.mean(matched_high) == pytest.approx(1.0)


def test_allocation_metrics_weighted_quantities() -> None:
    y = np.array([1, 0, 1, 0])
    moves = np.array([-0.05, 0.02, -0.10, 0.04])
    # Equal weights
    w = np.ones(4)
    m = allocation_metrics(y, moves, w)
    assert m["weighted_downside_event_rate"] == pytest.approx(0.5)
    assert m["weighted_mean_move_1d"] == pytest.approx((-0.05 + 0.02 - 0.10 + 0.04) / 4)
    # downside loss is max(-move, 0): (0.05 + 0 + 0.10 + 0) / 4 = 0.0375
    assert m["weighted_mean_downside_loss"] == pytest.approx(0.0375)
    assert m["effective_n"] == pytest.approx(4.0)


def test_fit_incremental_meta_model_synthetic() -> None:
    np.random.seed(42)
    n = 200
    # Simulate errors correlated with confidence and epistemic
    conf_train = np.random.uniform(0.1, 0.9, n)
    epi_train = np.random.uniform(1e-5, 1e-4, n)
    err_train = (conf_train > 0.5).astype(int)

    conf_test = np.random.uniform(0.1, 0.9, n)
    epi_test = np.random.uniform(1e-5, 1e-4, n)
    err_test = (conf_test > 0.5).astype(int)

    res = fit_incremental_meta_model(
        conf_train, epi_train, err_train,
        conf_test, epi_test, err_test
    )
    assert "delta_auroc" in res
    assert "delta_log_loss" in res
    assert 0.0 <= res["confidence_only_auroc"] <= 1.0
    assert 0.0 <= res["confidence_plus_epistemic_auroc"] <= 1.0


def test_subgroup_shift_definitions() -> None:
    train_symbols = {"AAPL", "MSFT", "GOOG"}
    eval_symbols = pd.Series(["AAPL", "NVDA", "MSFT", "AMZN"])
    unseen = define_unseen_companies(train_symbols, eval_symbols)
    assert unseen.tolist() == [False, True, False, True]

    pre_counts = {"AAPL": 10, "NVDA": 2, "MSFT": 5, "AMZN": 0}
    sparse = define_sparse_history_companies(pre_counts, eval_symbols, threshold=4)
    assert sparse.tolist() == [False, True, False, True]

    sector_counts = pd.Series({"Tech": 100, "Finance": 80, "Health": 60, "Energy": 10})
    eval_sectors = pd.Series(["Tech", "Energy", "Finance", "Energy"])
    is_under, under_set = define_underrepresented_sectors(
        sector_counts, eval_sectors, quantile=0.25
    )
    assert "Energy" in under_set
    assert "Tech" not in under_set
    assert is_under.tolist() == [False, True, False, True]
