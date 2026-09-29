"""Independent mathematical unit tests for numerical stability and invariants.

CRITICAL NOTE (FALSIFY-05 Remediation):
These tests independently verify mathematical and numerical behavior under
boundary conditions, extreme values, and asymptotic limits. They are completely
decoupled from committed research results and file snapshots.
"""

from __future__ import annotations

import numpy as np
import pytest

from bayesian_lora_portfolio_risk.uncertainty.evaluation import (
    allocation_metrics,
    confidence_uncertainty,
    effective_n,
    fit_incremental_meta_model,
    match_effective_n,
)


class TestMathematicalNumericalStability:
    """Independent unit tests for mathematical functions and edge cases."""

    def test_confidence_uncertainty_extreme_probabilities(self) -> None:
        """Verify confidence uncertainty across [0, 1] including float precision limits."""
        p_extreme = np.array([0.0, 1.0, 1e-15, 1.0 - 1e-15, 0.5])
        conf = confidence_uncertainty(p_extreme)
        assert np.all(np.isfinite(conf))
        assert conf[0] == 0.0
        assert conf[1] == 0.0
        assert conf[-1] == 1.0
        assert 0.0 <= conf[2] <= 1.0
        assert 0.0 <= conf[3] <= 1.0

    def test_effective_n_near_zero_weights(self) -> None:
        """Verify effective N behavior when single weight dominates."""
        tiny_weights = np.array([1e-15, 1e-15, 1.0])
        neff = effective_n(tiny_weights)
        assert neff == pytest.approx(1.0, abs=1e-5)

    def test_effective_n_uniform_scaling(self) -> None:
        """Verify scale invariance of effective N."""
        w = np.array([1.0, 2.0, 3.0, 4.0])
        neff_base = effective_n(w)
        neff_scaled = effective_n(w * 1e8)
        assert neff_scaled == pytest.approx(neff_base, abs=1e-10)

    def test_allocation_metrics_zero_moves(self) -> None:
        """Verify allocation metrics with zero asset returns."""
        y = np.array([0, 1])
        moves = np.array([0.0, 0.0])
        m = allocation_metrics(y, moves, np.array([1.0, 1.0]))
        assert m["weighted_mean_move_1d"] == 0.0
        assert m["weighted_mean_downside_loss"] == 0.0
        assert m["effective_n"] == pytest.approx(2.0)

    def test_match_effective_n_identical_scores(self) -> None:
        """Verify match_effective_n handles flat score vectors stably."""
        raw_flat = np.ones(5)
        matched, gamma = match_effective_n(raw_flat, 3.0)
        assert np.all(np.isfinite(matched))
        assert len(matched) == 5
        assert np.sum(matched) == pytest.approx(5.0)

    def test_incremental_meta_model_probabilities_in_unit_interval(self) -> None:
        """Verify meta-model outputs valid AUROC and log-loss values."""
        np.random.seed(12345)
        n = 150
        conf_tr = np.random.uniform(0.1, 0.9, n)
        epi_tr = np.random.uniform(1e-5, 1e-4, n)
        y_tr = (conf_tr + np.random.normal(0, 0.1, n) > 0.5).astype(int)

        conf_te = np.random.uniform(0.1, 0.9, n)
        epi_te = np.random.uniform(1e-5, 1e-4, n)
        y_te = (conf_te + np.random.normal(0, 0.1, n) > 0.5).astype(int)

        res = fit_incremental_meta_model(conf_tr, epi_tr, y_tr, conf_te, epi_te, y_te)
        assert 0.0 <= res["confidence_only_auroc"] <= 1.0
        assert 0.0 <= res["confidence_plus_epistemic_auroc"] <= 1.0
        assert res["confidence_only_log_loss"] >= 0.0
        assert res["confidence_plus_epistemic_log_loss"] >= 0.0
        assert np.isclose(
            res["delta_auroc"],
            res["confidence_plus_epistemic_auroc"] - res["confidence_only_auroc"],
        )
