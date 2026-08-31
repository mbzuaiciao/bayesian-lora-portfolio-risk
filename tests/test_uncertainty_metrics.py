import numpy as np

from bayesian_lora_portfolio_risk.uncertainty.metrics import (
    brier_score,
    error_rate_by_uncertainty_quantile,
    expected_calibration_error,
    selective_risk_curve,
)


def test_brier_score_perfect_predictions() -> None:
    y = np.array([0, 1, 1, 0])
    p = np.array([0.0, 1.0, 1.0, 0.0])
    assert brier_score(y, p) == 0.0


def test_expected_calibration_error_is_zero_for_perfect_predictions() -> None:
    y = np.array([0, 1, 1, 0])
    p = np.array([0.0, 1.0, 1.0, 0.0])
    assert expected_calibration_error(y, p) == 0.0


def test_error_rate_rises_with_constructed_uncertainty() -> None:
    correct = np.array([True, True, True, False, False, False])
    uncertainty = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    rates = error_rate_by_uncertainty_quantile(correct, uncertainty, n_quantiles=2)
    np.testing.assert_allclose(rates, np.array([0.0, 1.0]))


def test_selective_risk_prefers_low_uncertainty_examples() -> None:
    correct = np.array([True, True, False, False])
    uncertainty = np.array([0.1, 0.2, 0.8, 0.9])
    coverage, risk = selective_risk_curve(correct, uncertainty)

    np.testing.assert_allclose(coverage, np.array([0.25, 0.5, 0.75, 1.0]))
    np.testing.assert_allclose(risk, np.array([0.0, 0.0, 1 / 3, 0.5]))
