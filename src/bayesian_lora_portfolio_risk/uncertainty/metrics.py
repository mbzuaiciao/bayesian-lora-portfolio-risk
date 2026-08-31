"""Metrics for calibration and uncertainty usefulness in Milestone 0."""

from __future__ import annotations

import numpy as np


def brier_score(y_true: np.ndarray, p_positive: np.ndarray) -> float:
    """Return the binary Brier score."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_positive, dtype=float)
    if y.shape != p.shape:
        raise ValueError("y_true and p_positive must have the same shape")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    y_true: np.ndarray,
    p_positive: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Compute binary expected calibration error using equal-width bins."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p_positive, dtype=float)
    if y.shape != p.shape:
        raise ValueError("y_true and p_positive must have the same shape")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")

    predicted = (p >= 0.5).astype(int)
    confidence = np.maximum(p, 1.0 - p)
    accuracy = (predicted == y).astype(float)

    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        left, right = edges[idx], edges[idx + 1]
        if idx == n_bins - 1:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)
        if not np.any(mask):
            continue
        weight = float(np.mean(mask))
        ece += weight * abs(float(np.mean(accuracy[mask])) - float(np.mean(confidence[mask])))
    return ece


def error_rate_by_uncertainty_quantile(
    correct: np.ndarray,
    uncertainty: np.ndarray,
    *,
    n_quantiles: int = 5,
) -> np.ndarray:
    """Return error rates from lowest- to highest-uncertainty quantile."""
    if n_quantiles <= 0:
        raise ValueError("n_quantiles must be positive")

    correct_arr = np.asarray(correct, dtype=bool)
    uncertainty_arr = np.asarray(uncertainty, dtype=float)
    if correct_arr.shape != uncertainty_arr.shape:
        raise ValueError("correct and uncertainty must have the same shape")
    if correct_arr.size == 0:
        return np.array([], dtype=float)

    order = np.argsort(uncertainty_arr, kind="stable")
    groups = np.array_split(order, n_quantiles)
    rates = []
    for group in groups:
        if group.size == 0:
            rates.append(np.nan)
        else:
            rates.append(float(np.mean(~correct_arr[group])))
    return np.asarray(rates, dtype=float)


def selective_risk_curve(
    correct: np.ndarray,
    uncertainty: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return coverage and error rate when retaining least-uncertain examples first."""
    correct_arr = np.asarray(correct, dtype=bool)
    uncertainty_arr = np.asarray(uncertainty, dtype=float)
    if correct_arr.shape != uncertainty_arr.shape:
        raise ValueError("correct and uncertainty must have the same shape")
    if correct_arr.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    order = np.argsort(uncertainty_arr, kind="stable")
    retained_correct = correct_arr[order]
    counts = np.arange(1, retained_correct.size + 1)
    coverage = counts / retained_correct.size
    errors = (~retained_correct).astype(float)
    risk = np.cumsum(errors) / counts
    return coverage.astype(float), risk.astype(float)
