"""Core economic and uncertainty evaluation metrics for portfolio risk."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score


def confidence_uncertainty(probability: np.ndarray | float) -> np.ndarray:
    """Return deterministic confidence uncertainty score in [0, 1].

    Equal to 0 for p in {0, 1} and 1 for p = 0.5.
    Formula: 1 - 2 * |p - 0.5|
    """
    p = np.asarray(probability, dtype=float)
    return 1.0 - 2.0 * np.abs(p - 0.5)


def empirical_cdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Return empirical cumulative distribution function percentile P(ref <= val).

    Computed strictly using reference samples without using evaluation outcomes.
    """
    ref = np.sort(np.asarray(reference, dtype=float))
    if len(ref) == 0:
        raise ValueError("reference sample cannot be empty")
    return np.searchsorted(ref, np.asarray(values, dtype=float), side="right") / len(ref)


def effective_n(weights: np.ndarray) -> float:
    """Compute effective number of positions (inverse Herfindahl-Hirschman index).

    Formula: 1 / sum((w / sum(w))^2)
    """
    w = np.asarray(weights, dtype=float)
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("weights sum must be strictly positive")
    norm = w / total
    return float(1.0 / np.sum(norm**2))


def normalize_same_gross(raw: np.ndarray) -> np.ndarray:
    """Normalize raw weights so that their sum equals len(raw) (mean weight 1.0).

    Negative entries are clipped to 0.0.
    """
    raw_arr = np.maximum(np.asarray(raw, dtype=float), 0.0)
    total = float(raw_arr.sum())
    if total <= 0:
        raise ValueError("allocation rule produced zero gross exposure")
    return raw_arr * (len(raw_arr) / total)


def powered_weights(raw: np.ndarray, gamma: float) -> np.ndarray:
    """Raise positive weights to power gamma and gross-normalize."""
    x = np.maximum(np.asarray(raw, dtype=float), 1e-15)
    return normalize_same_gross(np.power(x, gamma))


def match_effective_n(
    raw: np.ndarray,
    target_effective_n: float,
    *,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> tuple[np.ndarray, float]:
    """Find power gamma by bisection to match target effective N.

    Preserves relative ranking while adjusting portfolio concentration.
    No forward evaluation outcomes are used.
    """
    raw_arr = np.asarray(raw, dtype=float)
    n = len(raw_arr)
    target = float(np.clip(target_effective_n, 1.0, float(n)))

    lo = 1e-4
    hi = 1.0
    neff_lo = effective_n(powered_weights(raw_arr, lo))
    neff_hi = effective_n(powered_weights(raw_arr, hi))

    while neff_hi > target and hi < 1024.0:
        hi *= 2.0
        neff_hi = effective_n(powered_weights(raw_arr, hi))

    if target >= neff_lo:
        return powered_weights(raw_arr, lo), lo

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        neff_mid = effective_n(powered_weights(raw_arr, mid))
        if abs(neff_mid - target) <= tol:
            return powered_weights(raw_arr, mid), mid
        if neff_mid > target:
            lo = mid
        else:
            hi = mid

    gamma = 0.5 * (lo + hi)
    return powered_weights(raw_arr, gamma), gamma


def allocation_metrics(
    y_true: np.ndarray,
    move_1d: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    """Calculate portfolio performance and downside risk metrics given weights."""
    y = np.asarray(y_true, dtype=float)
    move = np.asarray(move_1d, dtype=float)
    w = np.asarray(weights, dtype=float)
    total_w = float(w.sum())
    if total_w <= 0.0:
        raise ValueError("total weights must be strictly positive")
    norm = w / total_w
    downside_loss = np.maximum(-move, 0.0)

    return {
        "weighted_downside_event_rate": float(np.dot(norm, y)),
        "weighted_mean_move_1d": float(np.dot(norm, move)),
        "weighted_mean_downside_loss": float(np.dot(norm, downside_loss)),
        "effective_n": effective_n(w),
        "max_normalized_weight": float(np.max(norm)),
    }


def fit_incremental_meta_model(
    train_conf: np.ndarray,
    train_epi: np.ndarray,
    train_error: np.ndarray,
    test_conf: np.ndarray,
    test_epi: np.ndarray,
    test_error: np.ndarray,
) -> dict[str, float]:
    """Fit confidence-only and confidence+epistemic logistic error models.

    Reports out-of-sample AUROC, log-loss, and their deltas.
    """
    y_train = np.asarray(train_error, dtype=int)
    y_test = np.asarray(test_error, dtype=int)

    if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
        raise ValueError(
            "Both train and test sets must contain positive and negative error classes"
        )

    conf_tr = np.asarray(train_conf, dtype=float).reshape(-1, 1)
    conf_te = np.asarray(test_conf, dtype=float).reshape(-1, 1)
    log_epi_tr = np.log(
        np.maximum(np.asarray(train_epi, dtype=float), np.finfo(float).tiny)
    ).reshape(-1, 1)
    log_epi_te = np.log(
        np.maximum(np.asarray(test_epi, dtype=float), np.finfo(float).tiny)
    ).reshape(-1, 1)

    aug_tr = np.column_stack([conf_tr, log_epi_tr])
    aug_te = np.column_stack([conf_te, log_epi_te])

    m_base = LogisticRegression(max_iter=1000, solver="lbfgs")
    m_aug = LogisticRegression(max_iter=1000, solver="lbfgs")

    m_base.fit(conf_tr, y_train)
    m_aug.fit(aug_tr, y_train)

    p_base = m_base.predict_proba(conf_te)[:, 1]
    p_aug = m_aug.predict_proba(aug_te)[:, 1]

    base_auc = float(roc_auc_score(y_test, p_base))
    aug_auc = float(roc_auc_score(y_test, p_aug))
    base_ll = float(log_loss(y_test, p_base))
    aug_ll = float(log_loss(y_test, p_aug))

    return {
        "confidence_only_auroc": base_auc,
        "confidence_plus_epistemic_auroc": aug_auc,
        "delta_auroc": aug_auc - base_auc,
        "confidence_only_log_loss": base_ll,
        "confidence_plus_epistemic_log_loss": aug_ll,
        "delta_log_loss": aug_ll - base_ll,
    }


def define_unseen_companies(
    train_symbols: set[str],
    eval_symbols: pd.Series,
) -> pd.Series:
    """Mark symbols that do not appear in pre-period training symbols."""
    return ~eval_symbols.isin(train_symbols)


def define_sparse_history_companies(
    pre_counts: dict[str, int],
    eval_symbols: pd.Series,
    threshold: int = 4,
) -> pd.Series:
    """Mark symbols having count <= threshold in pre-period history."""
    counts = eval_symbols.map(lambda s: pre_counts.get(s, 0))
    return counts <= threshold


def define_underrepresented_sectors(
    pre_sector_counts: pd.Series,
    eval_sectors: pd.Series,
    quantile: float = 0.25,
) -> tuple[pd.Series, set[str]]:
    """Determine underrepresented sectors at or below frequency quantile."""
    non_empty = pre_sector_counts[pre_sector_counts.index != ""]
    thresh = float(non_empty.quantile(quantile))
    underrepresented = set(non_empty[non_empty <= thresh].index)
    is_underrepresented = eval_sectors.isin(underrepresented)
    return is_underrepresented, underrepresented
