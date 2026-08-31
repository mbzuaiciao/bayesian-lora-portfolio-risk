"""Forward-looking risk labels for Milestone 0.

The functions in this module intentionally use only prices strictly after the
observation index when constructing the future drawdown window.
"""

from __future__ import annotations

import numpy as np


def forward_max_drawdown(prices: np.ndarray, *, horizon: int = 20) -> np.ndarray:
    """Return forward maximum drawdown from each observation price.

    For observation t, the result is

        min_{1 <= k <= horizon} (P[t + k] / P[t] - 1).

    The current-day return (k=0) is excluded by construction. Entries without a
    complete forward window are returned as NaN.
    """
    values = np.asarray(prices, dtype=float)
    if values.ndim != 1:
        raise ValueError("prices must be one-dimensional")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if np.any(~np.isfinite(values)):
        raise ValueError("prices must be finite")
    if np.any(values <= 0.0):
        raise ValueError("prices must be strictly positive")

    result = np.full(values.shape, np.nan, dtype=float)
    if values.size <= horizon:
        return result

    for idx in range(values.size - horizon):
        future = values[idx + 1 : idx + horizon + 1]
        future_returns = future / values[idx] - 1.0
        result[idx] = float(np.min(future_returns))
    return result


def forward_drawdown_label(
    prices: np.ndarray,
    *,
    horizon: int = 20,
    threshold: float = 0.10,
) -> np.ndarray:
    """Return binary labels for a material future drawdown.

    A valid observation receives 1 when its forward maximum drawdown is less
    than or equal to ``-threshold`` and 0 otherwise. Observations without a
    complete forward window are returned as NaN so callers cannot accidentally
    treat censored examples as negatives.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie strictly between 0 and 1")

    drawdown = forward_max_drawdown(prices, horizon=horizon)
    labels = np.full(drawdown.shape, np.nan, dtype=float)
    valid = np.isfinite(drawdown)
    labels[valid] = (drawdown[valid] <= -threshold).astype(float)
    return labels
