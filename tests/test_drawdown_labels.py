import numpy as np
import pytest

from bayesian_lora_portfolio_risk.data.labels import (
    forward_drawdown_label,
    forward_max_drawdown,
)


def test_forward_max_drawdown_excludes_observation_day() -> None:
    prices = np.array([100.0, 95.0, 80.0, 110.0])
    drawdown = forward_max_drawdown(prices, horizon=2)

    assert drawdown[0] == pytest.approx(-0.20)
    assert drawdown[1] == pytest.approx(80.0 / 95.0 - 1.0)
    assert np.isnan(drawdown[2])
    assert np.isnan(drawdown[3])


def test_forward_drawdown_label_marks_material_event() -> None:
    prices = np.array([100.0, 95.0, 89.0, 105.0])
    labels = forward_drawdown_label(prices, horizon=2, threshold=0.10)

    np.testing.assert_allclose(labels[:2], np.array([1.0, 0.0]))
    assert np.isnan(labels[2])
    assert np.isnan(labels[3])


def test_forward_drawdown_label_requires_complete_future_window() -> None:
    prices = np.array([100.0, 101.0, 102.0])
    labels = forward_drawdown_label(prices, horizon=3, threshold=0.10)
    assert np.all(np.isnan(labels))


def test_invalid_threshold_is_rejected() -> None:
    with pytest.raises(ValueError):
        forward_drawdown_label(np.array([100.0, 90.0]), horizon=1, threshold=0.0)


def test_nonpositive_prices_are_rejected() -> None:
    with pytest.raises(ValueError):
        forward_max_drawdown(np.array([100.0, 0.0, 90.0]), horizon=1)
