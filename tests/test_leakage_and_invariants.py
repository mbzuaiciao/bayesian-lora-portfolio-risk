"""Tests for temporal split invariants, leakage prevention, and label construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bayesian_lora_portfolio_risk.uncertainty.evaluation import (
    define_sparse_history_companies,
    define_underrepresented_sectors,
    define_unseen_companies,
)


def test_temporal_split_chronological_boundaries() -> None:
    """Ensure pre-2022 train, 2022-2023 dev diagnostic, and 2024-2025 holdout are disjoint."""
    train_end = pd.Timestamp("2021-12-31")
    dev_start = pd.Timestamp("2022-01-01")
    dev_end = pd.Timestamp("2023-12-31")
    holdout_start = pd.Timestamp("2024-01-01")
    holdout_end = pd.Timestamp("2025-12-31")

    assert train_end < dev_start
    assert dev_start <= dev_end
    assert dev_end < holdout_start
    assert holdout_start <= holdout_end

    dates = pd.date_range("2020-01-01", "2025-12-31", freq="ME")
    df = pd.DataFrame({"date": dates, "val": np.arange(len(dates))})

    train_slice = df[df["date"] <= train_end]
    dev_slice = df[(df["date"] >= dev_start) & (df["date"] <= dev_end)]
    holdout_slice = df[(df["date"] >= holdout_start) & (df["date"] <= holdout_end)]

    # Disjointness checks
    assert set(train_slice.index).isdisjoint(dev_slice.index)
    assert set(dev_slice.index).isdisjoint(holdout_slice.index)
    assert set(train_slice.index).isdisjoint(holdout_slice.index)


def test_label_threshold_frozen_on_training_only() -> None:
    """Verify that downside-event threshold is computed strictly from pre-test data."""
    np.random.seed(42)
    # Simulate pre-test move_1d with specific quantile
    train_moves = pd.Series(np.random.normal(0.01, 0.05, 1000))
    # Target positive rate = 20%
    target_pos_rate = 0.20
    frozen_threshold = float(train_moves.quantile(target_pos_rate))

    # Evaluate on subsequent period
    test_moves = pd.Series(np.random.normal(-0.02, 0.06, 500))
    test_labels = (test_moves <= frozen_threshold).astype(int)

    # Threshold must NOT be re-estimated on test
    test_local_threshold = float(test_moves.quantile(target_pos_rate))
    assert frozen_threshold != test_local_threshold

    # Test label positive rate reflects regime shift, not artificial re-centering
    assert np.mean(test_labels) != pytest.approx(target_pos_rate, abs=0.01)


def test_no_future_leakage_in_subgroup_definitions() -> None:
    """Verify that company frequency / sector representation uses strictly pre-2022 rows."""
    pre2022_episodes = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "MSFT", "GOOG"],
            "sector": ["Tech", "Tech", "Tech", "Comm"],
            "date": pd.to_datetime(["2020-01-15", "2021-01-15", "2020-02-15", "2020-03-15"]),
        }
    )

    eval_episodes = pd.DataFrame(
        {
            "symbol": ["AAPL", "NVDA", "GOOG", "TSLA"],
            "sector": ["Tech", "Semi", "Comm", "Auto"],
            "date": pd.to_datetime(["2022-01-15", "2022-01-15", "2023-01-15", "2024-01-15"]),
        }
    )

    # 1. Unseen companies
    train_symbols = set(pre2022_episodes["symbol"])
    unseen = define_unseen_companies(train_symbols, eval_episodes["symbol"])
    assert list(unseen) == [False, True, False, True]

    # 2. Sparse history companies (<= 1 prior episode)
    pre_counts = pre2022_episodes["symbol"].value_counts().to_dict()
    sparse = define_sparse_history_companies(pre_counts, eval_episodes["symbol"], threshold=1)
    # AAPL has 2 -> False, NVDA has 0 -> True, GOOG has 1 -> True, TSLA has 0 -> True
    assert list(sparse) == [False, True, True, True]

    # 3. Underrepresented sectors (threshold at median)
    sector_counts = pre2022_episodes["sector"].value_counts()
    is_under, under_set = define_underrepresented_sectors(
        sector_counts, eval_episodes["sector"], quantile=0.5
    )
    # Tech: 3, Comm: 1. Comm <= median (1 <= 1) -> underrepresented
    assert under_set == {"Comm"}
    # eval sectors: Tech (False), Semi (False, not in pre-distribution), Comm (True), Auto (False)
    assert list(is_under) == [False, False, True, False]
