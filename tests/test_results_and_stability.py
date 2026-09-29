"""Tests for result serialization, schema consistency, and numerical stability."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_lora_portfolio_risk.uncertainty.evaluation import (
    allocation_metrics,
    confidence_uncertainty,
    effective_n,
    match_effective_n,
)

RESULTS_DIR = Path("results")


def test_committed_results_schemas() -> None:
    """Validate structure and non-empty status of all committed JSON result files."""
    assert RESULTS_DIR.exists()
    json_files = list(RESULTS_DIR.rglob("*.json"))
    assert len(json_files) >= 10, f"Expected >= 10 result files, found {len(json_files)}"

    for jf in json_files:
        with jf.open("r") as f:
            data = json.load(f)
        assert isinstance(data, (dict, list)), f"File {jf} is not a valid JSON dict/list"
        assert len(data) > 0, f"File {jf} is empty"


def test_uncertainty_control_summary_schema() -> None:
    path = RESULTS_DIR / "earnings_bayesian_lora" / "uncertainty_control_summary.json"
    assert path.exists()
    with path.open() as f:
        data = json.load(f)
    assert data["complete"] is True
    assert len(data["runs"]) == 3
    for run in data["runs"]:
        assert "seed" in run
        assert "epistemic_error_detection_auroc" in run
        assert "confidence_error_detection_auroc" in run
        assert "epistemic_minus_confidence_error_detection_auroc" in run
        assert run["epistemic_minus_confidence_error_detection_auroc"] < 0  # Negative result


def test_concentration_control_summary_schema() -> None:
    path = (
        RESULTS_DIR
        / "earnings_event_risk_allocation_concentration_control"
        / "summary.json"
    )
    assert path.exists()
    with path.open() as f:
        data = json.load(f)
    assert "runs" in data
    assert len(data["runs"]) == 3
    assert "aggregate" in data
    # Epistemic should not beat confidence at either concentration matching regime
    agg_2023 = data["aggregate"]["2023_from_2022_calibration"]
    assert agg_2023["at_epistemic_concentration"]["epistemic_beats_seed_count"] == 0
    assert agg_2023["at_confidence_concentration"]["epistemic_beats_seed_count"] == 0


def test_model_risk_penalty_summary_schema() -> None:
    path = RESULTS_DIR / "earnings_model_risk_penalty" / "summary.json"
    assert path.exists()
    with path.open() as f:
        data = json.load(f)
    agg = data["aggregate"]
    assert agg["2023_from_2022_calibration"]["selected_lambdas"] == [0.0, 0.0, 0.0]
    assert agg["2024_2025_from_pre2024_calibration"]["selected_lambdas"] == [0.0, 0.0, 0.0]


def test_numerical_stability_edge_cases() -> None:
    """Test extreme probabilities, tiny variance, and large weight bounds."""
    # 1. Extreme probabilities for confidence uncertainty
    p_extreme = np.array([0.0, 1.0, 1e-15, 1.0 - 1e-15, 0.5])
    conf = confidence_uncertainty(p_extreme)
    assert np.all(np.isfinite(conf))
    assert conf[0] == 0.0
    assert conf[1] == 0.0
    assert conf[-1] == 1.0

    # 2. Near-zero weights in effective N
    tiny_weights = np.array([1e-15, 1e-15, 1.0])
    neff = effective_n(tiny_weights)
    assert neff == pytest.approx(1.0, abs=1e-5)

    # 3. Allocation metrics with zero moves
    y = np.array([0, 1])
    moves = np.array([0.0, 0.0])
    m = allocation_metrics(y, moves, np.array([1.0, 1.0]))
    assert m["weighted_mean_move_1d"] == 0.0
    assert m["weighted_mean_downside_loss"] == 0.0

    # 4. match_effective_n with identical raw scores (cannot concentrate further or flatter)
    raw_flat = np.ones(5)
    matched, gamma = match_effective_n(raw_flat, 3.0)
    assert np.all(np.isfinite(matched))
    assert len(matched) == 5
