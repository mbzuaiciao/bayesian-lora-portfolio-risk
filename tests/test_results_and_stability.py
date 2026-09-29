"""Regression snapshot and schema validation for committed research result files.

CRITICAL NOTE (FALSIFY-05 Remediation):
These tests verify that frozen historical results committed under `results/`
maintain schema integrity and match published paper numbers. They serve as
regression snapshot guards and artifact invariants, NOT independent functional
proofs of algorithmic correctness.

Independent mathematical and algorithmic unit tests are strictly decoupled and
located in:
- `tests/test_numerical_stability.py` (boundary conditions, precision limits, mathematical properties)
- `tests/test_evaluation_metrics.py` (metric formulas, weighting, concentration matching)
- `tests/test_bayesian_lora.py` (variational layers, ELBO, reparameterization)
- `tests/test_leakage_and_invariants.py` (temporal ordering, label boundaries)
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path("results")


def test_committed_results_schemas() -> None:
    """Validate structure and non-empty status of all committed JSON result files."""
    assert RESULTS_DIR.exists(), f"Results directory {RESULTS_DIR} must exist"
    json_files = list(RESULTS_DIR.rglob("*.json"))
    assert len(json_files) >= 10, f"Expected >= 10 result files, found {len(json_files)}"

    for jf in json_files:
        with jf.open("r") as f:
            data = json.load(f)
        assert isinstance(data, (dict, list)), f"File {jf} is not a valid JSON dict/list"
        assert len(data) > 0, f"File {jf} is empty"


def test_uncertainty_control_summary_schema() -> None:
    """Verify frozen development diagnostic summary schema."""
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
    """Verify frozen concentration-controlled portfolio summary schema."""
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
    agg_2023 = data["aggregate"]["2023_from_2022_calibration"]
    assert agg_2023["at_epistemic_concentration"]["epistemic_beats_seed_count"] == 0
    assert agg_2023["at_confidence_concentration"]["epistemic_beats_seed_count"] == 0


def test_model_risk_penalty_summary_schema() -> None:
    """Verify frozen model risk penalty summary schema and lambda selection."""
    path = RESULTS_DIR / "earnings_model_risk_penalty" / "summary.json"
    assert path.exists()
    with path.open() as f:
        data = json.load(f)
    agg = data["aggregate"]
    assert agg["2023_from_2022_calibration"]["selected_lambdas"] == [0.0, 0.0, 0.0]
    assert agg["2024_2025_from_pre2024_calibration"]["selected_lambdas"] == [0.0, 0.0, 0.0]
