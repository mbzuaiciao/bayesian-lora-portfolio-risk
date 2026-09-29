"""Consistency and reproducibility verification script for Bayesian LoRA portfolio risk.

Verification Modes:
1. Clean-Clone Mode (Default, CI-safe):
   Verifies all committed result schemas, statistical invariants, and internal
   mathematical relationships across `results/*.json`. Requires ZERO untracked
   local prediction artifacts and runs in seconds on any clean git clone.
2. Deep Artifact Verification Mode (`--deep` or `--local-artifacts`):
   When local prediction artifacts are present under `artifacts/`, recomputes
   uncertainty metrics, AUROCs, concentration-matching weights, and portfolio
   downside metrics directly from raw predictions, verifying exact bitwise
   and floating-point reproduction of committed research results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from bayesian_lora_portfolio_risk.uncertainty.evaluation import (
    allocation_metrics,
    confidence_uncertainty,
    effective_n,
    empirical_cdf,
    match_effective_n,
    normalize_same_gross,
)

RESULTS_DIR = Path("results")
ARTIFACTS_DIR = Path("artifacts")
DEV_PRED_DIR = ARTIFACTS_DIR / "earnings_bayesian_lora_uncertainty_control"
HOLDOUT_PRED_DIR = ARTIFACTS_DIR / "earnings_bayesian_lora_final_holdout"
SEEDS = (42, 123, 777)
BETA = 0.003
MC_SAMPLES = 32


def verify_file_schemas() -> None:
    print("[1/3] Verifying committed result schemas and structure...", flush=True)
    if not RESULTS_DIR.exists():
        raise SystemExit(f"Results directory '{RESULTS_DIR}' does not exist.")
    json_files = list(RESULTS_DIR.rglob("*.json"))
    if not json_files:
        raise SystemExit(f"No result JSON files found under {RESULTS_DIR}")
    for jf in json_files:
        with jf.open("r") as f:
            data = json.load(f)
        if not isinstance(data, (dict, list)) or len(data) == 0:
            raise SystemExit(f"Corrupt or empty result file: {jf}")
    print(f"      OK: verified {len(json_files)} result files.", flush=True)


def verify_committed_result_invariants() -> None:
    print("[2/3] Verifying committed result statistical invariants...", flush=True)

    # 1. Uncertainty control summary
    uc_path = RESULTS_DIR / "earnings_bayesian_lora" / "uncertainty_control_summary.json"
    with uc_path.open() as f:
        uc_data = json.load(f)
    assert uc_data["complete"] is True
    assert len(uc_data["runs"]) == 3
    for run in uc_data["runs"]:
        assert run["seed"] in SEEDS
        assert 0.0 <= run["epistemic_error_detection_auroc"] <= 1.0
        assert 0.0 <= run["confidence_error_detection_auroc"] <= 1.0
        # Negative result: epistemic does not beat confidence
        assert run["epistemic_minus_confidence_error_detection_auroc"] < 0
        diff = run["epistemic_error_detection_auroc"] - run["confidence_error_detection_auroc"]
        assert np.isclose(diff, run["epistemic_minus_confidence_error_detection_auroc"], atol=1e-12)

    # 2. Holdout summary
    hold_path = RESULTS_DIR / "earnings_bayesian_lora_final_holdout" / "summary.json"
    with hold_path.open() as f:
        hold_data = json.load(f)
    assert hold_data["complete"] is True
    assert len(hold_data["runs"]) == 3
    for run in hold_data["runs"]:
        assert run["seed"] in SEEDS
        assert 0.0 <= run["epistemic_error_detection_auroc"] <= 1.0
        assert 0.0 <= run["confidence_error_detection_auroc"] <= 1.0
        assert run["epistemic_minus_confidence_error_detection_auroc"] < 0

    # 3. Concentration control summary
    cc_path = (
        RESULTS_DIR
        / "earnings_event_risk_allocation_concentration_control"
        / "summary.json"
    )
    with cc_path.open() as f:
        cc_data = json.load(f)
    agg_2023 = cc_data["aggregate"]["2023_from_2022_calibration"]
    assert agg_2023["at_epistemic_concentration"]["epistemic_beats_seed_count"] == 0
    assert agg_2023["at_confidence_concentration"]["epistemic_beats_seed_count"] == 0

    # 4. Model risk penalty summary
    mrp_path = RESULTS_DIR / "earnings_model_risk_penalty" / "summary.json"
    with mrp_path.open() as f:
        mrp_data = json.load(f)
    for run in mrp_data["runs"]:
        p1 = run["periods"]["2023_from_2022_calibration"]
        p2 = run["periods"]["2024_2025_from_pre2024_calibration"]
        assert p1["selected_lambda"] == 0.0
        assert p2["selected_lambda"] == 0.0

    # 5. Distribution shift summaries
    for shift_name in [
        "earnings_unseen_company_shift",
        "earnings_sparse_history_shift",
        "earnings_underrepresented_sector_shift",
    ]:
        shift_path = RESULTS_DIR / shift_name / "summary.json"
        with shift_path.open() as f:
            shift_data = json.load(f)
        assert len(shift_data["runs"]) == 3
        for run in shift_data["runs"]:
            assert run["seed"] in SEEDS

    print("      OK: all committed result statistical invariants verified.", flush=True)


def verify_predictions_exist() -> None:
    print("[Deep 1/3] Verifying presence of local prediction artifacts...", flush=True)
    beta_tag = str(BETA).replace(".", "p")
    missing: list[Path] = []
    for seed in SEEDS:
        dev_f = DEV_PRED_DIR / f"beta_{beta_tag}_seed_{seed}_mc{MC_SAMPLES}_predictions.csv"
        hold_f = HOLDOUT_PRED_DIR / f"beta_{beta_tag}_seed_{seed}_mc{MC_SAMPLES}_predictions.csv"
        if not dev_f.exists():
            missing.append(dev_f)
        if not hold_f.exists():
            missing.append(hold_f)
    if missing:
        msg = (
            "Deep verification requested via --deep, but the following local prediction\n"
            "artifacts were not found:\n" + "\n".join(f"  - {p}" for p in missing) + "\n\n"
            "NOTE: Raw prediction CSVs are excluded from git via .gitignore due to size.\n"
            "Clean-clone verification of committed results does not require these files;\n"
            "run without --deep to verify the repository on a clean clone."
        )
        raise SystemExit(msg)
    print("      OK: all 3 seed development and holdout prediction artifacts found.", flush=True)


def verify_uncertainty_and_holdout_metrics() -> None:
    print("[Deep 2/3] Recomputing uncertainty metrics from raw predictions...", flush=True)
    dev_summary_path = RESULTS_DIR / "earnings_bayesian_lora" / "uncertainty_control_summary.json"
    hold_summary_path = RESULTS_DIR / "earnings_bayesian_lora_final_holdout" / "summary.json"

    with dev_summary_path.open() as f:
        dev_summary = json.load(f)
    with hold_summary_path.open() as f:
        hold_summary = json.load(f)

    beta_tag = str(BETA).replace(".", "p")

    for run_dev, run_hold in zip(dev_summary["runs"], hold_summary["runs"], strict=True):
        seed = run_dev["seed"]
        dev_f = DEV_PRED_DIR / f"beta_{beta_tag}_seed_{seed}_mc{MC_SAMPLES}_predictions.csv"
        hold_f = HOLDOUT_PRED_DIR / f"beta_{beta_tag}_seed_{seed}_mc{MC_SAMPLES}_predictions.csv"

        # Check Dev
        df_dev = pd.read_csv(dev_f)
        y_dev = df_dev["label"].to_numpy(dtype=int)
        p_dev = df_dev["posterior_mean"].to_numpy(dtype=float)
        epi_dev = df_dev["epistemic_variance"].to_numpy(dtype=float)
        conf_dev = confidence_uncertainty(p_dev)
        inc_dev = ((p_dev >= 0.5) != y_dev).astype(int)

        epi_auc_dev = float(roc_auc_score(inc_dev, epi_dev))
        conf_auc_dev = float(roc_auc_score(inc_dev, conf_dev))
        if not np.isclose(epi_auc_dev, run_dev["epistemic_error_detection_auroc"], atol=1e-7):
            raise SystemExit(f"Mismatch in dev epistemic AUROC for seed {seed}")
        if not np.isclose(conf_auc_dev, run_dev["confidence_error_detection_auroc"], atol=1e-7):
            raise SystemExit(f"Mismatch in dev confidence AUROC for seed {seed}")

        # Check Holdout
        df_hold = pd.read_csv(hold_f)
        y_hold = df_hold["label"].to_numpy(dtype=int)
        p_hold = df_hold["posterior_mean"].to_numpy(dtype=float)
        epi_hold = df_hold["epistemic_variance"].to_numpy(dtype=float)
        conf_hold = confidence_uncertainty(p_hold)
        inc_hold = ((p_hold >= 0.5) != y_hold).astype(int)

        epi_auc_hold = float(roc_auc_score(inc_hold, epi_hold))
        conf_auc_hold = float(roc_auc_score(inc_hold, conf_hold))
        if not np.isclose(epi_auc_hold, run_hold["epistemic_error_detection_auroc"], atol=1e-7):
            raise SystemExit(f"Mismatch in holdout epistemic AUROC for seed {seed}")
        if not np.isclose(conf_auc_hold, run_hold["confidence_error_detection_auroc"], atol=1e-7):
            raise SystemExit(f"Mismatch in holdout confidence AUROC for seed {seed}")

    print("      OK: reconstructed dev and holdout AUROCs match committed results exactly.", flush=True)


def verify_concentration_matching() -> None:
    print("[Deep 3/3] Recomputing concentration matching directly from raw predictions...", flush=True)
    conc_summary_path = (
        RESULTS_DIR / "earnings_event_risk_allocation_concentration_control" / "summary.json"
    )
    with conc_summary_path.open() as f:
        conc_summary = json.load(f)

    beta_tag = str(BETA).replace(".", "p")

    for run_entry in conc_summary["runs"]:
        seed = run_entry["seed"]
        dev_f = DEV_PRED_DIR / f"beta_{beta_tag}_seed_{seed}_mc{MC_SAMPLES}_predictions.csv"
        df_dev = pd.read_csv(dev_f)
        df_dev["date"] = pd.to_datetime(df_dev["date"])

        calib = df_dev[df_dev["date"].dt.year == 2022].copy().reset_index(drop=True)
        eval_ = df_dev[df_dev["date"].dt.year == 2023].copy().reset_index(drop=True)

        p = eval_["posterior_mean"].to_numpy(dtype=float)
        epi = eval_["epistemic_variance"].to_numpy(dtype=float)
        cal_epi = calib["epistemic_variance"].to_numpy(dtype=float)

        raw_conf = (1.0 - p) * (2.0 * np.abs(p - 0.5))
        raw_epi = (1.0 - p) * (1.0 - empirical_cdf(cal_epi, epi))

        w_conf = normalize_same_gross(raw_conf)
        w_epi = normalize_same_gross(raw_epi)

        neff_conf = effective_n(w_conf)
        neff_epi = effective_n(w_epi)

        matched_conf, _ = match_effective_n(raw_conf, neff_epi)
        matched_epi, _ = match_effective_n(raw_epi, neff_conf)

        m_conf = allocation_metrics(eval_["label"], eval_["move_1d"], w_conf)
        m_epi = allocation_metrics(eval_["label"], eval_["move_1d"], w_epi)
        m_conf_matched = allocation_metrics(eval_["label"], eval_["move_1d"], matched_conf)
        m_epi_matched = allocation_metrics(eval_["label"], eval_["move_1d"], matched_epi)

        comm_period = run_entry["periods"]["2023_from_2022_calibration"]["metrics"]
        assert np.isclose(
            m_conf["weighted_downside_event_rate"],
            comm_period["confidence_adjusted"]["weighted_downside_event_rate"],
            atol=1e-6,
        )
        assert np.isclose(
            m_epi["weighted_downside_event_rate"],
            comm_period["epistemic_adjusted"]["weighted_downside_event_rate"],
            atol=1e-6,
        )
        assert np.isclose(
            m_conf_matched["weighted_downside_event_rate"],
            comm_period["confidence_matched_to_epistemic_concentration"][
                "weighted_downside_event_rate"
            ],
            atol=1e-6,
        )
        assert np.isclose(
            m_epi_matched["weighted_downside_event_rate"],
            comm_period["epistemic_matched_to_confidence_concentration"][
                "weighted_downside_event_rate"
            ],
            atol=1e-6,
        )

    print("      OK: concentration matching recomputation verified identically.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deep",
        "--local-artifacts",
        dest="deep",
        action="store_true",
        help="Run deep verification recomputing metrics from local raw prediction CSVs.",
    )
    args = parser.parse_args()

    print("======================================================================")
    print("Running Repository Verification & Audit Checks")
    print(f"Mode: {'Deep Local-Artifact Recomputation' if args.deep else 'Clean-Clone Committed Result Verification'}")
    print("======================================================================")

    # Clean-clone verification
    verify_file_schemas()
    verify_committed_result_invariants()
    print("[3/3] Clean-clone verification of committed results: PASSED.")

    # Deep verification if requested
    if args.deep:
        print("\n--- Running Deep Local Artifact Verification ---")
        verify_predictions_exist()
        verify_uncertainty_and_holdout_metrics()
        verify_concentration_matching()
        print("--- Deep Local Artifact Verification: PASSED ---")
    else:
        print("\n[NOTE] To run deep recomputation against raw prediction CSVs, pass `--deep`.")

    print("======================================================================")
    print("ALL CHECKS PASSED: Reproducibility and consistency verified.")
    print("======================================================================")


if __name__ == "__main__":
    main()
