"""Lightweight consistency and verification script for Bayesian LoRA portfolio risk.

Establish internal consistency across all committed results, metadata,
and local prediction artifacts without rerunning expensive GPU model training.
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
    print("[1/5] Verifying committed result schemas...", flush=True)
    json_files = list(RESULTS_DIR.rglob("*.json"))
    if not json_files:
        raise SystemExit(f"No result json files found under {RESULTS_DIR}")
    for jf in json_files:
        with jf.open("r") as f:
            data = json.load(f)
        if not isinstance(data, (dict, list)) or len(data) == 0:
            raise SystemExit(f"Corrupt or empty result file: {jf}")
    print(f"      OK: verified {len(json_files)} result files.", flush=True)


def verify_predictions_exist() -> None:
    print("[2/5] Verifying presence of prediction artifacts...", flush=True)
    beta_tag = str(BETA).replace(".", "p")
    for seed in SEEDS:
        dev_f = DEV_PRED_DIR / f"beta_{beta_tag}_seed_{seed}_mc{MC_SAMPLES}_predictions.csv"
        hold_f = HOLDOUT_PRED_DIR / f"beta_{beta_tag}_seed_{seed}_mc{MC_SAMPLES}_predictions.csv"
        if not dev_f.exists():
            raise SystemExit(f"Missing development predictions: {dev_f}")
        if not hold_f.exists():
            raise SystemExit(f"Missing holdout predictions: {hold_f}")
    print("      OK: all 3 seed development and holdout prediction artifacts found.", flush=True)


def verify_uncertainty_and_holdout_metrics() -> None:
    print("[3/5] Verifying uncertainty control and holdout error metrics...", flush=True)
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

    print("      OK: reconstructed dev and holdout AUROCs match committed results.", flush=True)


def verify_concentration_matching() -> None:
    print("[4/5] Verifying concentration-matching portfolio metrics...", flush=True)
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

        # Period 2023 from 2022 calibration
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

    print("      OK: concentration matching calculations verified identically.", flush=True)


def verify_model_risk_penalty_selection() -> None:
    print("[5/5] Verifying model-risk penalty lambda selection...", flush=True)
    mrp_summary_path = RESULTS_DIR / "earnings_model_risk_penalty" / "summary.json"
    with mrp_summary_path.open() as f:
        mrp_summary = json.load(f)

    for run_entry in mrp_summary["runs"]:
        p1 = run_entry["periods"]["2023_from_2022_calibration"]
        p2 = run_entry["periods"]["2024_2025_from_pre2024_calibration"]
        if p1["selected_lambda"] != 0.0 or p2["selected_lambda"] != 0.0:
            raise SystemExit("Unexpected non-zero lambda in committed model risk penalty summary")

    print("      OK: model risk penalty lambda=0 verified across all runs and periods.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.parse_args()

    print("======================================================================")
    print("Running Repository Audit & Consistency Verification (CPU-only)")
    print("======================================================================")
    verify_file_schemas()
    verify_predictions_exist()
    verify_uncertainty_and_holdout_metrics()
    verify_concentration_matching()
    verify_model_risk_penalty_selection()
    print("======================================================================")
    print("ALL AUDIT CHECKS PASSED: Internal consistency confirmed.")
    print("======================================================================")


if __name__ == "__main__":
    main()
