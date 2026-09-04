"""Test Bayesian epistemic uncertainty as an incremental model-risk penalty.

Motivation
----------
Earlier analyses showed that epistemic uncertainty is associated with harder
predictions, but its direct allocation advantage disappears after controlling for
portfolio concentration. This script therefore tests a narrower question:

    Does epistemic uncertainty add value *on top of* ordinary classifier
    confidence when used as a separate model-risk penalty?

The baseline allocation is the confidence-adjusted rule

    confidence_raw = (1 - p) * 2*abs(p - 0.5)

The model-risk rule adds an exponential penalty based on the percentile rank of
Bayesian epistemic variance relative to a prior calibration period

    augmented_raw = confidence_raw * exp(-lambda * epistemic_percentile)

Lambda is selected using calibration-period outcomes only from the fixed grid
{0, 0.25, 0.5, 1, 2}. For each candidate lambda, the augmented calibration
weights are power-transformed to match the confidence rule's effective number of
positions, so lambda selection cannot win merely by changing concentration.

After lambda is selected, the same frozen lambda is applied to the forward
evaluation period. The augmented evaluation weights are again concentration-
matched to the confidence baseline using weights only (no evaluation outcomes).

Two temporal tests are reported:

1. 2022 calibration -> 2023 evaluation
2. 2022-2023 calibration -> 2024-2025 evaluation

Methodological status
---------------------
This is a post-hoc robustness analysis on already-inspected periods, not a new
pristine confirmatory backtest. It is designed as a falsification test for
incremental economic value beyond ordinary confidence.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LAMBDA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 777])
    parser.add_argument(
        "--development-dir",
        type=Path,
        default=Path("artifacts/earnings_bayesian_lora_uncertainty_control"),
    )
    parser.add_argument(
        "--holdout-dir",
        type=Path,
        default=Path("artifacts/earnings_bayesian_lora_final_holdout"),
    )
    parser.add_argument("--kl-beta", type=float, default=0.003)
    parser.add_argument("--mc-samples", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/earnings_model_risk_penalty/summary.json"),
    )
    return parser.parse_args()


def mean(xs: list[float]) -> float:
    return float(statistics.fmean(xs))


def pstdev(xs: list[float]) -> float:
    return float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0


def empirical_cdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference, dtype=float))
    return np.searchsorted(ref, np.asarray(values, dtype=float), side="right") / len(ref)


def normalize_same_gross(raw: np.ndarray) -> np.ndarray:
    raw = np.maximum(np.asarray(raw, dtype=float), 0.0)
    total = float(raw.sum())
    if total <= 0:
        raise ValueError("allocation rule produced zero gross exposure")
    return raw * (len(raw) / total)


def effective_n(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    norm = w / float(w.sum())
    return float(1.0 / np.sum(norm**2))


def powered_weights(raw: np.ndarray, gamma: float) -> np.ndarray:
    x = np.maximum(np.asarray(raw, dtype=float), 1e-15)
    return normalize_same_gross(np.power(x, gamma))


def match_effective_n(raw: np.ndarray, target_effective_n: float) -> tuple[np.ndarray, float]:
    """Power-transform raw scores until their effective N matches target."""
    n = len(raw)
    target = float(np.clip(target_effective_n, 1.0, float(n)))
    lo = 1e-4
    hi = 1.0
    neff_lo = effective_n(powered_weights(raw, lo))
    neff_hi = effective_n(powered_weights(raw, hi))

    while neff_hi > target and hi < 1024.0:
        hi *= 2.0
        neff_hi = effective_n(powered_weights(raw, hi))

    if target >= neff_lo:
        return powered_weights(raw, lo), lo

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        neff_mid = effective_n(powered_weights(raw, mid))
        if neff_mid > target:
            lo = mid
        else:
            hi = mid

    gamma = 0.5 * (lo + hi)
    return powered_weights(raw, gamma), gamma


def allocation_metrics(frame: pd.DataFrame, weights: np.ndarray) -> dict[str, float]:
    y = frame["label"].to_numpy(dtype=float)
    move = frame["move_1d"].to_numpy(dtype=float)
    w = np.asarray(weights, dtype=float)
    norm = w / float(w.sum())
    downside_loss = np.maximum(-move, 0.0)
    return {
        "weighted_downside_event_rate": float(np.dot(norm, y)),
        "weighted_mean_move_1d": float(np.dot(norm, move)),
        "weighted_mean_downside_loss": float(np.dot(norm, downside_loss)),
        "effective_n": effective_n(w),
        "max_normalized_weight": float(np.max(norm)),
    }


def prediction_path(directory: Path, beta: float, seed: int, mc_samples: int) -> Path:
    beta_tag = str(beta).replace(".", "p")
    return directory / f"beta_{beta_tag}_seed_{seed}_mc{mc_samples}_predictions.csv"


def confidence_raw(frame: pd.DataFrame) -> np.ndarray:
    p = frame["posterior_mean"].to_numpy(dtype=float)
    return (1.0 - p) * (2.0 * np.abs(p - 0.5))


def augmented_raw(
    frame: pd.DataFrame,
    reference_epistemic: np.ndarray,
    lam: float,
) -> np.ndarray:
    epi = frame["epistemic_variance"].to_numpy(dtype=float)
    epi_percentile = empirical_cdf(reference_epistemic, epi)
    return confidence_raw(frame) * np.exp(-lam * epi_percentile)


def select_lambda(calibration: pd.DataFrame) -> tuple[float, list[dict[str, float]]]:
    """Select lambda on calibration outcomes at confidence-matched concentration."""
    ref_epi = calibration["epistemic_variance"].to_numpy(dtype=float)
    conf_raw = confidence_raw(calibration)
    conf_weights = normalize_same_gross(conf_raw)
    target_neff = effective_n(conf_weights)

    rows: list[dict[str, float]] = []
    best_lambda = LAMBDA_GRID[0]
    best_risk = float("inf")

    for lam in LAMBDA_GRID:
        raw = augmented_raw(calibration, ref_epi, lam)
        matched, gamma = match_effective_n(raw, target_neff)
        metrics = allocation_metrics(calibration, matched)
        row = {
            "lambda": float(lam),
            "matching_gamma": float(gamma),
            **metrics,
        }
        rows.append(row)

        risk = metrics["weighted_downside_event_rate"]
        if risk < best_risk - 1e-15:
            best_risk = risk
            best_lambda = lam
        elif abs(risk - best_risk) <= 1e-15 and lam < best_lambda:
            best_lambda = lam

    return float(best_lambda), rows


def evaluate_period(calibration: pd.DataFrame, evaluation: pd.DataFrame) -> dict[str, Any]:
    selected_lambda, calibration_grid = select_lambda(calibration)

    ref_epi = calibration["epistemic_variance"].to_numpy(dtype=float)
    conf_raw_eval = confidence_raw(evaluation)
    conf_weights = normalize_same_gross(conf_raw_eval)
    conf_neff = effective_n(conf_weights)

    aug_raw_eval = augmented_raw(evaluation, ref_epi, selected_lambda)
    aug_matched, gamma = match_effective_n(aug_raw_eval, conf_neff)

    conf_metrics = allocation_metrics(evaluation, conf_weights)
    aug_metrics = allocation_metrics(evaluation, aug_matched)

    comparison = {
        "delta_downside_event_rate_augmented_minus_confidence": (
            aug_metrics["weighted_downside_event_rate"] - conf_metrics["weighted_downside_event_rate"]
        ),
        "delta_mean_move_1d_augmented_minus_confidence": (
            aug_metrics["weighted_mean_move_1d"] - conf_metrics["weighted_mean_move_1d"]
        ),
        "delta_downside_loss_augmented_minus_confidence": (
            aug_metrics["weighted_mean_downside_loss"] - conf_metrics["weighted_mean_downside_loss"]
        ),
    }

    return {
        "n_calibration": int(len(calibration)),
        "n_evaluation": int(len(evaluation)),
        "evaluation_start": str(pd.to_datetime(evaluation["date"]).min().date()),
        "evaluation_end": str(pd.to_datetime(evaluation["date"]).max().date()),
        "lambda_grid": list(LAMBDA_GRID),
        "calibration_selection": calibration_grid,
        "selected_lambda": selected_lambda,
        "evaluation_matching_gamma": float(gamma),
        "metrics": {
            "confidence_adjusted": conf_metrics,
            "confidence_plus_epistemic_model_risk_penalty": aug_metrics,
        },
        "comparison": comparison,
    }


def main() -> None:
    args = parse_args()
    runs: list[dict[str, Any]] = []

    for seed in args.seeds:
        dev_path = prediction_path(args.development_dir, args.kl_beta, seed, args.mc_samples)
        holdout_path = prediction_path(args.holdout_dir, args.kl_beta, seed, args.mc_samples)
        if not dev_path.exists():
            raise SystemExit(f"Missing development predictions: {dev_path}")
        if not holdout_path.exists():
            raise SystemExit(f"Missing holdout predictions: {holdout_path}")

        dev = pd.read_csv(dev_path)
        holdout = pd.read_csv(holdout_path)
        dev_dates = pd.to_datetime(dev["date"])

        calibration_2022 = dev.loc[dev_dates.dt.year == 2022].copy()
        evaluation_2023 = dev.loc[dev_dates.dt.year == 2023].copy()
        if calibration_2022.empty or evaluation_2023.empty:
            raise SystemExit("Development predictions must contain both 2022 and 2023 rows")

        runs.append(
            {
                "seed": seed,
                "periods": {
                    "2023_from_2022_calibration": evaluate_period(calibration_2022, evaluation_2023),
                    "2024_2025_from_pre2024_calibration": evaluate_period(dev.copy(), holdout),
                },
            }
        )

    aggregate: dict[str, Any] = {}
    for period_name in ("2023_from_2022_calibration", "2024_2025_from_pre2024_calibration"):
        risk = [
            float(r["periods"][period_name]["comparison"]
                  ["delta_downside_event_rate_augmented_minus_confidence"])
            for r in runs
        ]
        move = [
            float(r["periods"][period_name]["comparison"]
                  ["delta_mean_move_1d_augmented_minus_confidence"])
            for r in runs
        ]
        loss = [
            float(r["periods"][period_name]["comparison"]
                  ["delta_downside_loss_augmented_minus_confidence"])
            for r in runs
        ]
        selected = [float(r["periods"][period_name]["selected_lambda"]) for r in runs]

        aggregate[period_name] = {
            "selected_lambdas": selected,
            "delta_downside_event_rate_mean": mean(risk),
            "delta_downside_event_rate_std": pstdev(risk),
            "augmented_beats_confidence_downside_event_rate_seed_count": sum(x < 0 for x in risk),
            "delta_mean_move_1d_mean": mean(move),
            "augmented_beats_confidence_mean_move_seed_count": sum(x > 0 for x in move),
            "delta_downside_loss_mean": mean(loss),
            "augmented_beats_confidence_downside_loss_seed_count": sum(x < 0 for x in loss),
        }

    summary = {
        "purpose": "test epistemic uncertainty as an incremental model-risk penalty beyond confidence",
        "methodological_status": (
            "post-hoc robustness analysis on already-inspected periods; not a pristine confirmatory backtest"
        ),
        "design": {
            "baseline": "(1-p) * 2*abs(p-0.5)",
            "augmented": "baseline * exp(-lambda * prior-period epistemic percentile)",
            "lambda_selection": (
                "choose lambda from fixed grid on calibration outcomes only after matching each candidate "
                "to confidence effective N"
            ),
            "evaluation_control": (
                "match augmented evaluation effective N to confidence using weights only, then compare outcomes"
            ),
        },
        "runs": runs,
        "aggregate": aggregate,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Summary written to {args.output}")


if __name__ == "__main__":
    main()
