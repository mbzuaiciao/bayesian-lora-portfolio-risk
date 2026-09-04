"""Control for portfolio concentration in uncertainty-aware earnings allocation.

The earlier exploratory allocation analyses found that epistemic-adjusted sizing
outperformed confidence-adjusted sizing, but the epistemic rule also produced a
lower effective number of positions. This script tests whether that advantage is
merely a consequence of concentration.

For each seed and evaluation period we construct the usual frozen raw scores:

    mean-only:   (1 - p)
    confidence:  (1 - p) * 2*abs(p - 0.5)
    epistemic:   (1 - p) * (1 - prior-period epistemic CDF percentile)

We then preserve each rule's ranking while changing only its concentration by
raising its non-negative raw scores to a positive power gamma. Gamma is chosen
by bisection so that one strategy matches the other strategy's effective number
of positions. This produces two symmetric controls:

1. confidence_matched_to_epistemic_concentration
   Concentrate the confidence rule until its effective N matches epistemic.

2. epistemic_matched_to_confidence_concentration
   Flatten the epistemic rule until its effective N matches confidence.

All final weights are gross-normalized to mean weight one. No evaluation-period
outcomes are used to select gamma; gamma depends only on the candidate weights'
concentration.

Methodological status: robustness analysis on already-inspected periods, not a
pristine confirmatory backtest.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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
        default=Path("artifacts/earnings_event_risk_allocation_concentration_control/summary.json"),
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
    """Return power-transformed weights whose effective N matches target.

    Effective N decreases monotonically as gamma grows for non-uniform positive
    weights. We search over a broad positive interval without using outcomes.
    """
    n = len(raw)
    target = float(np.clip(target_effective_n, 1.0, float(n)))

    lo = 1e-4
    hi = 1.0
    neff_lo = effective_n(powered_weights(raw, lo))
    neff_hi = effective_n(powered_weights(raw, hi))

    # Increase gamma until weights are concentrated enough for the target.
    while neff_hi > target and hi < 1024.0:
        hi *= 2.0
        neff_hi = effective_n(powered_weights(raw, hi))

    # If target is flatter than our near-zero power limit, return the flattest
    # attainable positive-power version. This should be rare in this use case.
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


def evaluate_period(
    calibration: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> dict[str, Any]:
    p = evaluation["posterior_mean"].to_numpy(dtype=float)
    epi = evaluation["epistemic_variance"].to_numpy(dtype=float)
    calibration_epi = calibration["epistemic_variance"].to_numpy(dtype=float)

    base_raw = 1.0 - p
    confidence_raw = base_raw * (2.0 * np.abs(p - 0.5))
    epi_percentile = empirical_cdf(calibration_epi, epi)
    epistemic_raw = base_raw * (1.0 - epi_percentile)

    confidence = normalize_same_gross(confidence_raw)
    epistemic = normalize_same_gross(epistemic_raw)
    conf_neff = effective_n(confidence)
    epi_neff = effective_n(epistemic)

    confidence_matched, gamma_conf_to_epi = match_effective_n(confidence_raw, epi_neff)
    epistemic_matched, gamma_epi_to_conf = match_effective_n(epistemic_raw, conf_neff)

    weights = {
        "confidence_adjusted": confidence,
        "epistemic_adjusted": epistemic,
        "confidence_matched_to_epistemic_concentration": confidence_matched,
        "epistemic_matched_to_confidence_concentration": epistemic_matched,
    }
    metrics = {name: allocation_metrics(evaluation, w) for name, w in weights.items()}

    # Two complementary apples-to-apples comparisons.
    # A: confidence is given the same concentration as the original epistemic rule.
    cmp_at_epi_concentration = {
        "delta_downside_event_rate_epistemic_minus_matched_confidence": (
            metrics["epistemic_adjusted"]["weighted_downside_event_rate"]
            - metrics["confidence_matched_to_epistemic_concentration"]["weighted_downside_event_rate"]
        ),
        "delta_mean_move_1d_epistemic_minus_matched_confidence": (
            metrics["epistemic_adjusted"]["weighted_mean_move_1d"]
            - metrics["confidence_matched_to_epistemic_concentration"]["weighted_mean_move_1d"]
        ),
        "delta_downside_loss_epistemic_minus_matched_confidence": (
            metrics["epistemic_adjusted"]["weighted_mean_downside_loss"]
            - metrics["confidence_matched_to_epistemic_concentration"]["weighted_mean_downside_loss"]
        ),
    }

    # B: epistemic is flattened to the same concentration as original confidence.
    cmp_at_conf_concentration = {
        "delta_downside_event_rate_matched_epistemic_minus_confidence": (
            metrics["epistemic_matched_to_confidence_concentration"]["weighted_downside_event_rate"]
            - metrics["confidence_adjusted"]["weighted_downside_event_rate"]
        ),
        "delta_mean_move_1d_matched_epistemic_minus_confidence": (
            metrics["epistemic_matched_to_confidence_concentration"]["weighted_mean_move_1d"]
            - metrics["confidence_adjusted"]["weighted_mean_move_1d"]
        ),
        "delta_downside_loss_matched_epistemic_minus_confidence": (
            metrics["epistemic_matched_to_confidence_concentration"]["weighted_mean_downside_loss"]
            - metrics["confidence_adjusted"]["weighted_mean_downside_loss"]
        ),
    }

    return {
        "n_calibration": int(len(calibration)),
        "n_evaluation": int(len(evaluation)),
        "evaluation_start": str(pd.to_datetime(evaluation["date"]).min().date()),
        "evaluation_end": str(pd.to_datetime(evaluation["date"]).max().date()),
        "matching": {
            "original_confidence_effective_n": conf_neff,
            "original_epistemic_effective_n": epi_neff,
            "gamma_confidence_to_epistemic_effective_n": gamma_conf_to_epi,
            "gamma_epistemic_to_confidence_effective_n": gamma_epi_to_conf,
        },
        "metrics": metrics,
        "comparison_at_epistemic_concentration": cmp_at_epi_concentration,
        "comparison_at_confidence_concentration": cmp_at_conf_concentration,
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

        # Pre-2024 calibration uses all available 2022-2023 development predictions.
        pre2024 = dev.copy()

        runs.append(
            {
                "seed": seed,
                "periods": {
                    "2023_from_2022_calibration": evaluate_period(calibration_2022, evaluation_2023),
                    "2024_2025_from_pre2024_calibration": evaluate_period(pre2024, holdout),
                },
            }
        )

    aggregate: dict[str, Any] = {}
    for period_name in ("2023_from_2022_calibration", "2024_2025_from_pre2024_calibration"):
        a_risk = [
            float(r["periods"][period_name]["comparison_at_epistemic_concentration"]
                  ["delta_downside_event_rate_epistemic_minus_matched_confidence"])
            for r in runs
        ]
        a_move = [
            float(r["periods"][period_name]["comparison_at_epistemic_concentration"]
                  ["delta_mean_move_1d_epistemic_minus_matched_confidence"])
            for r in runs
        ]
        a_loss = [
            float(r["periods"][period_name]["comparison_at_epistemic_concentration"]
                  ["delta_downside_loss_epistemic_minus_matched_confidence"])
            for r in runs
        ]
        b_risk = [
            float(r["periods"][period_name]["comparison_at_confidence_concentration"]
                  ["delta_downside_event_rate_matched_epistemic_minus_confidence"])
            for r in runs
        ]
        b_move = [
            float(r["periods"][period_name]["comparison_at_confidence_concentration"]
                  ["delta_mean_move_1d_matched_epistemic_minus_confidence"])
            for r in runs
        ]
        b_loss = [
            float(r["periods"][period_name]["comparison_at_confidence_concentration"]
                  ["delta_downside_loss_matched_epistemic_minus_confidence"])
            for r in runs
        ]

        aggregate[period_name] = {
            "at_epistemic_concentration": {
                "epistemic_minus_matched_confidence_downside_event_rate_mean": mean(a_risk),
                "downside_event_rate_std": pstdev(a_risk),
                "epistemic_beats_seed_count": sum(x < 0 for x in a_risk),
                "epistemic_minus_matched_confidence_mean_move_1d_mean": mean(a_move),
                "mean_move_epistemic_beats_seed_count": sum(x > 0 for x in a_move),
                "epistemic_minus_matched_confidence_downside_loss_mean": mean(a_loss),
                "downside_loss_epistemic_beats_seed_count": sum(x < 0 for x in a_loss),
            },
            "at_confidence_concentration": {
                "matched_epistemic_minus_confidence_downside_event_rate_mean": mean(b_risk),
                "downside_event_rate_std": pstdev(b_risk),
                "epistemic_beats_seed_count": sum(x < 0 for x in b_risk),
                "matched_epistemic_minus_confidence_mean_move_1d_mean": mean(b_move),
                "mean_move_epistemic_beats_seed_count": sum(x > 0 for x in b_move),
                "matched_epistemic_minus_confidence_downside_loss_mean": mean(b_loss),
                "downside_loss_epistemic_beats_seed_count": sum(x < 0 for x in b_loss),
            },
        }

    summary = {
        "purpose": "test whether epistemic allocation advantage survives concentration matching",
        "methodological_status": "robustness analysis on already-inspected periods; not a pristine confirmatory backtest",
        "control_design": (
            "power-transform each rule's raw weights to match the other rule's effective number of positions; "
            "ranking is preserved and no outcomes are used to choose the power"
        ),
        "runs": runs,
        "aggregate": aggregate,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Summary written to {args.output}")


if __name__ == "__main__":
    main()
