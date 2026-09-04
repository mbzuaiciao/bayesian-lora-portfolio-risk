"""Rolling temporal robustness check for uncertainty-aware earnings allocation.

This is a CPU-only follow-up to the exploratory event-risk allocation analysis.
It does not create a new pristine holdout. Instead it asks whether the same frozen
allocation logic behaves similarly when evaluated sequentially through time.

Protocol
--------
For each Bayesian seed, combine the existing 2022-2023 development predictions
with the existing 2024-2025 holdout predictions. Evaluate two forward slices:

- 2023, with epistemic CDF calibration using 2022 only;
- 2024-2025, with epistemic CDF calibration using all observations before 2024.

The three allocation rules remain frozen:
1. posterior_mean_only: raw weight = 1 - p;
2. confidence_adjusted: raw weight = (1 - p) * 2*abs(p - 0.5);
3. epistemic_adjusted: raw weight = (1 - p) * (1 - prior-period epistemic CDF percentile).

All rules are normalized to identical gross exposure within each evaluation slice.
Because all periods have already been inspected elsewhere in the project, this is
a temporal robustness analysis, not a confirmatory backtest.
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
    parser.add_argument("--kl-beta", type=float, default=0.003)
    parser.add_argument("--mc-samples", type=int, default=32)
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/earnings_event_risk_allocation_rolling/summary.json"),
    )
    return parser.parse_args()


def mean(xs: list[float]) -> float:
    return float(statistics.fmean(xs))


def pstdev(xs: list[float]) -> float:
    return float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0


def prediction_path(directory: Path, beta: float, seed: int, mc_samples: int) -> Path:
    beta_tag = str(beta).replace(".", "p")
    return directory / f"beta_{beta_tag}_seed_{seed}_mc{mc_samples}_predictions.csv"


def empirical_cdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference, dtype=float))
    if len(ref) == 0:
        raise ValueError("empty calibration sample")
    return np.searchsorted(ref, np.asarray(values, dtype=float), side="right") / len(ref)


def normalize_same_gross(raw: np.ndarray) -> np.ndarray:
    raw = np.maximum(np.asarray(raw, dtype=float), 0.0)
    total = float(raw.sum())
    if total <= 0:
        raise ValueError("allocation rule produced zero gross exposure")
    return raw * (len(raw) / total)


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
        "effective_n": float(1.0 / np.sum(norm**2)),
    }


def evaluate_slice(calibration: pd.DataFrame, evaluation: pd.DataFrame) -> dict[str, Any]:
    p = evaluation["posterior_mean"].to_numpy(dtype=float)
    epi = evaluation["epistemic_variance"].to_numpy(dtype=float)
    calibration_epi = calibration["epistemic_variance"].to_numpy(dtype=float)

    base_raw = 1.0 - p
    confidence_reliability = 2.0 * np.abs(p - 0.5)
    epistemic_reliability = 1.0 - empirical_cdf(calibration_epi, epi)

    weights = {
        "posterior_mean_only": normalize_same_gross(base_raw),
        "confidence_adjusted": normalize_same_gross(base_raw * confidence_reliability),
        "epistemic_adjusted": normalize_same_gross(base_raw * epistemic_reliability),
    }
    metrics = {name: allocation_metrics(evaluation, w) for name, w in weights.items()}
    conf = metrics["confidence_adjusted"]
    epi_m = metrics["epistemic_adjusted"]
    return {
        "n_calibration": int(len(calibration)),
        "n_evaluation": int(len(evaluation)),
        "evaluation_start": str(evaluation["date"].min().date()),
        "evaluation_end": str(evaluation["date"].max().date()),
        "metrics": metrics,
        "epistemic_minus_confidence": {
            "downside_event_rate": epi_m["weighted_downside_event_rate"] - conf["weighted_downside_event_rate"],
            "mean_move_1d": epi_m["weighted_mean_move_1d"] - conf["weighted_mean_move_1d"],
            "mean_downside_loss": epi_m["weighted_mean_downside_loss"] - conf["weighted_mean_downside_loss"],
        },
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
        dev["date"] = pd.to_datetime(dev["date"])
        holdout["date"] = pd.to_datetime(holdout["date"])
        full = pd.concat([dev, holdout], ignore_index=True).sort_values("date").reset_index(drop=True)

        calibration_2022 = full[full["date"].dt.year == 2022].copy()
        eval_2023 = full[full["date"].dt.year == 2023].copy()
        calibration_pre_2024 = full[full["date"] < pd.Timestamp("2024-01-01")].copy()
        eval_2024_2025 = full[full["date"] >= pd.Timestamp("2024-01-01")].copy()

        if calibration_2022.empty or eval_2023.empty or calibration_pre_2024.empty or eval_2024_2025.empty:
            raise SystemExit(f"Insufficient rolling data for seed {seed}")

        runs.append(
            {
                "seed": seed,
                "periods": {
                    "2023_from_2022_calibration": evaluate_slice(calibration_2022, eval_2023),
                    "2024_2025_from_pre2024_calibration": evaluate_slice(
                        calibration_pre_2024, eval_2024_2025
                    ),
                },
            }
        )

    period_names = ["2023_from_2022_calibration", "2024_2025_from_pre2024_calibration"]
    aggregate: dict[str, Any] = {}
    for period in period_names:
        risk_delta = [
            float(r["periods"][period]["epistemic_minus_confidence"]["downside_event_rate"])
            for r in runs
        ]
        move_delta = [
            float(r["periods"][period]["epistemic_minus_confidence"]["mean_move_1d"])
            for r in runs
        ]
        loss_delta = [
            float(r["periods"][period]["epistemic_minus_confidence"]["mean_downside_loss"])
            for r in runs
        ]
        aggregate[period] = {
            "epistemic_minus_confidence_downside_event_rate_mean": mean(risk_delta),
            "epistemic_minus_confidence_downside_event_rate_std": pstdev(risk_delta),
            "epistemic_beats_confidence_downside_event_rate_seed_count": sum(x < 0 for x in risk_delta),
            "epistemic_minus_confidence_mean_move_1d_mean": mean(move_delta),
            "epistemic_beats_confidence_mean_move_seed_count": sum(x > 0 for x in move_delta),
            "epistemic_minus_confidence_downside_loss_mean": mean(loss_delta),
            "epistemic_beats_confidence_downside_loss_seed_count": sum(x < 0 for x in loss_delta),
        }

    summary = {
        "purpose": "rolling temporal robustness of uncertainty-aware earnings allocation",
        "methodological_status": (
            "robustness analysis on already-inspected periods; not a pristine confirmatory backtest"
        ),
        "frozen_rules": {
            "posterior_mean_only": "1 - p",
            "confidence_adjusted": "(1 - p) * 2*abs(p - 0.5)",
            "epistemic_adjusted": "(1 - p) * (1 - prior-period epistemic CDF percentile)",
            "gross_normalization": "identical gross exposure within each evaluation slice",
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
