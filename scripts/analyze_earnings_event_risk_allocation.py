"""Test whether Bayesian uncertainty changes downstream event-risk allocation.

This is an economic bridge between the earnings classifier and a later full
covariance-aware portfolio optimizer. It does not claim to be a complete
portfolio backtest: each earnings episode is treated as one independent capital
allocation opportunity.

The analysis compares three frozen allocation rules:

1. posterior_mean_only
   Allocate more capital when the Bayesian posterior mean assigns lower downside
   probability: raw weight = 1 - p.

2. confidence_adjusted
   Start from the same posterior-mean weight and shrink it when the classifier is
   close to 0.5: reliability = 2 * abs(p - 0.5).

3. epistemic_adjusted
   Start from the same posterior-mean weight and shrink it when epistemic variance
   is high. Epistemic variance is converted to a reliability score using the
   empirical CDF from the 2022-2023 development period only. No holdout outcome
   is used to fit that mapping.

All strategies are normalized to the same gross exposure on the evaluation
period so lower risk cannot be achieved merely by investing less capital.

Important methodological note
-----------------------------
The 2024-2025 holdout has already been inspected for predictive uncertainty.
Therefore this downstream analysis is exploratory/post-hoc, not a new pristine
confirmatory test. The allocation rules themselves are fixed and use development
information only where scaling is required.
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
        default=Path("artifacts/earnings_event_risk_allocation/summary.json"),
    )
    return parser.parse_args()


def mean(xs: list[float]) -> float:
    return float(statistics.fmean(xs))


def pstdev(xs: list[float]) -> float:
    return float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0


def empirical_cdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Return P_reference(X <= value) using only the supplied reference sample."""
    ref = np.sort(np.asarray(reference, dtype=float))
    return np.searchsorted(ref, np.asarray(values, dtype=float), side="right") / len(ref)


def normalize_same_gross(raw: np.ndarray) -> np.ndarray:
    raw = np.maximum(np.asarray(raw, dtype=float), 0.0)
    total = float(raw.sum())
    if total <= 0:
        raise ValueError("allocation rule produced zero gross exposure")
    # Mean weight one => every strategy has identical total gross exposure N.
    return raw * (len(raw) / total)


def allocation_metrics(frame: pd.DataFrame, weights: np.ndarray) -> dict[str, float]:
    y = frame["label"].to_numpy(dtype=float)
    move = frame["move_1d"].to_numpy(dtype=float)
    w = np.asarray(weights, dtype=float)
    gross = float(w.sum())
    norm = w / gross

    bad = y == 1
    good = ~bad
    downside_loss = np.maximum(-move, 0.0)

    return {
        "gross_exposure": gross,
        "weighted_downside_event_rate": float(np.dot(norm, y)),
        "weighted_mean_move_1d": float(np.dot(norm, move)),
        "weighted_mean_downside_loss": float(np.dot(norm, downside_loss)),
        "mean_weight_bad_events": float(np.mean(w[bad])) if np.any(bad) else float("nan"),
        "mean_weight_good_events": float(np.mean(w[good])) if np.any(good) else float("nan"),
        "bad_to_good_mean_weight_ratio": (
            float(np.mean(w[bad]) / np.mean(w[good]))
            if np.any(bad) and np.any(good) and np.mean(w[good]) != 0
            else float("nan")
        ),
        "effective_n": float(1.0 / np.sum(norm**2)),
        "max_normalized_weight": float(np.max(norm)),
    }


def prediction_path(directory: Path, beta: float, seed: int, mc_samples: int) -> Path:
    beta_tag = str(beta).replace(".", "p")
    return directory / f"beta_{beta_tag}_seed_{seed}_mc{mc_samples}_predictions.csv"


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

        p = holdout["posterior_mean"].to_numpy(dtype=float)
        epi = holdout["epistemic_variance"].to_numpy(dtype=float)
        dev_epi = dev["epistemic_variance"].to_numpy(dtype=float)

        base_raw = 1.0 - p
        confidence_reliability = 2.0 * np.abs(p - 0.5)
        # High epistemic percentile means low reliability. The CDF mapping is fit
        # entirely from development epistemic values and then frozen on holdout.
        epi_percentile = empirical_cdf(dev_epi, epi)
        epistemic_reliability = 1.0 - epi_percentile

        weights = {
            "posterior_mean_only": normalize_same_gross(base_raw),
            "confidence_adjusted": normalize_same_gross(base_raw * confidence_reliability),
            "epistemic_adjusted": normalize_same_gross(base_raw * epistemic_reliability),
        }
        metrics = {name: allocation_metrics(holdout, w) for name, w in weights.items()}
        base = metrics["posterior_mean_only"]

        comparisons: dict[str, dict[str, float]] = {}
        for name in ("confidence_adjusted", "epistemic_adjusted"):
            row = metrics[name]
            comparisons[name] = {
                "delta_downside_event_rate_vs_mean_only": (
                    row["weighted_downside_event_rate"] - base["weighted_downside_event_rate"]
                ),
                "delta_mean_move_1d_vs_mean_only": (
                    row["weighted_mean_move_1d"] - base["weighted_mean_move_1d"]
                ),
                "delta_mean_downside_loss_vs_mean_only": (
                    row["weighted_mean_downside_loss"] - base["weighted_mean_downside_loss"]
                ),
            }

        epi_vs_conf = {
            "delta_downside_event_rate_epistemic_minus_confidence": (
                metrics["epistemic_adjusted"]["weighted_downside_event_rate"]
                - metrics["confidence_adjusted"]["weighted_downside_event_rate"]
            ),
            "delta_mean_move_1d_epistemic_minus_confidence": (
                metrics["epistemic_adjusted"]["weighted_mean_move_1d"]
                - metrics["confidence_adjusted"]["weighted_mean_move_1d"]
            ),
            "delta_mean_downside_loss_epistemic_minus_confidence": (
                metrics["epistemic_adjusted"]["weighted_mean_downside_loss"]
                - metrics["confidence_adjusted"]["weighted_mean_downside_loss"]
            ),
        }

        runs.append(
            {
                "seed": seed,
                "n_holdout": int(len(holdout)),
                "holdout_start": str(pd.to_datetime(holdout["date"]).min().date()),
                "holdout_end": str(pd.to_datetime(holdout["date"]).max().date()),
                "allocation_metrics": metrics,
                "comparisons_vs_posterior_mean_only": comparisons,
                "epistemic_vs_confidence": epi_vs_conf,
                "development_predictions": str(dev_path),
                "holdout_predictions": str(holdout_path),
            }
        )

    epi_delta_risk = [
        float(r["comparisons_vs_posterior_mean_only"]["epistemic_adjusted"]["delta_downside_event_rate_vs_mean_only"])
        for r in runs
    ]
    conf_delta_risk = [
        float(r["comparisons_vs_posterior_mean_only"]["confidence_adjusted"]["delta_downside_event_rate_vs_mean_only"])
        for r in runs
    ]
    epi_minus_conf_risk = [
        float(r["epistemic_vs_confidence"]["delta_downside_event_rate_epistemic_minus_confidence"])
        for r in runs
    ]
    epi_minus_conf_loss = [
        float(r["epistemic_vs_confidence"]["delta_mean_downside_loss_epistemic_minus_confidence"])
        for r in runs
    ]
    epi_minus_conf_move = [
        float(r["epistemic_vs_confidence"]["delta_mean_move_1d_epistemic_minus_confidence"])
        for r in runs
    ]

    summary = {
        "purpose": "exploratory economic bridge from Bayesian earnings risk to capital allocation",
        "methodological_status": (
            "post-hoc downstream analysis: 2024-2025 was already inspected for predictive uncertainty; "
            "do not describe this as a pristine confirmatory portfolio test"
        ),
        "allocation_rules": {
            "posterior_mean_only": "gross-normalized weight proportional to 1 - posterior downside probability",
            "confidence_adjusted": "posterior-mean weight multiplied by 2*abs(p-0.5)",
            "epistemic_adjusted": (
                "posterior-mean weight multiplied by 1 - development-CDF percentile of epistemic variance"
            ),
        },
        "fairness_constraint": "all strategies normalized to identical gross exposure",
        "primary_risk_metric": "weighted downside-event rate using the already-fixed downside label",
        "runs": runs,
        "aggregate": {
            "epistemic_delta_downside_event_rate_vs_mean_only_mean": mean(epi_delta_risk),
            "epistemic_delta_downside_event_rate_vs_mean_only_std": pstdev(epi_delta_risk),
            "confidence_delta_downside_event_rate_vs_mean_only_mean": mean(conf_delta_risk),
            "confidence_delta_downside_event_rate_vs_mean_only_std": pstdev(conf_delta_risk),
            "epistemic_minus_confidence_downside_event_rate_mean": mean(epi_minus_conf_risk),
            "epistemic_beats_confidence_downside_event_rate_seed_count": sum(x < 0 for x in epi_minus_conf_risk),
            "epistemic_minus_confidence_downside_loss_mean": mean(epi_minus_conf_loss),
            "epistemic_beats_confidence_downside_loss_seed_count": sum(x < 0 for x in epi_minus_conf_loss),
            "epistemic_minus_confidence_mean_move_1d_mean": mean(epi_minus_conf_move),
            "epistemic_beats_confidence_mean_move_seed_count": sum(x > 0 for x in epi_minus_conf_move),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Summary written to {args.output}")


if __name__ == "__main__":
    main()
