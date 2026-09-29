"""Compare Bayesian epistemic uncertainty against a simple confidence baseline.

This is the next controlled experiment after multi-seed replication selected
KL beta=0.003. The question is whether Bayesian epistemic variance identifies
errors better than the much simpler signal "the classifier probability is near
0.5".

Protocol
--------
- Freeze beta=0.003; do not tune it further here.
- Run seeds 42, 123, and 777 by default.
- Use 32 posterior samples for a more stable epistemic-variance estimate.
- Keep the same pre-2022 train/validation and 2022-2023 development diagnostic
  period used in earlier experiments.
- Treat 2024-2025 as untouched future holdout; this script does not inspect it.
- Compare error-detection AUROC and selective risk for:
    1) Bayesian epistemic variance;
    2) deterministic confidence uncertainty = 1 - 2*abs(p - 0.5).

The confidence score is monotonic with binary predictive entropy, so it is a
simple deterministic control for whether "uncertainty" merely means p is near
0.5.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from bayesian_lora_portfolio_risk.uncertainty.metrics import selective_risk_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kl-beta", type=float, default=0.003)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 777])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--mc-samples", type=int, default=32)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/earnings_bayesian_lora_uncertainty_control"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/earnings_bayesian_lora_uncertainty_control/summary.json"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun a seed even when its JSON artifact already exists.",
    )
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def pstdev(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def risk_at_coverage(correct: np.ndarray, uncertainty: np.ndarray, target: float) -> float:
    coverage, risk = selective_risk_curve(correct, uncertainty)
    index = int(np.argmin(np.abs(coverage - target)))
    return float(risk[index])


def confidence_uncertainty(probability: np.ndarray) -> np.ndarray:
    """Return 0 for p in {0,1} and 1 for p=0.5; monotonic with binary entropy."""
    return 1.0 - 2.0 * np.abs(probability - 0.5)


def write_summary(args: argparse.Namespace, runs: list[dict[str, Any]]) -> None:
    complete = len(runs) == len(args.seeds)
    epi_auc = [float(row["epistemic_error_detection_auroc"]) for row in runs]
    conf_auc = [float(row["confidence_error_detection_auroc"]) for row in runs]
    auc_delta = [a - b for a, b in zip(epi_auc, conf_auc, strict=True)]
    epi_risk = [float(row["epistemic_selective_risk_50"]) for row in runs]
    conf_risk = [float(row["confidence_selective_risk_50"]) for row in runs]
    risk_delta = [b - a for a, b in zip(epi_risk, conf_risk, strict=True)]

    summary: dict[str, Any] = {
        "purpose": "Bayesian epistemic uncertainty versus deterministic confidence control",
        "important_note": (
            "2022-2023 remains development/diagnostic data. Do not use these results "
            "to tune additional hyperparameters; reserve 2024-2025 for final holdout confirmation."
        ),
        "kl_beta_frozen": args.kl_beta,
        "mc_samples": args.mc_samples,
        "configs_completed": len(runs),
        "configs_total": len(args.seeds),
        "complete": complete,
        "runs": runs,
    }
    if runs:
        summary["aggregate"] = {
            "epistemic_error_detection_auroc_mean": mean(epi_auc),
            "epistemic_error_detection_auroc_std": pstdev(epi_auc),
            "confidence_error_detection_auroc_mean": mean(conf_auc),
            "confidence_error_detection_auroc_std": pstdev(conf_auc),
            "epistemic_minus_confidence_error_detection_auroc_mean": mean(auc_delta),
            "epistemic_beats_confidence_auc_seed_count": sum(x > 0 for x in auc_delta),
            "epistemic_selective_risk_50_mean": mean(epi_risk),
            "confidence_selective_risk_50_mean": mean(conf_risk),
            "confidence_minus_epistemic_selective_risk_50_mean": mean(risk_delta),
            "epistemic_beats_confidence_selective_risk_seed_count": sum(x > 0 for x in risk_delta),
        }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []

    for seed in args.seeds:
        tag = f"beta_{str(args.kl_beta).replace('.', 'p')}_seed_{seed}_mc{args.mc_samples}"
        output = args.output_dir / f"{tag}.json"
        predictions = args.output_dir / f"{tag}_predictions.csv"
        checkpoint = args.output_dir / f"{tag}_state.pt"

        if output.exists() and predictions.exists() and not args.force:
            print(f"[resume] seed {seed}: using existing artifacts", flush=True)
        else:
            command = [
                sys.executable,
                "scripts/run_earnings_bayesian_lora.py",
                "--batch-size",
                str(args.batch_size),
                "--epochs",
                str(args.epochs),
                "--lora-r",
                str(args.lora_r),
                "--learning-rate",
                str(args.learning_rate),
                "--kl-beta",
                str(args.kl_beta),
                "--mc-samples",
                str(args.mc_samples),
                "--seed",
                str(seed),
                "--output",
                str(output),
                "--predictions-output",
                str(predictions),
                "--checkpoint",
                str(checkpoint),
            ]
            if args.require_cuda:
                command.append("--require-cuda")
            print(f"[run] seed {seed}: {' '.join(command)}", flush=True)
            subprocess.run(command, check=True)

        result = json.loads(output.read_text())
        frame = pd.read_csv(predictions)
        y = frame["label"].to_numpy(dtype=int)
        p = frame["posterior_mean"].to_numpy(dtype=float)
        epi = frame["epistemic_variance"].to_numpy(dtype=float)
        predicted = p >= 0.5
        correct = predicted == y
        incorrect = (~correct).astype(int)
        conf = confidence_uncertainty(p)

        epi_auc = float(roc_auc_score(incorrect, epi))
        conf_auc = float(roc_auc_score(incorrect, conf))
        epi_risk_50 = risk_at_coverage(correct, epi, 0.5)
        conf_risk_50 = risk_at_coverage(correct, conf, 0.5)
        full_risk = risk_at_coverage(correct, epi, 1.0)

        runs.append(
            {
                "seed": seed,
                "validation_auroc": float(result["best_validation_auroc_posterior_mean"]),
                "diagnostic_test_posterior_predictive_auroc": float(
                    result["test_posterior_predictive"]["auroc"]
                ),
                "epistemic_error_detection_auroc": epi_auc,
                "confidence_error_detection_auroc": conf_auc,
                "epistemic_minus_confidence_error_detection_auroc": epi_auc - conf_auc,
                "epistemic_selective_risk_50": epi_risk_50,
                "confidence_selective_risk_50": conf_risk_50,
                "full_risk": full_risk,
                "full_minus_epistemic_selective_risk_50": full_risk - epi_risk_50,
                "full_minus_confidence_selective_risk_50": full_risk - conf_risk_50,
                "mean_epistemic_variance": float(np.mean(epi)),
                "artifact": str(output),
                "predictions": str(predictions),
            }
        )
        write_summary(args, runs)

    summary = json.loads(args.summary.read_text())
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Summary written to {args.summary}", flush=True)


if __name__ == "__main__":
    main()
