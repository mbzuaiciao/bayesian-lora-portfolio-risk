"""Replicate Bayesian-LoRA uncertainty across beta values and random seeds.

This stage follows the KL-beta sweep. The goal is not to keep tuning the model;
it is to test whether the useful uncertainty ordering survives random-seed changes.

Protocol
--------
- Compare only the two plausible beta values from the beta sweep: 0.003 and 0.01.
- Run each beta with seeds 42, 123, and 777 by default.
- Preserve the underlying pre-2022 validation AUROC as the model-selection metric.
- Treat 2022-2023 uncertainty statistics as a development diagnostic, not a final
  untouched test. A later 2024-2025 holdout should be reserved for final claims.
- Persist progress after every run so interrupted cloud sessions can resume.

The summary emphasizes reproducibility of:
1. validation posterior-mean AUROC;
2. error-detection AUROC from epistemic variance;
3. low-to-high uncertainty-quintile error ordering;
4. selective-risk reduction at 50% coverage.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--betas", nargs="+", type=float, default=[0.003, 0.01])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 777])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--mc-samples", type=int, default=8)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/earnings_bayesian_lora_repro"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/earnings_bayesian_lora_repro/summary.json"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun configurations even when their JSON artifacts already exist.",
    )
    return parser.parse_args()


def beta_tag(beta: float) -> str:
    return f"beta_{beta:g}".replace(".", "p").replace("-", "m")


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def pstdev(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def summarize_beta(beta: float, runs: list[dict[str, Any]]) -> dict[str, Any]:
    validation_auc = [float(row["validation_auroc"]) for row in runs]
    error_auc = [float(row["error_detection_auroc"]) for row in runs]
    low_error = [float(row["uncertainty_quintile_errors"][0]) for row in runs]
    high_error = [float(row["uncertainty_quintile_errors"][-1]) for row in runs]
    quintile_gap = [high - low for low, high in zip(low_error, high_error)]
    selective_50 = [float(row["selective_risk_50"]) for row in runs]
    full_risk = [float(row["selective_risk_100"]) for row in runs]
    selective_improvement = [full - half for full, half in zip(full_risk, selective_50)]
    monotone = [bool(row["strictly_monotone_quintiles"]) for row in runs]

    return {
        "kl_beta": beta,
        "n_seeds": len(runs),
        "validation_auroc_mean": mean(validation_auc),
        "validation_auroc_std": pstdev(validation_auc),
        "error_detection_auroc_mean": mean(error_auc),
        "error_detection_auroc_std": pstdev(error_auc),
        "lowest_uncertainty_quintile_error_mean": mean(low_error),
        "highest_uncertainty_quintile_error_mean": mean(high_error),
        "high_minus_low_quintile_error_gap_mean": mean(quintile_gap),
        "high_minus_low_quintile_error_gap_std": pstdev(quintile_gap),
        "quintile_gap_positive_all_seeds": all(gap > 0 for gap in quintile_gap),
        "strictly_monotone_quintiles_seed_count": sum(monotone),
        "selective_risk_50_mean": mean(selective_50),
        "full_risk_mean": mean(full_risk),
        "full_minus_50pct_selective_risk_mean": mean(selective_improvement),
        "selective_risk_improves_all_seeds": all(x > 0 for x in selective_improvement),
        "mean_epistemic_variance": mean(
            [float(row["mean_epistemic_variance"]) for row in runs]
        ),
        "runs": runs,
    }


def write_summary(args: argparse.Namespace, records: list[dict[str, Any]]) -> None:
    grouped: list[dict[str, Any]] = []
    for beta in args.betas:
        beta_runs = [row for row in records if float(row["kl_beta"]) == float(beta)]
        if beta_runs:
            grouped.append(summarize_beta(beta, beta_runs))

    complete = len(records) == len(args.betas) * len(args.seeds)
    summary = {
        "purpose": "multi-seed replication of Bayesian-LoRA uncertainty",
        "important_note": (
            "2022-2023 uncertainty metrics are development diagnostics because that period "
            "has already been inspected repeatedly. Reserve 2024-2025 for a later untouched holdout."
        ),
        "selection_metric": "pre-2022 validation posterior-mean AUROC",
        "configs_completed": len(records),
        "configs_total": len(args.betas) * len(args.seeds),
        "complete": complete,
        "beta_summaries": grouped,
    }
    if complete and grouped:
        summary["selected_by_mean_validation_auroc"] = max(
            grouped, key=lambda row: row["validation_auroc_mean"]
        )["kl_beta"]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for beta in args.betas:
        for seed in args.seeds:
            tag = f"{beta_tag(beta)}_seed_{seed}"
            output = args.output_dir / f"{tag}.json"
            predictions = args.output_dir / f"{tag}_predictions.csv"
            checkpoint = args.output_dir / f"{tag}_state.pt"

            if output.exists() and not args.force:
                print(f"[resume] {tag}: using {output}", flush=True)
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
                    str(beta),
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
                print(f"[run] {tag}: {' '.join(command)}", flush=True)
                subprocess.run(command, check=True)

            result = load_json(output)
            uncertainty = result["uncertainty"]
            quintiles = [
                float(x)
                for x in uncertainty["error_rate_by_uncertainty_quintile_low_to_high"]
            ]
            selective = uncertainty["selective_risk"]
            record = {
                "kl_beta": beta,
                "seed": seed,
                "validation_auroc": float(
                    result["best_validation_auroc_posterior_mean"]
                ),
                "diagnostic_test_posterior_predictive_auroc": float(
                    result["test_posterior_predictive"]["auroc"]
                ),
                "error_detection_auroc": float(uncertainty["error_detection_auroc"]),
                "uncertainty_quintile_errors": quintiles,
                "strictly_monotone_quintiles": all(
                    right > left for left, right in zip(quintiles, quintiles[1:])
                ),
                "selective_risk_50": float(selective["coverage_0.5"]),
                "selective_risk_100": float(selective["coverage_1.0"]),
                "mean_epistemic_variance": float(
                    uncertainty["mean_epistemic_variance"]
                ),
                "artifact": str(output),
            }
            records.append(record)
            write_summary(args, records)

    summary = load_json(args.summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Summary written to {args.summary}", flush=True)


if __name__ == "__main__":
    main()
