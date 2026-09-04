"""Run the frozen Bayesian-LoRA specification on the untouched 2024-2025 holdout.

This is a confirmation experiment, not another tuning stage.

Frozen before inspecting 2024-2025
----------------------------------
- KL beta: 0.003
- LoRA rank: 16
- learning rate: 2e-4
- epochs: 4
- seeds: 42, 123, 777
- posterior samples: 32
- training/validation data: pre-2022 only
- holdout: 2024-01-01 through 2025-12-31 (available data ends earlier if the source does)

Pre-specified questions
-----------------------
1. Does epistemic variance identify errors on the untouched future period?
2. Does it beat ordinary confidence uncertainty = 1 - 2*abs(p - 0.5)?
3. Does selecting the least-uncertain 50% reduce error more than confidence selection?
4. Using the already-fixed meta-model specification from development, does adding
   log epistemic variance to confidence improve error prediction out of sample?

For question 4, the logistic meta-model is fit on 2022 development predictions
from the earlier uncertainty-control experiment and evaluated only on 2024-2025.
No 2024-2025 labels are used for fitting or hyperparameter selection.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

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
    parser.add_argument("--holdout-start", default="2024-01-01")
    parser.add_argument("--holdout-end", default="2025-12-31")
    parser.add_argument(
        "--development-dir",
        type=Path,
        default=Path("artifacts/earnings_bayesian_lora_uncertainty_control"),
        help="Directory containing the frozen beta=0.003, MC32 development predictions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/earnings_bayesian_lora_final_holdout"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/earnings_bayesian_lora_final_holdout/summary.json"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun a seed even when its holdout artifacts already exist.",
    )
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def pstdev(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def confidence_uncertainty(probability: np.ndarray) -> np.ndarray:
    return 1.0 - 2.0 * np.abs(probability - 0.5)


def risk_at_coverage(correct: np.ndarray, uncertainty: np.ndarray, target: float) -> float:
    coverage, risk = selective_risk_curve(correct, uncertainty)
    index = int(np.argmin(np.abs(coverage - target)))
    return float(risk[index])


def error_features(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = frame["label"].to_numpy(dtype=int)
    p = frame["posterior_mean"].to_numpy(dtype=float)
    epi = frame["epistemic_variance"].to_numpy(dtype=float)
    incorrect = ((p >= 0.5) != y).astype(int)
    conf = confidence_uncertainty(p)
    log_epi = np.log(np.maximum(epi, np.finfo(float).tiny))
    return incorrect, conf, log_epi


def fit_meta_models(dev: pd.DataFrame) -> tuple[LogisticRegression, LogisticRegression]:
    incorrect, conf, log_epi = error_features(dev)
    baseline = LogisticRegression(max_iter=1000, solver="lbfgs")
    augmented = LogisticRegression(max_iter=1000, solver="lbfgs")
    baseline.fit(conf.reshape(-1, 1), incorrect)
    augmented.fit(np.column_stack([conf, log_epi]), incorrect)
    return baseline, augmented


def evaluate_meta_models(
    frame: pd.DataFrame,
    baseline: LogisticRegression,
    augmented: LogisticRegression,
) -> dict[str, Any]:
    incorrect, conf, log_epi = error_features(frame)
    p_base = baseline.predict_proba(conf.reshape(-1, 1))[:, 1]
    p_aug = augmented.predict_proba(np.column_stack([conf, log_epi]))[:, 1]
    base_auc = float(roc_auc_score(incorrect, p_base))
    aug_auc = float(roc_auc_score(incorrect, p_aug))
    base_ll = float(log_loss(incorrect, p_base))
    aug_ll = float(log_loss(incorrect, p_aug))
    return {
        "confidence_only": {"auroc": base_auc, "log_loss": base_ll},
        "confidence_plus_epistemic": {"auroc": aug_auc, "log_loss": aug_ll},
        "delta_auroc": aug_auc - base_auc,
        "delta_log_loss": aug_ll - base_ll,
    }


def write_summary(args: argparse.Namespace, runs: list[dict[str, Any]]) -> None:
    summary: dict[str, Any] = {
        "purpose": "frozen 2024-2025 Bayesian-LoRA holdout confirmation",
        "frozen_before_holdout": {
            "kl_beta": args.kl_beta,
            "lora_r": args.lora_r,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "mc_samples": args.mc_samples,
            "seeds": args.seeds,
            "holdout_start": args.holdout_start,
            "holdout_end": args.holdout_end,
        },
        "important_note": (
            "This file contains the first inspection of the reserved 2024-2025 holdout. "
            "Do not tune the method on these results."
        ),
        "configs_completed": len(runs),
        "configs_total": len(args.seeds),
        "complete": len(runs) == len(args.seeds),
        "runs": runs,
    }
    if runs:
        epi_auc = [float(r["epistemic_error_detection_auroc"]) for r in runs]
        conf_auc = [float(r["confidence_error_detection_auroc"]) for r in runs]
        epi_risk = [float(r["epistemic_selective_risk_50"]) for r in runs]
        conf_risk = [float(r["confidence_selective_risk_50"]) for r in runs]
        meta_delta_auc = [float(r["incremental_meta_model"]["delta_auroc"]) for r in runs]
        meta_delta_ll = [float(r["incremental_meta_model"]["delta_log_loss"]) for r in runs]
        summary["aggregate"] = {
            "epistemic_error_detection_auroc_mean": mean(epi_auc),
            "epistemic_error_detection_auroc_std": pstdev(epi_auc),
            "confidence_error_detection_auroc_mean": mean(conf_auc),
            "confidence_error_detection_auroc_std": pstdev(conf_auc),
            "epistemic_minus_confidence_error_detection_auroc_mean": mean(
                [a - b for a, b in zip(epi_auc, conf_auc)]
            ),
            "epistemic_beats_confidence_auc_seed_count": sum(a > b for a, b in zip(epi_auc, conf_auc)),
            "epistemic_selective_risk_50_mean": mean(epi_risk),
            "confidence_selective_risk_50_mean": mean(conf_risk),
            "epistemic_beats_confidence_selective_risk_seed_count": sum(
                a < b for a, b in zip(epi_risk, conf_risk)
            ),
            "incremental_meta_model_delta_auroc_mean": mean(meta_delta_auc),
            "incremental_meta_model_delta_auroc_std": pstdev(meta_delta_auc),
            "incremental_meta_model_augmented_beats_auc_seed_count": sum(x > 0 for x in meta_delta_auc),
            "incremental_meta_model_delta_log_loss_mean": mean(meta_delta_ll),
            "incremental_meta_model_augmented_beats_log_loss_seed_count": sum(x < 0 for x in meta_delta_ll),
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
            print(f"[resume] holdout seed {seed}: using existing artifacts", flush=True)
        else:
            command = [
                sys.executable,
                "scripts/run_earnings_bayesian_lora.py",
                "--batch-size", str(args.batch_size),
                "--epochs", str(args.epochs),
                "--lora-r", str(args.lora_r),
                "--learning-rate", str(args.learning_rate),
                "--kl-beta", str(args.kl_beta),
                "--mc-samples", str(args.mc_samples),
                "--seed", str(seed),
                "--test-start", args.holdout_start,
                "--test-end", args.holdout_end,
                "--output", str(output),
                "--predictions-output", str(predictions),
                "--checkpoint", str(checkpoint),
            ]
            if args.require_cuda:
                command.append("--require-cuda")
            print(f"[run] holdout seed {seed}: {' '.join(command)}", flush=True)
            subprocess.run(command, check=True)

        result = json.loads(output.read_text())
        holdout = pd.read_csv(predictions)
        y = holdout["label"].to_numpy(dtype=int)
        p = holdout["posterior_mean"].to_numpy(dtype=float)
        epi = holdout["epistemic_variance"].to_numpy(dtype=float)
        predicted = p >= 0.5
        correct = predicted == y
        incorrect = (~correct).astype(int)
        conf = confidence_uncertainty(p)

        epi_auc = float(roc_auc_score(incorrect, epi))
        conf_auc = float(roc_auc_score(incorrect, conf))
        epi_risk_50 = risk_at_coverage(correct, epi, 0.5)
        conf_risk_50 = risk_at_coverage(correct, conf, 0.5)
        full_risk = float(np.mean(incorrect))

        dev_predictions = args.development_dir / f"{tag}_predictions.csv"
        if not dev_predictions.exists():
            raise SystemExit(
                f"Missing development predictions for seed {seed}: {dev_predictions}. "
                "The frozen meta-model must be fit from the prior 2022 development experiment."
            )
        dev = pd.read_csv(dev_predictions)
        dev["date"] = pd.to_datetime(dev["date"])
        dev_2022 = dev[dev["date"].dt.year == 2022].copy()
        if dev_2022.empty:
            raise SystemExit(f"No 2022 development rows found in {dev_predictions}")
        baseline_meta, augmented_meta = fit_meta_models(dev_2022)
        meta = evaluate_meta_models(holdout, baseline_meta, augmented_meta)

        runs.append(
            {
                "seed": seed,
                "holdout_n": int(len(holdout)),
                "holdout_start_observed": str(pd.to_datetime(holdout["date"]).min().date()),
                "holdout_end_observed": str(pd.to_datetime(holdout["date"]).max().date()),
                "holdout_error_rate": full_risk,
                "holdout_posterior_predictive_auroc": float(result["test_posterior_predictive"]["auroc"]),
                "epistemic_error_detection_auroc": epi_auc,
                "confidence_error_detection_auroc": conf_auc,
                "epistemic_minus_confidence_error_detection_auroc": epi_auc - conf_auc,
                "epistemic_selective_risk_50": epi_risk_50,
                "confidence_selective_risk_50": conf_risk_50,
                "full_minus_epistemic_selective_risk_50": full_risk - epi_risk_50,
                "full_minus_confidence_selective_risk_50": full_risk - conf_risk_50,
                "mean_epistemic_variance": float(np.mean(epi)),
                "incremental_meta_model": meta,
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
