"""Test whether Bayesian epistemic uncertainty adds value beyond simple confidence.

This analysis reuses completed 2022-2023 development predictions from the frozen
Bayesian-LoRA configuration (beta=0.003, MC=32). It does NOT retrain the model.

Question
--------
Given two predictions with similar posterior confidence, does epistemic variance
still help identify which prediction is more likely to be wrong?

Protocol
--------
For each seed:
- use 2022 rows to fit two simple logistic error models;
- evaluate on 2023 rows only;
- baseline model: confidence uncertainty only;
- augmented model: confidence uncertainty + log epistemic variance;
- report the out-of-year AUROC and log-loss improvement of the augmented model.

This is still a development diagnostic. Do not tune further on 2022-2023 based on
these results. Reserve 2024-2025 for final untouched holdout confirmation.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 777])
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("artifacts/earnings_bayesian_lora_uncertainty_control"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/earnings_bayesian_lora_uncertainty_control/"
            "incremental_epistemic_value.json"
        ),
    )
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def pstdev(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def confidence_uncertainty(probability: np.ndarray) -> np.ndarray:
    # 1 at p=0.5 and 0 at p in {0,1}; larger means less confident.
    return 1.0 - np.abs(2.0 * probability - 1.0)


def safe_log_variance(variance: np.ndarray) -> np.ndarray:
    return np.log(np.clip(variance, 1e-12, None))


def fit_score(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> dict[str, float]:
    scaler = StandardScaler()
    train_xs = scaler.fit_transform(train_x)
    test_xs = scaler.transform(test_x)
    model = LogisticRegression(max_iter=2000, solver="lbfgs")
    model.fit(train_xs, train_y)
    probability = model.predict_proba(test_xs)[:, 1]
    return {
        "auroc": float(roc_auc_score(test_y, probability)),
        "log_loss": float(log_loss(test_y, probability, labels=[0, 1])),
    }


def analyze_seed(path: Path, seed: int) -> dict[str, Any]:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", "posterior_mean", "epistemic_variance", "label"])

    probability = frame["posterior_mean"].to_numpy(dtype=float)
    label = frame["label"].to_numpy(dtype=int)
    predicted = (probability >= 0.5).astype(int)
    error = (predicted != label).astype(int)

    frame = frame.assign(
        error=error,
        confidence_uncertainty=confidence_uncertainty(probability),
        log_epistemic_variance=safe_log_variance(
            frame["epistemic_variance"].to_numpy(dtype=float)
        ),
    )

    train = frame[frame["date"].dt.year == 2022].copy()
    test = frame[frame["date"].dt.year == 2023].copy()
    if train.empty or test.empty:
        raise ValueError(f"seed {seed}: expected both 2022 and 2023 rows")
    if train["error"].nunique() < 2 or test["error"].nunique() < 2:
        raise ValueError(f"seed {seed}: error target lacks both classes")

    y_train = train["error"].to_numpy(dtype=int)
    y_test = test["error"].to_numpy(dtype=int)

    confidence_only = fit_score(
        train[["confidence_uncertainty"]].to_numpy(dtype=float),
        y_train,
        test[["confidence_uncertainty"]].to_numpy(dtype=float),
        y_test,
    )
    confidence_plus_epistemic = fit_score(
        train[["confidence_uncertainty", "log_epistemic_variance"]].to_numpy(dtype=float),
        y_train,
        test[["confidence_uncertainty", "log_epistemic_variance"]].to_numpy(dtype=float),
        y_test,
    )

    # A direct conditional ranking diagnostic: fit a linear relation from confidence
    # uncertainty to log epistemic variance on 2022, then ask whether the residual
    # epistemic component ranks 2023 errors above correct predictions.
    x = train["confidence_uncertainty"].to_numpy(dtype=float)
    z = train["log_epistemic_variance"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, z, deg=1)
    test_residual = (
        test["log_epistemic_variance"].to_numpy(dtype=float)
        - (slope * test["confidence_uncertainty"].to_numpy(dtype=float) + intercept)
    )
    residual_epistemic_error_auroc = float(roc_auc_score(y_test, test_residual))

    return {
        "seed": seed,
        "train_year": 2022,
        "test_year": 2023,
        "train_n": int(len(train)),
        "test_n": int(len(test)),
        "train_error_rate": float(y_train.mean()),
        "test_error_rate": float(y_test.mean()),
        "confidence_only": confidence_only,
        "confidence_plus_epistemic": confidence_plus_epistemic,
        "delta_auroc": (
            confidence_plus_epistemic["auroc"] - confidence_only["auroc"]
        ),
        "delta_log_loss": (
            confidence_plus_epistemic["log_loss"] - confidence_only["log_loss"]
        ),
        "residual_epistemic_error_auroc": residual_epistemic_error_auroc,
        "input": str(path),
    }


def main() -> None:
    args = parse_args()
    runs: list[dict[str, Any]] = []
    for seed in args.seeds:
        path = args.input_dir / f"beta_0p003_seed_{seed}_mc32_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        run = analyze_seed(path, seed)
        runs.append(run)
        print(json.dumps(run, indent=2), flush=True)

    delta_auc = [float(run["delta_auroc"]) for run in runs]
    delta_log_loss = [float(run["delta_log_loss"]) for run in runs]
    residual_auc = [float(run["residual_epistemic_error_auroc"]) for run in runs]

    summary = {
        "purpose": "incremental epistemic value beyond ordinary confidence",
        "important_note": (
            "This uses 2022 to fit a simple error model and evaluates on 2023. "
            "Both years are development diagnostics; reserve 2024-2025 for final confirmation."
        ),
        "baseline": "logistic error model using confidence uncertainty only",
        "augmented": (
            "logistic error model using confidence uncertainty plus log epistemic variance"
        ),
        "runs": runs,
        "aggregate": {
            "delta_auroc_mean": mean(delta_auc),
            "delta_auroc_std": pstdev(delta_auc),
            "augmented_beats_baseline_auroc_seed_count": sum(x > 0 for x in delta_auc),
            "delta_log_loss_mean": mean(delta_log_loss),
            "delta_log_loss_std": pstdev(delta_log_loss),
            "augmented_beats_baseline_log_loss_seed_count": sum(
                x < 0 for x in delta_log_loss
            ),
            "residual_epistemic_error_auroc_mean": mean(residual_auc),
            "residual_epistemic_error_auroc_std": pstdev(residual_auc),
            "residual_epistemic_above_random_seed_count": sum(x > 0.5 for x in residual_auc),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Summary written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
