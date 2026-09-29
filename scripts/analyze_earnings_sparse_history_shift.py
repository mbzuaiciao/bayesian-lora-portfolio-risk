"""Pre-specified sparse-history company shift test for Bayesian-LoRA uncertainty.

Motivation
----------
The strict unseen-company subgroup was scientifically interesting but very small
(only a handful of companies). This analysis broadens the same hypothesis without
using outcomes to define the subgroup.

Primary shift definition
------------------------
A company is "sparse-history" if it has at most 4 usable earnings episodes in the
pre-2022 model-development data. This includes truly unseen companies (0 episodes)
and few-shot companies (1-4 episodes). A company is "well-covered" if it has more
than 4 pre-2022 episodes.

The threshold of 4 is fixed here before inspecting subgroup outcomes. It is not
selected from predictive performance.

Evaluation plan
---------------
1. Evaluate sparse-history companies in 2023. Fit confidence-only and
   confidence+epistemic error meta-models on well-covered 2022 examples only.
2. Evaluate sparse-history companies in 2024-2025. Fit the same meta-models on
   well-covered 2022-2023 examples only.

Primary criterion
-----------------
Epistemic uncertainty is useful under sparse-history shift only if adding it to
ordinary confidence improves out-of-sample error prediction on the shifted
subgroup. Higher raw uncertainty alone is insufficient.

Methodological note
-------------------
This subgroup definition is pre-specified before its outcomes are inspected, but
the underlying dates and the stricter unseen-company subgroup have already been
analyzed. Treat this as a targeted robustness/shift analysis, not a pristine new
holdout.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import pandas as pd
from analyze_earnings_unseen_company_shift import (
    attach_metadata,
    basic_metrics,
    fit_and_evaluate_meta,
    load_metadata,
    prediction_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="RudrakshNanavaty/earnings-call-data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-rows", type=int, default=20_000)
    parser.add_argument("--train-end", default="2021-12-31")
    parser.add_argument("--max-prior-episodes", type=int, default=4)
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
        default=Path("artifacts/earnings_sparse_history_shift/summary.json"),
    )
    return parser.parse_args()


def mean(xs: list[float]) -> float:
    return float(statistics.fmean(xs))


def pstdev(xs: list[float]) -> float:
    return float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0


def main() -> None:
    args = parse_args()
    source = load_metadata(args)
    train_end = pd.Timestamp(args.train_end)

    prior = source[source["date"] <= train_end].copy()
    prior_counts = prior.groupby("symbol", dropna=False).size().to_dict()

    def prior_count(symbol: str) -> int:
        return int(prior_counts.get(symbol, 0))

    runs: list[dict[str, Any]] = []
    for seed in args.seeds:
        dev_path = prediction_path(args.development_dir, args.kl_beta, seed, args.mc_samples)
        holdout_path = prediction_path(args.holdout_dir, args.kl_beta, seed, args.mc_samples)
        if not dev_path.exists() or not holdout_path.exists():
            raise SystemExit(f"Missing prediction artifacts for seed {seed}")

        dev = attach_metadata(pd.read_csv(dev_path), source, start="2022-01-01", end="2023-12-31")
        holdout = attach_metadata(pd.read_csv(holdout_path), source, start="2024-01-01", end="2025-12-31")

        for frame in (dev, holdout):
            frame["pre2022_episode_count"] = frame["symbol"].map(prior_count).astype(int)
            frame["sparse_history_company"] = (
                frame["pre2022_episode_count"] <= args.max_prior_episodes
            )

        dev_dates = pd.to_datetime(dev["date"])
        sparse_2023 = dev[(dev_dates.dt.year == 2023) & dev["sparse_history_company"]].copy()
        covered_2022 = dev[
            (dev_dates.dt.year == 2022) & (~dev["sparse_history_company"])
        ].copy()
        covered_pre2024 = dev[~dev["sparse_history_company"]].copy()
        sparse_holdout = holdout[holdout["sparse_history_company"]].copy()

        runs.append(
            {
                "seed": seed,
                "max_prior_episodes": args.max_prior_episodes,
                "2023": {
                    "well_covered": basic_metrics(
                        dev[(dev_dates.dt.year == 2023) & (~dev["sparse_history_company"])]
                    ),
                    "sparse_history": basic_metrics(sparse_2023),
                    "incremental_meta_model_well_covered_2022_to_sparse_2023": (
                        fit_and_evaluate_meta(covered_2022, sparse_2023)
                    ),
                },
                "2024_2025": {
                    "well_covered": basic_metrics(holdout[~holdout["sparse_history_company"]]),
                    "sparse_history": basic_metrics(sparse_holdout),
                    "incremental_meta_model_well_covered_pre2024_to_sparse_2024_2025": (
                        fit_and_evaluate_meta(covered_pre2024, sparse_holdout)
                    ),
                },
            }
        )

    aggregate: dict[str, Any] = {}
    for period, meta_key in (
        ("2023", "incremental_meta_model_well_covered_2022_to_sparse_2023"),
        ("2024_2025", "incremental_meta_model_well_covered_pre2024_to_sparse_2024_2025"),
    ):
        raw_delta = [
            r[period]["sparse_history"].get(
                "epistemic_minus_confidence_error_detection_auroc"
            )
            for r in runs
        ]
        raw_delta = [float(x) for x in raw_delta if x is not None]
        delta_auc = [
            float(r[period][meta_key]["delta_auroc"])
            for r in runs
            if r[period][meta_key].get("status") == "ok"
        ]
        delta_ll = [
            float(r[period][meta_key]["delta_log_loss"])
            for r in runs
            if r[period][meta_key].get("status") == "ok"
        ]
        aggregate[period] = {
            "sparse_history_n_per_seed": [
                int(r[period]["sparse_history"].get("n", 0)) for r in runs
            ],
            "sparse_history_epistemic_minus_confidence_error_detection_auroc_mean": (
                mean(raw_delta) if raw_delta else None
            ),
            "sparse_history_epistemic_minus_confidence_error_detection_auroc_std": (
                pstdev(raw_delta) if raw_delta else None
            ),
            "epistemic_beats_confidence_error_detection_seed_count": sum(
                x > 0 for x in raw_delta
            ),
            "incremental_meta_model_delta_auroc_mean": (
                mean(delta_auc) if delta_auc else None
            ),
            "incremental_meta_model_delta_auroc_std": (
                pstdev(delta_auc) if delta_auc else None
            ),
            "incremental_meta_model_delta_log_loss_mean": (
                mean(delta_ll) if delta_ll else None
            ),
            "incremental_epistemic_beats_auc_seed_count": sum(x > 0 for x in delta_auc),
            "incremental_epistemic_beats_log_loss_seed_count": sum(x < 0 for x in delta_ll),
        }

    pre_counts = source[source["date"] <= train_end].groupby("symbol").size()
    summary = {
        "purpose": "pre-specified sparse-history company distribution-shift test",
        "shift_definition": (
            f"company has <= {args.max_prior_episodes} usable pre-2022 earnings episodes"
        ),
        "primary_criterion": (
            "epistemic uncertainty must add out-of-sample error-prediction information "
            "beyond confidence on sparse-history companies"
        ),
        "methodological_status": (
            "targeted robustness/shift analysis; subgroup threshold fixed before its outcomes, "
            "but underlying dates and strict unseen-company subgroup were previously inspected"
        ),
        "pre2022_company_count": int(len(pre_counts)),
        "pre2022_sparse_company_count": int((pre_counts <= args.max_prior_episodes).sum()),
        "pre2022_well_covered_company_count": int((pre_counts > args.max_prior_episodes).sum()),
        "runs": runs,
        "aggregate": aggregate,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Summary written to {args.output}")


if __name__ == "__main__":
    main()
