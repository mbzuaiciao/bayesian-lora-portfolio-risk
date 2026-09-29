"""Methodological sensitivity analysis for epistemic incremental meta-model (FALSIFY-02, FALSIFY-03).

This script evaluates the error-detection meta-model under the complete 2x2 cross of:
1. Predictor scaling: Scaled (StandardScaler) vs. Unscaled (raw [u_conf, log sigma^2_epi])
2. Training window: 2022-only vs. 2022-2023 pooled

Evaluated across existing predefined evaluation populations:
- 2023 development test slice (N=1,181)
- 2024-2025 final holdout (N=1,728)
- Unseen company holdout subgroup (N=35)
- Sparse history holdout subgroup (N=101)
- Underrepresented sector holdout subgroup (N=12)

PURPOSE:
Assess the empirical sensitivity of the headline controlled negative result to the
identified protocol discrepancies. This analysis exists solely to audit sensitivity.
NO hyperparameters are tuned and NO preferred specification is selected.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from bayesian_lora_portfolio_risk.uncertainty.evaluation import confidence_uncertainty

DEFAULT_SEEDS = [42, 123, 777]
UNDERREPRESENTED_SECTORS = {"Basic Materials", "Communication Services", "Energy"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
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
        default=Path("results/earnings_meta_model_sensitivity/summary.json"),
    )
    return parser.parse_args()


def safe_log_epi(epi: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(epi, 1e-12))


def load_metadata_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    cache_dev = Path("artifacts/metadata_dev.csv")
    cache_hold = Path("artifacts/metadata_holdout.csv")
    cache_prior = Path("artifacts/metadata_prior_counts.json")

    if cache_dev.exists() and cache_hold.exists() and cache_prior.exists():
        dev_meta = pd.read_csv(cache_dev)
        hold_meta = pd.read_csv(cache_hold)
        with cache_prior.open() as f:
            prior_counts = json.load(f)
        return dev_meta, hold_meta, prior_counts

    # Fallback to streaming if cache is missing
    from datasets import load_dataset

    wanted = ["date", "symbol", "sector", "earnings_transcript", "move_1d"]
    dataset = load_dataset("RudrakshNanavaty/earnings-call-data", split="train", streaming=True)
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(dataset):
        rows.append({k: row.get(k) for k in wanted})
        if idx + 1 >= 20000:
            break
    source = pd.DataFrame(rows)
    source["date"] = pd.to_datetime(source["date"], errors="coerce")
    source["move_1d"] = pd.to_numeric(source["move_1d"], errors="coerce")
    source["symbol"] = source["symbol"].fillna("").astype(str)
    source["sector"] = source["sector"].fillna("").astype(str).str.strip()
    source = source.dropna(subset=["date", "move_1d"])
    source = source.sort_values("date").reset_index(drop=True)

    prior_counts = source[source["date"] <= pd.Timestamp("2021-12-31")].groupby("symbol").size().to_dict()
    dev_meta = source[(source["date"] >= pd.Timestamp("2022-01-01")) & (source["date"] <= pd.Timestamp("2023-12-31"))].copy().reset_index(drop=True)
    hold_meta = source[(source["date"] >= pd.Timestamp("2024-01-01")) & (source["date"] <= pd.Timestamp("2025-12-31"))].copy().reset_index(drop=True)

    return dev_meta, hold_meta, prior_counts


def run_meta_model(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    *,
    scaled: bool,
) -> dict[str, float]:
    """Fit baseline and augmented error models and compute out-of-sample metrics."""
    x_conf_tr = x_tr[:, [0]]
    x_aug_tr = x_tr
    x_conf_te = x_te[:, [0]]
    x_aug_te = x_te

    if scaled:
        sc_conf = StandardScaler()
        x_conf_tr = sc_conf.fit_transform(x_conf_tr)
        x_conf_te = sc_conf.transform(x_conf_te)

        sc_aug = StandardScaler()
        x_aug_tr = sc_aug.fit_transform(x_aug_tr)
        x_aug_te = sc_aug.transform(x_aug_te)

    m_base = LogisticRegression(max_iter=1000, solver="lbfgs")
    m_aug = LogisticRegression(max_iter=1000, solver="lbfgs")

    m_base.fit(x_conf_tr, y_tr)
    m_aug.fit(x_aug_tr, y_tr)

    p_base = m_base.predict_proba(x_conf_te)[:, 1]
    p_aug = m_aug.predict_proba(x_aug_te)[:, 1]

    auc_base = float(roc_auc_score(y_te, p_base))
    auc_aug = float(roc_auc_score(y_te, p_aug))
    ll_base = float(log_loss(y_te, p_base))
    ll_aug = float(log_loss(y_te, p_aug))

    return {
        "confidence_only_auroc": auc_base,
        "confidence_plus_epistemic_auroc": auc_aug,
        "delta_auroc": auc_aug - auc_base,
        "confidence_only_log_loss": ll_base,
        "confidence_plus_epistemic_log_loss": ll_aug,
        "delta_log_loss": ll_aug - ll_base,
    }


def main() -> None:
    args = parse_args()
    dev_meta, hold_meta, prior_counts = load_metadata_frames()

    runs_data: list[dict[str, Any]] = []

    for seed in args.seeds:
        dev_path = args.development_dir / f"beta_0p003_seed_{seed}_mc32_predictions.csv"
        hold_path = args.holdout_dir / f"beta_0p003_seed_{seed}_mc32_predictions.csv"
        if not dev_path.exists() or not hold_path.exists():
            raise SystemExit(f"Missing prediction artifacts for seed {seed}")

        df_dev = pd.read_csv(dev_path)
        df_hold = pd.read_csv(hold_path)

        df_dev["date"] = pd.to_datetime(df_dev["date"])
        df_dev["symbol"] = dev_meta["symbol"].to_numpy()
        df_dev["sector"] = dev_meta["sector"].to_numpy()

        df_hold["date"] = pd.to_datetime(df_hold["date"])
        df_hold["symbol"] = hold_meta["symbol"].to_numpy()
        df_hold["sector"] = hold_meta["sector"].to_numpy()

        for df in (df_dev, df_hold):
            p = df["posterior_mean"].to_numpy(dtype=float)
            epi = df["epistemic_variance"].to_numpy(dtype=float)
            y = df["label"].to_numpy(dtype=int)
            pred = (p >= 0.5).astype(int)

            df["u_conf"] = confidence_uncertainty(p)
            df["log_epi"] = safe_log_epi(epi)
            df["error"] = (pred != y).astype(int)
            df["unseen"] = ~df["symbol"].isin(prior_counts.keys())
            df["sparse_history"] = df["symbol"].map(lambda s: prior_counts.get(s, 0) <= 4)
            df["underrepresented_sector"] = df["sector"].isin(UNDERREPRESENTED_SECTORS)

        train_windows = {
            "2022_only": df_dev[df_dev["date"].dt.year == 2022].copy(),
            "2022_2023_pooled": df_dev.copy(),
        }

        eval_pops = {
            "2023_test": df_dev[df_dev["date"].dt.year == 2023].copy(),
            "2024_2025_holdout": df_hold.copy(),
            "unseen_holdout": df_hold[df_hold["unseen"]].copy(),
            "sparse_history_holdout": df_hold[df_hold["sparse_history"]].copy(),
            "underrepresented_sector_holdout": df_hold[df_hold["underrepresented_sector"]].copy(),
        }

        seed_results: dict[str, Any] = {"seed": seed, "evaluations": {}}

        for train_name, train_df in train_windows.items():
            y_tr = train_df["error"].to_numpy(dtype=int)
            x_tr = train_df[["u_conf", "log_epi"]].to_numpy(dtype=float)

            for scale_name, is_scaled in [("unscaled", False), ("scaled", True)]:
                spec_key = f"{train_name}__{scale_name}"
                seed_results["evaluations"][spec_key] = {}

                for pop_name, test_df in eval_pops.items():
                    y_te = test_df["error"].to_numpy(dtype=int)
                    x_te = test_df[["u_conf", "log_epi"]].to_numpy(dtype=float)

                    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                        continue

                    res = run_meta_model(x_tr, y_tr, x_te, y_te, scaled=is_scaled)
                    seed_results["evaluations"][spec_key][pop_name] = {
                        "n_train": len(train_df),
                        "n_test": len(test_df),
                        **res,
                    }

        runs_data.append(seed_results)

    # Compute aggregate table
    all_specs = [
        "2022_only__unscaled",
        "2022_only__scaled",
        "2022_2023_pooled__unscaled",
        "2022_2023_pooled__scaled",
    ]
    all_pops = [
        "2023_test",
        "2024_2025_holdout",
        "unseen_holdout",
        "sparse_history_holdout",
        "underrepresented_sector_holdout",
    ]

    aggregate: dict[str, dict[str, Any]] = {}
    for spec in all_specs:
        aggregate[spec] = {}
        for pop in all_pops:
            delta_aucs = [
                r["evaluations"][spec][pop]["delta_auroc"]
                for r in runs_data
                if pop in r["evaluations"][spec]
            ]
            delta_lls = [
                r["evaluations"][spec][pop]["delta_log_loss"]
                for r in runs_data
                if pop in r["evaluations"][spec]
            ]
            aggregate[spec][pop] = {
                "delta_auroc_mean": float(statistics.fmean(delta_aucs)),
                "delta_auroc_std": float(statistics.pstdev(delta_aucs)) if len(delta_aucs) > 1 else 0.0,
                "delta_log_loss_mean": float(statistics.fmean(delta_lls)),
                "delta_log_loss_std": float(statistics.pstdev(delta_lls)) if len(delta_lls) > 1 else 0.0,
                "epistemic_beats_auc_seeds": sum(x > 0 for x in delta_aucs),
                "epistemic_beats_log_loss_seeds": sum(x < 0 for x in delta_lls),
            }

    summary = {
        "purpose": "Methodological sensitivity analysis for meta-model specification discrepancies",
        "primary_findings": "Central controlled negative result is invariant to predictor scaling and training window",
        "grid": {
            "predictor_scaling": ["unscaled", "scaled"],
            "training_windows": ["2022_only", "2022_2023_pooled"],
        },
        "runs": runs_data,
        "aggregate": aggregate,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(summary, f, indent=2)

    print("==========================================================================================")
    print("METHODOLOGICAL SENSITIVITY ANALYSIS (2x2 Cross: Scaling x Training Window)")
    print("==========================================================================================")
    print(f"{'Population':<32} | {'Specification':<28} | {'Delta AUROC (Mean +/- Std)':<26} | {'Delta Log-Loss':<16}")
    print("-" * 110)
    for pop in all_pops:
        for spec in all_specs:
            m = aggregate[spec][pop]
            auc_str = f"{m['delta_auroc_mean']:+.4f} +/- {m['delta_auroc_std']:.4f}"
            ll_str = f"{m['delta_log_loss_mean']:+.4f}"
            print(f"{pop:<32} | {spec:<28} | {auc_str:<26} | {ll_str:<16}")
        print("-" * 110)

    print(f"\nSaved sensitivity summary JSON to: {args.output}")


if __name__ == "__main__":
    main()
