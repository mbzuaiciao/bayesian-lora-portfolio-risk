"""Pre-specified underrepresented-sector distribution-shift test.

Research question
-----------------
Does Bayesian epistemic uncertainty add information beyond ordinary confidence
when the earnings model is evaluated on sectors that were underrepresented in
the pre-2022 development period?

Shift definition
----------------
Using only usable pre-2022 rows, count the number of earnings episodes per
non-empty sector. A sector is "underrepresented" if its pre-2022 episode count
is at or below the 25th percentile across sectors. The threshold is therefore
determined from training-distribution frequency only and does not use outcomes.

Evaluation plan
---------------
1. 2023 evaluation on underrepresented sectors, with error meta-models fit on
   2022 well-represented-sector examples only.
2. 2024-2025 evaluation on underrepresented sectors, with error meta-models fit
   on 2022-2023 well-represented-sector examples only.

Primary criterion
-----------------
Epistemic uncertainty must add out-of-sample error-prediction information beyond
ordinary confidence on the shifted subset. Higher raw uncertainty alone is not
sufficient.

Methodological note
-------------------
This is a targeted robustness/shift analysis. The sector-frequency rule is fixed
before inspecting its subgroup outcomes, but the underlying dates have been used
in prior analyses, so this is not a pristine confirmatory holdout.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="RudrakshNanavaty/earnings-call-data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-rows", type=int, default=20_000)
    parser.add_argument("--train-end", default="2021-12-31")
    parser.add_argument("--sector-quantile", type=float, default=0.25)
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
        default=Path("artifacts/earnings_underrepresented_sector_shift/summary.json"),
    )
    return parser.parse_args()


def mean(xs: list[float]) -> float:
    return float(statistics.fmean(xs))


def pstdev(xs: list[float]) -> float:
    return float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0


def confidence_uncertainty(p: np.ndarray) -> np.ndarray:
    return 1.0 - 2.0 * np.abs(np.asarray(p, dtype=float) - 0.5)


def prediction_path(directory: Path, beta: float, seed: int, mc_samples: int) -> Path:
    beta_tag = str(beta).replace(".", "p")
    return directory / f"beta_{beta_tag}_seed_{seed}_mc{mc_samples}_predictions.csv"


def load_metadata(args: argparse.Namespace) -> pd.DataFrame:
    from datasets import load_dataset

    wanted = ["date", "symbol", "sector", "earnings_transcript", "move_1d"]
    dataset = load_dataset(args.repo_id, split=args.split, streaming=True)
    rows: list[dict[str, object]] = []
    for index, row in enumerate(dataset):
        rows.append({key: row.get(key) for key in wanted})
        if index + 1 >= args.max_rows:
            break

    frame = pd.DataFrame.from_records(rows)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["move_1d"] = pd.to_numeric(frame["move_1d"], errors="coerce")
    frame["earnings_transcript"] = frame["earnings_transcript"].fillna("").astype(str)
    frame["symbol"] = frame["symbol"].fillna("").astype(str)
    frame["sector"] = frame["sector"].fillna("").astype(str).str.strip()
    frame = frame.dropna(subset=["date", "move_1d"])
    frame = frame[frame["earnings_transcript"].str.len() > 0]
    return frame.sort_values("date").reset_index(drop=True)


def attach_metadata(pred: pd.DataFrame, source: pd.DataFrame, *, start: str, end: str) -> pd.DataFrame:
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end)
    metadata = source[(source["date"] >= lo) & (source["date"] <= hi)].copy().reset_index(drop=True)
    out = pred.copy().reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"])
    if len(out) != len(metadata):
        raise SystemExit(
            f"Prediction/metadata row-count mismatch for {start}..{end}: "
            f"predictions={len(out)}, metadata={len(metadata)}"
        )
    if not np.array_equal(out["date"].to_numpy(), metadata["date"].to_numpy()):
        raise SystemExit(f"Date alignment failed for {start}..{end}")
    if not np.allclose(
        out["move_1d"].to_numpy(dtype=float), metadata["move_1d"].to_numpy(dtype=float),
        rtol=1e-10, atol=1e-12, equal_nan=True,
    ):
        raise SystemExit(f"move_1d alignment failed for {start}..{end}")
    out["symbol"] = metadata["symbol"].to_numpy()
    out["sector"] = metadata["sector"].to_numpy()
    return out


def basic_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"n": 0}
    y = frame["label"].to_numpy(dtype=int)
    p = frame["posterior_mean"].to_numpy(dtype=float)
    epi = frame["epistemic_variance"].to_numpy(dtype=float)
    incorrect = ((p >= 0.5) != y).astype(int)
    conf = confidence_uncertainty(p)
    result: dict[str, Any] = {
        "n": int(len(frame)),
        "unique_symbols": int(frame["symbol"].nunique()),
        "unique_sectors": int(frame["sector"].nunique()),
        "error_rate": float(np.mean(incorrect)),
        "mean_epistemic_variance": float(np.mean(epi)),
        "mean_confidence_uncertainty": float(np.mean(conf)),
    }
    if np.unique(incorrect).size == 2:
        epi_auc = float(roc_auc_score(incorrect, epi))
        conf_auc = float(roc_auc_score(incorrect, conf))
        result.update(
            {
                "epistemic_error_detection_auroc": epi_auc,
                "confidence_error_detection_auroc": conf_auc,
                "epistemic_minus_confidence_error_detection_auroc": epi_auc - conf_auc,
            }
        )
    else:
        result.update(
            {
                "epistemic_error_detection_auroc": None,
                "confidence_error_detection_auroc": None,
                "epistemic_minus_confidence_error_detection_auroc": None,
            }
        )
    return result


def fit_and_evaluate_meta(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    if train.empty or test.empty:
        return {"status": "insufficient_rows"}

    def features(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = frame["label"].to_numpy(dtype=int)
        p = frame["posterior_mean"].to_numpy(dtype=float)
        epi = frame["epistemic_variance"].to_numpy(dtype=float)
        incorrect = ((p >= 0.5) != y).astype(int)
        conf = confidence_uncertainty(p)
        log_epi = np.log(np.maximum(epi, np.finfo(float).tiny))
        return incorrect, conf, log_epi

    y_train, conf_train, epi_train = features(train)
    y_test, conf_test, epi_test = features(test)
    if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
        return {"status": "insufficient_error_classes"}

    baseline = LogisticRegression(max_iter=1000, solver="lbfgs")
    augmented = LogisticRegression(max_iter=1000, solver="lbfgs")
    baseline.fit(conf_train.reshape(-1, 1), y_train)
    augmented.fit(np.column_stack([conf_train, epi_train]), y_train)

    p_base = baseline.predict_proba(conf_test.reshape(-1, 1))[:, 1]
    p_aug = augmented.predict_proba(np.column_stack([conf_test, epi_test]))[:, 1]
    base_auc = float(roc_auc_score(y_test, p_base))
    aug_auc = float(roc_auc_score(y_test, p_aug))
    base_ll = float(log_loss(y_test, p_base))
    aug_ll = float(log_loss(y_test, p_aug))
    return {
        "status": "ok",
        "train_n": int(len(train)),
        "test_n": int(len(test)),
        "confidence_only_auroc": base_auc,
        "confidence_plus_epistemic_auroc": aug_auc,
        "delta_auroc": aug_auc - base_auc,
        "confidence_only_log_loss": base_ll,
        "confidence_plus_epistemic_log_loss": aug_ll,
        "delta_log_loss": aug_ll - base_ll,
    }


def main() -> None:
    args = parse_args()
    source = load_metadata(args)
    pre2022 = source[(source["date"] <= pd.Timestamp(args.train_end)) & (source["sector"] != "")].copy()
    sector_counts = pre2022["sector"].value_counts().sort_index()
    if sector_counts.empty:
        raise SystemExit("No non-empty pre-2022 sector labels found")
    threshold = float(sector_counts.quantile(args.sector_quantile))
    underrepresented_sectors = set(sector_counts[sector_counts <= threshold].index)
    well_represented_sectors = set(sector_counts[sector_counts > threshold].index)

    runs: list[dict[str, Any]] = []
    for seed in args.seeds:
        dev_path = prediction_path(args.development_dir, args.kl_beta, seed, args.mc_samples)
        holdout_path = prediction_path(args.holdout_dir, args.kl_beta, seed, args.mc_samples)
        if not dev_path.exists() or not holdout_path.exists():
            raise SystemExit(f"Missing prediction artifacts for seed {seed}")

        dev = attach_metadata(pd.read_csv(dev_path), source, start="2022-01-01", end="2023-12-31")
        holdout = attach_metadata(pd.read_csv(holdout_path), source, start="2024-01-01", end="2025-12-31")
        dev["underrepresented_sector"] = dev["sector"].isin(underrepresented_sectors)
        holdout["underrepresented_sector"] = holdout["sector"].isin(underrepresented_sectors)

        dev_dates = pd.to_datetime(dev["date"])
        well_2022 = dev[(dev_dates.dt.year == 2022) & (~dev["underrepresented_sector"])].copy()
        under_2023 = dev[(dev_dates.dt.year == 2023) & dev["underrepresented_sector"]].copy()
        well_pre2024 = dev[~dev["underrepresented_sector"]].copy()
        under_holdout = holdout[holdout["underrepresented_sector"]].copy()

        runs.append(
            {
                "seed": seed,
                "2023": {
                    "well_represented": basic_metrics(dev[(dev_dates.dt.year == 2023) & (~dev["underrepresented_sector"])]),
                    "underrepresented": basic_metrics(under_2023),
                    "incremental_meta_model_well_2022_to_underrepresented_2023": fit_and_evaluate_meta(
                        well_2022, under_2023
                    ),
                },
                "2024_2025": {
                    "well_represented": basic_metrics(holdout[~holdout["underrepresented_sector"]]),
                    "underrepresented": basic_metrics(under_holdout),
                    "incremental_meta_model_well_pre2024_to_underrepresented_2024_2025": fit_and_evaluate_meta(
                        well_pre2024, under_holdout
                    ),
                },
            }
        )

    aggregate: dict[str, Any] = {}
    for period, meta_key in (
        ("2023", "incremental_meta_model_well_2022_to_underrepresented_2023"),
        ("2024_2025", "incremental_meta_model_well_pre2024_to_underrepresented_2024_2025"),
    ):
        epi_minus_conf = [
            r[period]["underrepresented"].get("epistemic_minus_confidence_error_detection_auroc")
            for r in runs
        ]
        epi_minus_conf = [float(x) for x in epi_minus_conf if x is not None]
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
            "underrepresented_n_per_seed": [int(r[period]["underrepresented"].get("n", 0)) for r in runs],
            "epistemic_minus_confidence_error_detection_auroc_mean": mean(epi_minus_conf) if epi_minus_conf else None,
            "epistemic_minus_confidence_error_detection_auroc_std": pstdev(epi_minus_conf) if epi_minus_conf else None,
            "epistemic_beats_confidence_error_detection_seed_count": sum(x > 0 for x in epi_minus_conf),
            "incremental_meta_model_delta_auroc_mean": mean(delta_auc) if delta_auc else None,
            "incremental_meta_model_delta_auroc_std": pstdev(delta_auc) if delta_auc else None,
            "incremental_meta_model_delta_log_loss_mean": mean(delta_ll) if delta_ll else None,
            "incremental_epistemic_beats_auc_seed_count": sum(x > 0 for x in delta_auc),
            "incremental_epistemic_beats_log_loss_seed_count": sum(x < 0 for x in delta_ll),
        }

    summary = {
        "purpose": "pre-specified underrepresented-sector distribution-shift test",
        "shift_definition": "sector pre-2022 episode count at or below the 25th percentile across non-empty sectors",
        "sector_quantile": args.sector_quantile,
        "pre2022_sector_count": int(len(sector_counts)),
        "pre2022_sector_episode_count_quantile_threshold": threshold,
        "underrepresented_sectors": sorted(underrepresented_sectors),
        "well_represented_sector_count": int(len(well_represented_sectors)),
        "primary_criterion": (
            "epistemic uncertainty must add out-of-sample error-prediction information beyond confidence on underrepresented sectors"
        ),
        "methodological_status": (
            "targeted robustness/shift analysis; sector-frequency rule fixed before subgroup outcomes, but underlying dates were previously inspected"
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
