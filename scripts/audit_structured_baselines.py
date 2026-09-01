"""Audit whether ticker identity and recent volatility explain a downside target.

This script is deliberately run before any language-model training. It fits simple
chronological baselines to determine how much of the target is predictable from
asset identity and trailing market risk alone.

Example:
    uv run python scripts/audit_structured_baselines.py \
        --dataset artifacts/fnspid_slice_full/model_ready.csv \
        --prices data/FNSPID/Stock_price/full_history \
        --label-column label \
        --train-end 2021-12-31 \
        --test-start 2022-01-01 \
        --test-end 2022-12-31
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bayesian_lora_portfolio_risk.data.fnspid import load_fnspid_price_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--test-end", required=True)
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("artifacts/structured_baselines.json"))
    return parser.parse_args()


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    result = {
        "brier": float(brier_score_loss(y, p)),
        "nll": float(log_loss(y, np.column_stack([1.0 - p, p]), labels=[0, 1])),
    }
    if np.unique(y).size == 2:
        result["auroc"] = float(roc_auc_score(y, p))
    else:
        result["auroc"] = float("nan")
    return result


def _add_trailing_volatility(
    dataset: pd.DataFrame,
    prices_root: Path,
    *,
    window: int,
) -> pd.DataFrame:
    if window <= 1:
        raise ValueError("vol-window must be greater than 1")

    tickers = sorted(dataset["ticker"].astype(str).str.upper().unique())
    prices = load_fnspid_price_history(prices_root, tickers)
    prices = prices.sort_values(["ticker", "date"], kind="stable").copy()
    prices["return"] = prices.groupby("ticker", sort=False)["close"].pct_change()
    prices["trailing_vol"] = (
        prices.groupby("ticker", sort=False)["return"]
        .rolling(window=window, min_periods=window)
        .std()
        .reset_index(level=0, drop=True)
    )

    # shift one trading day so the feature is based strictly on returns known
    # before the labeled observation date.
    prices["trailing_vol"] = prices.groupby("ticker", sort=False)["trailing_vol"].shift(1)

    merged = dataset.merge(
        prices[["ticker", "date", "trailing_vol"]],
        on=["ticker", "date"],
        how="left",
        validate="many_to_one",
    )
    return merged


def main() -> None:
    args = parse_args()
    if not args.dataset.is_file():
        raise SystemExit(f"Dataset not found: {args.dataset}")
    if not args.prices.is_dir():
        raise SystemExit(f"Price directory not found: {args.prices}")

    frame = pd.read_csv(args.dataset)
    if args.label_column not in frame.columns:
        raise SystemExit(f"Label column not found: {args.label_column}")
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame[args.label_column] = pd.to_numeric(frame[args.label_column], errors="raise").astype(int)
    frame = _add_trailing_volatility(frame, args.prices, window=args.vol_window)
    frame = frame.dropna(subset=["trailing_vol"]).copy()

    train_end = pd.Timestamp(args.train_end)
    test_start = pd.Timestamp(args.test_start)
    test_end = pd.Timestamp(args.test_end)
    if not (train_end < test_start <= test_end):
        raise ValueError("require train-end < test-start <= test-end")

    train = frame[frame["date"] <= train_end].copy()
    test = frame[(frame["date"] >= test_start) & (frame["date"] <= test_end)].copy()
    if train.empty or test.empty:
        raise SystemExit("Train or test split is empty.")

    y_train = train[args.label_column].to_numpy()
    y_test = test[args.label_column].to_numpy()

    results: dict[str, object] = {
        "label_column": args.label_column,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_positive_rate": float(np.mean(y_train)),
        "test_positive_rate": float(np.mean(y_test)),
    }

    constant_p = np.full(len(test), float(np.mean(y_train)))
    results["constant"] = _metrics(y_test, constant_p)

    ticker_model = Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    [("ticker", OneHotEncoder(handle_unknown="ignore"), ["ticker"])],
                    remainder="drop",
                ),
            ),
            ("model", LogisticRegression(max_iter=1000, class_weight=None)),
        ]
    )
    ticker_model.fit(train[["ticker"]], y_train)
    ticker_p = ticker_model.predict_proba(test[["ticker"]])[:, 1]
    results["ticker_only"] = _metrics(y_test, ticker_p)

    vol_model = Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    [("vol", StandardScaler(), ["trailing_vol"])],
                    remainder="drop",
                ),
            ),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    vol_model.fit(train[["trailing_vol"]], y_train)
    vol_p = vol_model.predict_proba(test[["trailing_vol"]])[:, 1]
    results["volatility_only"] = _metrics(y_test, vol_p)

    combined_model = Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    [
                        ("ticker", OneHotEncoder(handle_unknown="ignore"), ["ticker"]),
                        ("vol", StandardScaler(), ["trailing_vol"]),
                    ]
                ),
            ),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    combined_model.fit(train[["ticker", "trailing_vol"]], y_train)
    combined_p = combined_model.predict_proba(test[["ticker", "trailing_vol"]])[:, 1]
    results["ticker_plus_volatility"] = _metrics(y_test, combined_p)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=True) + "\n")
    print(json.dumps(results, indent=2, allow_nan=True))
    print(f"\nArtifact written to {args.output}")


if __name__ == "__main__":
    main()
