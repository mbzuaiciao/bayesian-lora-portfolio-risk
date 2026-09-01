"""Build a volatility-normalized downside target from the Milestone 0 dataset.

The absolute 10% drawdown target is strongly predictable from ticker identity.
This script rescales each forward drawdown by trailing realized volatility known
strictly before the observation date:

    downside_score = -forward_drawdown / (trailing_vol * sqrt(horizon))

A threshold is selected using the training period only to achieve a requested
training positive rate. The selected threshold is then frozen and applied to all
later observations.

Example:
    uv run python scripts/build_volatility_normalized_target.py \
        --dataset artifacts/fnspid_slice_full/model_ready.csv \
        --prices data/FNSPID/Stock_price/full_history \
        --train-end 2021-12-31 \
        --target-positive-rate 0.125 \
        --output artifacts/fnspid_slice_full/model_ready_volnorm.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bayesian_lora_portfolio_risk.data.fnspid import load_fnspid_price_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--target-positive-rate", type=float, default=0.125)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/fnspid_slice_full/model_ready_volnorm.csv"),
    )
    return parser.parse_args()


def add_trailing_volatility(
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
    # Use only information available before the labeled observation date.
    prices["trailing_vol"] = prices.groupby("ticker", sort=False)["trailing_vol"].shift(1)

    return dataset.merge(
        prices[["ticker", "date", "trailing_vol"]],
        on=["ticker", "date"],
        how="left",
        validate="many_to_one",
    )


def main() -> None:
    args = parse_args()
    if not args.dataset.is_file():
        raise SystemExit(f"Dataset not found: {args.dataset}")
    if not args.prices.is_dir():
        raise SystemExit(f"Price directory not found: {args.prices}")
    if args.horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0.0 < args.target_positive_rate < 1.0:
        raise ValueError("target-positive-rate must lie strictly between 0 and 1")

    frame = pd.read_csv(args.dataset)
    required = {"ticker", "date", "forward_drawdown"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["forward_drawdown"] = pd.to_numeric(frame["forward_drawdown"], errors="raise")
    frame = add_trailing_volatility(frame, args.prices, window=args.vol_window)
    frame = frame.dropna(subset=["trailing_vol"]).copy()
    frame = frame[frame["trailing_vol"] > 0.0].copy()

    scale = frame["trailing_vol"] * np.sqrt(float(args.horizon))
    frame["normalized_downside_score"] = -frame["forward_drawdown"] / scale

    train_end = pd.Timestamp(args.train_end)
    train_mask = frame["date"] <= train_end
    if not train_mask.any():
        raise SystemExit("Training period contains no rows.")

    train_scores = frame.loc[train_mask, "normalized_downside_score"]
    quantile = 1.0 - args.target_positive_rate
    threshold = float(train_scores.quantile(quantile))
    frame["normalized_label"] = (frame["normalized_downside_score"] >= threshold).astype(int)

    yearly = (
        frame.assign(year=frame["date"].dt.year)
        .groupby("year", sort=True)["normalized_label"]
        .agg(observations="size", positive_rate="mean")
        .reset_index()
    )
    ticker = (
        frame.groupby("ticker", sort=True)["normalized_label"]
        .agg(observations="size", positive_rate="mean")
        .reset_index()
    )

    report = {
        "rows": int(len(frame)),
        "train_rows": int(train_mask.sum()),
        "train_end": args.train_end,
        "horizon": args.horizon,
        "vol_window": args.vol_window,
        "target_train_positive_rate": args.target_positive_rate,
        "selected_score_threshold": threshold,
        "actual_train_positive_rate": float(frame.loc[train_mask, "normalized_label"].mean()),
        "full_positive_rate": float(frame["normalized_label"].mean()),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2))
    print("\nYearly normalized-label summary")
    print(yearly.to_string(index=False))
    print("\nTicker normalized-label summary")
    print(ticker.to_string(index=False))
    print(f"\nDataset written to {args.output}")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
