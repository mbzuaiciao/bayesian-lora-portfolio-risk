"""Build and audit a small FNSPID Milestone 0 slice.

Pilot example using a smaller Hugging Face derivative:
    uv run python scripts/audit_fnspid_slice.py \
        --hf-news-repo benstaf/FNSPID-filtered-nasdaq-100 \
        --prices data/FNSPID/Stock_price/full_history \
        --tickers AAPL MSFT NVDA AMZN COST PEP CSCO GILD \
        --start 2018-01-01 --end 2023-06-30 \
        --output artifacts/fnspid_slice

Canonical local-CSV mode remains available via --news.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_lora_portfolio_risk.data.audit import (
    summarize_model_dataset,
    ticker_label_summary,
    yearly_label_summary,
)
from bayesian_lora_portfolio_risk.data.builder import (
    aggregate_asset_day_news,
    align_news_to_trading_days,
    assign_information_date,
    build_price_labels,
)
from bayesian_lora_portfolio_risk.data.fnspid import (
    load_fnspid_news_slice,
    load_fnspid_price_history,
)
from bayesian_lora_portfolio_risk.data.hf_news import load_hf_fnspid_news_slice


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    news_group = parser.add_mutually_exclusive_group(required=True)
    news_group.add_argument("--news", type=Path, help="Raw canonical FNSPID news CSV")
    news_group.add_argument(
        "--hf-news-repo",
        help="Hugging Face dataset repo for pilot news ingestion",
    )
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--prices", type=Path, required=True, help="Directory of per-ticker price CSVs")
    parser.add_argument("--tickers", nargs="+", required=True, help="Ticker symbols for the audit slice")
    parser.add_argument("--start", required=True, help="Inclusive news start date")
    parser.add_argument("--end", required=True, help="Inclusive news end date")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--max-news-rows", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/fnspid_slice"))
    return parser.parse_args()


def _validate_paths(args: argparse.Namespace) -> None:
    if args.news is not None and not args.news.is_file():
        raise SystemExit(
            f"News CSV not found: {args.news}\n"
            "Either download the canonical FNSPID news file and pass its real path, "
            "or use --hf-news-repo for the smaller pilot dataset."
        )
    if not args.prices.is_dir():
        raise SystemExit(
            f"Price directory not found: {args.prices}\n"
            "Download and extract FNSPID Stock_price/full_history.zip first."
        )


def main() -> None:
    args = parse_args()
    _validate_paths(args)
    args.output.mkdir(parents=True, exist_ok=True)

    if args.hf_news_repo is not None:
        news = load_hf_fnspid_news_slice(
            repo_id=args.hf_news_repo,
            split=args.hf_split,
            tickers=args.tickers,
            start=args.start,
            end=args.end,
            max_rows=args.max_news_rows,
        )
    else:
        news = load_fnspid_news_slice(
            args.news,
            tickers=args.tickers,
            start=args.start,
            end=args.end,
            chunksize=args.chunksize,
            max_rows=args.max_news_rows,
        )

    if news.empty:
        raise SystemExit("No news rows matched the requested slice.")

    assigned = assign_information_date(news)
    asset_days = aggregate_asset_day_news(assigned)

    prices = load_fnspid_price_history(args.prices, args.tickers)
    price_labels = build_price_labels(
        prices,
        horizon=args.horizon,
        threshold=args.threshold,
    )
    dataset = align_news_to_trading_days(asset_days, price_labels)
    if dataset.empty:
        raise SystemExit("No model-ready rows remain after trading-day alignment and labeling.")

    summary = summarize_model_dataset(dataset)
    yearly = yearly_label_summary(dataset)
    ticker = ticker_label_summary(dataset)

    dataset.to_csv(args.output / "model_ready.csv", index=False)
    yearly.to_csv(args.output / "yearly_label_summary.csv", index=False)
    ticker.to_csv(args.output / "ticker_label_summary.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print("\nYearly label summary")
    print(yearly.to_string(index=False))
    print("\nTicker label summary")
    print(ticker.to_string(index=False))
    print(f"\nArtifacts written to {args.output}")


if __name__ == "__main__":
    main()
