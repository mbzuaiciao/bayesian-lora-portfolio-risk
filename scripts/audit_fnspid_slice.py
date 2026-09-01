"""Build and audit a small FNSPID Milestone 0 slice.

Example:
    uv run python scripts/audit_fnspid_slice.py \
        --news data/FNSPID/Stock_news/nasdaq_exteral_data.csv \
        --prices data/FNSPID/Stock_price/full_history \
        --tickers AAPL MSFT NVDA JPM XOM \
        --start 2018-01-01 --end 2023-06-30 \
        --output artifacts/fnspid_slice
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--news", type=Path, required=True, help="Raw FNSPID news CSV")
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


def validate_input_paths(news_path: Path, prices_path: Path) -> None:
    """Fail early with actionable guidance when FNSPID data is not present."""
    if not news_path.is_file():
        raise SystemExit(
            "FNSPID news CSV not found:\n"
            f"  {news_path}\n\n"
            "Download/place the FNSPID news CSV locally, or pass its actual location with --news.\n"
            "The commonly documented filename is 'nasdaq_exteral_data.csv' (spelling as released).\n\n"
            "Example:\n"
            "  --news /absolute/path/to/nasdaq_exteral_data.csv\n\n"
            "To locate an existing download on macOS, try:\n"
            "  find ~/Downloads ~/Documents -name 'nasdaq_exteral_data.csv' 2>/dev/null\n"
        )

    if not prices_path.is_dir():
        raise SystemExit(
            "FNSPID price-history directory not found:\n"
            f"  {prices_path}\n\n"
            "Extract the FNSPID full_history archive, or pass the directory containing the "
            "per-ticker price CSV files with --prices.\n\n"
            "Example:\n"
            "  --prices /absolute/path/to/full_history\n"
        )


def main() -> None:
    args = parse_args()
    validate_input_paths(args.news, args.prices)
    args.output.mkdir(parents=True, exist_ok=True)

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
