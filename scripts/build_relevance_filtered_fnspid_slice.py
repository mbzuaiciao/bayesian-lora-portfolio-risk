"""Build a relevance-filtered FNSPID pilot slice using full article text when available.

This script is intentionally conservative. A row is retained only when the target
company is explicitly mentioned in the title or article body by a known company
alias, cashtag, or parenthesized ticker. It is designed for the current eight-name
pilot and should be generalized before broader experiments.

Example:
    uv run python scripts/build_relevance_filtered_fnspid_slice.py \
        --hf-news-repo benstaf/FNSPID-filtered-nasdaq-100 \
        --prices data/FNSPID/Stock_price/full_history \
        --tickers AAPL MSFT NVDA AMZN COST PEP CSCO GILD \
        --start 2018-01-01 --end 2023-06-30 \
        --output artifacts/fnspid_relevance_filtered
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

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
from bayesian_lora_portfolio_risk.data.fnspid import load_fnspid_price_history

COMPANY_ALIASES: dict[str, tuple[str, ...]] = {
    "AAPL": ("apple", "apple inc"),
    "MSFT": ("microsoft", "microsoft corp", "microsoft corporation"),
    "NVDA": ("nvidia", "nvidia corp", "nvidia corporation"),
    "AMZN": ("amazon", "amazon.com"),
    "COST": ("costco", "costco wholesale"),
    "PEP": ("pepsi", "pepsico"),
    "CSCO": ("cisco", "cisco systems"),
    "GILD": ("gilead", "gilead sciences"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-news-repo", required=True)
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--max-article-chars", type=int, default=6000)
    parser.add_argument("--output", type=Path, default=Path("artifacts/fnspid_relevance_filtered"))
    return parser.parse_args()


def _contains_target(text: str, ticker: str) -> bool:
    if not text:
        return False
    lowered = text.casefold()
    aliases = COMPANY_ALIASES.get(ticker, ())
    if any(re.search(rf"\b{re.escape(alias.casefold())}\b", lowered) for alias in aliases):
        return True

    # Plain ticker matching is unsafe for symbols that are ordinary words (e.g. COST).
    # Restrict ticker evidence to common finance-specific forms.
    ticker_re = re.escape(ticker)
    return bool(
        re.search(rf"\${ticker_re}\b", text, flags=re.IGNORECASE)
        or re.search(rf"\({ticker_re}\)", text, flags=re.IGNORECASE)
        or re.search(rf"\bNASDAQ\s*:\s*{ticker_re}\b", text, flags=re.IGNORECASE)
    )


def _compose_text(title: str, article: str, max_article_chars: int) -> str:
    title = title.strip()
    article = article.strip()
    if article:
        article = article[:max_article_chars]
    if title and article:
        return f"TITLE: {title}\n\nARTICLE: {article}"
    if article:
        return article
    return title


def _load_filtered_news(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install project dependencies with `uv sync`") from exc

    wanted = {ticker.strip().upper() for ticker in args.tickers}
    unknown = sorted(wanted.difference(COMPANY_ALIASES))
    if unknown:
        raise SystemExit(
            "No company aliases configured for: " + ", ".join(unknown) + ". "
            "Add aliases before using this conservative filter."
        )

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    if start > end:
        raise ValueError("start must not be after end")

    dataset = load_dataset(args.hf_news_repo, split=args.hf_split, streaming=True)
    records: list[dict[str, object]] = []
    scanned_matching_ticker = 0
    rejected_irrelevant = 0
    retained_by_ticker = {ticker: 0 for ticker in sorted(wanted)}

    for row in dataset:
        ticker = str(row.get("Stock_symbol", "")).strip().upper()
        if ticker not in wanted:
            continue
        timestamp = pd.to_datetime(row.get("Date"), utc=True, errors="coerce")
        if pd.isna(timestamp) or timestamp < start or timestamp > end:
            continue

        scanned_matching_ticker += 1
        title = str(row.get("Article_title") or "").strip()
        article = str(row.get("Article") or "").strip()
        evidence_text = f"{title}\n{article}"
        if not _contains_target(evidence_text, ticker):
            rejected_irrelevant += 1
            continue

        text = _compose_text(title, article, args.max_article_chars)
        if not text:
            continue
        records.append({"ticker": ticker, "timestamp": timestamp, "text": text})
        retained_by_ticker[ticker] += 1

    frame = pd.DataFrame.from_records(records, columns=["ticker", "timestamp", "text"])
    audit = {
        "matching_ticker_rows_scanned": scanned_matching_ticker,
        "retained_rows": int(len(frame)),
        "rejected_irrelevant_rows": rejected_irrelevant,
        "retention_rate": float(len(frame) / scanned_matching_ticker) if scanned_matching_ticker else 0.0,
        "retained_by_ticker": retained_by_ticker,
    }
    return frame, audit


def main() -> None:
    args = parse_args()
    if not args.prices.is_dir():
        raise SystemExit(f"Price directory not found: {args.prices}")
    if args.max_article_chars <= 0:
        raise ValueError("max-article-chars must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    news, relevance_audit = _load_filtered_news(args)
    if news.empty:
        raise SystemExit("No relevance-filtered news rows remain.")

    assigned = assign_information_date(news)
    asset_days = aggregate_asset_day_news(assigned)

    prices = load_fnspid_price_history(args.prices, args.tickers)
    price_labels = build_price_labels(prices, horizon=args.horizon, threshold=args.threshold)
    model_ready = align_news_to_trading_days(asset_days, price_labels)
    if model_ready.empty:
        raise SystemExit("No model-ready rows remain after alignment and labeling.")

    summary = summarize_model_dataset(model_ready)
    yearly = yearly_label_summary(model_ready)
    ticker = ticker_label_summary(model_ready)

    model_ready.to_csv(args.output / "model_ready.csv", index=False)
    yearly.to_csv(args.output / "yearly_label_summary.csv", index=False)
    ticker.to_csv(args.output / "ticker_label_summary.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "relevance_audit.json").write_text(json.dumps(relevance_audit, indent=2) + "\n")

    print("Relevance audit")
    print(json.dumps(relevance_audit, indent=2))
    print("\nModel-ready summary")
    print(json.dumps(summary, indent=2))
    print("\nYearly label summary")
    print(yearly.to_string(index=False))
    print("\nTicker label summary")
    print(ticker.to_string(index=False))
    print(f"\nArtifacts written to {args.output}")


if __name__ == "__main__":
    main()
