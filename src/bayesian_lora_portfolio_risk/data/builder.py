"""Leakage-safe asset-day construction for Milestone 0.

The builder accepts normalized news and price frames rather than depending on a
particular storage format. Raw FNSPID columns should be renamed at the ingestion
boundary before calling these functions.
"""

from __future__ import annotations

import pandas as pd

from bayesian_lora_portfolio_risk.data.labels import (
    forward_drawdown_label,
    forward_max_drawdown,
)

_REQUIRED_NEWS_COLUMNS = {"ticker", "timestamp", "text"}
_REQUIRED_PRICE_COLUMNS = {"ticker", "date", "close"}


def normalize_news_frame(news: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a news frame to ticker/timestamp/text columns."""
    missing = _REQUIRED_NEWS_COLUMNS.difference(news.columns)
    if missing:
        raise ValueError(f"missing required news columns: {sorted(missing)}")

    frame = news.loc[:, ["ticker", "timestamp", "text"]].copy()
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    frame["text"] = frame["text"].fillna("").astype(str).str.strip()
    frame = frame[(frame["ticker"] != "") & (frame["text"] != "")]
    return frame.sort_values(["ticker", "timestamp"], kind="stable").reset_index(drop=True)


def assign_information_date(
    news: pd.DataFrame,
    *,
    market_timezone: str = "America/New_York",
    cutoff_hour: int = 16,
) -> pd.DataFrame:
    """Assign each article to a conservative calendar information date.

    Articles timestamped at or after the market-close cutoff are assigned to the
    following calendar day. Weekend/holiday alignment to an actual trading day is
    handled when the frame is merged against the trading calendar from prices.
    """
    if not 0 <= cutoff_hour <= 23:
        raise ValueError("cutoff_hour must lie in [0, 23]")

    frame = normalize_news_frame(news)
    local = frame["timestamp"].dt.tz_convert(market_timezone)
    information_date = local.dt.normalize().dt.tz_localize(None)
    after_cutoff = local.dt.hour >= cutoff_hour
    information_date = information_date + pd.to_timedelta(after_cutoff.astype(int), unit="D")
    frame["information_date"] = information_date
    return frame


def aggregate_asset_day_news(news: pd.DataFrame, *, separator: str = "\n\n") -> pd.DataFrame:
    """Aggregate eligible articles into one text observation per ticker/date."""
    if "information_date" not in news.columns:
        raise ValueError("news must contain information_date")
    required = {"ticker", "information_date", "text"}
    missing = required.difference(news.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    frame = news.loc[:, ["ticker", "information_date", "text"]].copy()
    frame["text"] = frame["text"].fillna("").astype(str).str.strip()
    frame = frame[frame["text"] != ""]

    grouped = (
        frame.groupby(["ticker", "information_date"], sort=True)["text"]
        .agg(lambda values: separator.join(values))
        .rename("text")
        .reset_index()
    )
    counts = (
        frame.groupby(["ticker", "information_date"], sort=True)
        .size()
        .rename("article_count")
        .reset_index()
    )
    return grouped.merge(counts, on=["ticker", "information_date"], validate="one_to_one")


def build_price_labels(
    prices: pd.DataFrame,
    *,
    horizon: int = 20,
    threshold: float = 0.10,
) -> pd.DataFrame:
    """Attach forward drawdown and binary event labels within each ticker."""
    missing = _REQUIRED_PRICE_COLUMNS.difference(prices.columns)
    if missing:
        raise ValueError(f"missing required price columns: {sorted(missing)}")

    frame = prices.loc[:, ["ticker", "date", "close"]].copy()
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["close"] = pd.to_numeric(frame["close"], errors="raise")
    frame = frame.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)

    if frame.duplicated(["ticker", "date"]).any():
        raise ValueError("price frame contains duplicate ticker/date rows")

    pieces: list[pd.DataFrame] = []
    for _, group in frame.groupby("ticker", sort=False):
        group = group.copy()
        values = group["close"].to_numpy(dtype=float)
        group["forward_drawdown"] = forward_max_drawdown(values, horizon=horizon)
        group["label"] = forward_drawdown_label(
            values,
            horizon=horizon,
            threshold=threshold,
        )
        pieces.append(group)

    if not pieces:
        return frame.assign(forward_drawdown=pd.Series(dtype=float), label=pd.Series(dtype=float))
    return pd.concat(pieces, ignore_index=True)


def align_news_to_trading_days(
    asset_days: pd.DataFrame,
    price_labels: pd.DataFrame,
) -> pd.DataFrame:
    """Map each asset-day information date to that ticker's next trading day.

    This prevents weekend, holiday, and after-close news from being paired with a
    price observation that predates the information.
    """
    required_news = {"ticker", "information_date", "text", "article_count"}
    missing_news = required_news.difference(asset_days.columns)
    if missing_news:
        raise ValueError(f"missing asset-day columns: {sorted(missing_news)}")

    required_prices = {"ticker", "date", "close", "forward_drawdown", "label"}
    missing_prices = required_prices.difference(price_labels.columns)
    if missing_prices:
        raise ValueError(f"missing price-label columns: {sorted(missing_prices)}")

    news = asset_days.sort_values(["ticker", "information_date"], kind="stable").copy()
    prices = price_labels.sort_values(["ticker", "date"], kind="stable").copy()

    aligned_parts: list[pd.DataFrame] = []
    for ticker, news_group in news.groupby("ticker", sort=False):
        price_group = prices[prices["ticker"] == ticker]
        if price_group.empty:
            continue
        merged = pd.merge_asof(
            news_group.sort_values("information_date"),
            price_group.sort_values("date"),
            left_on="information_date",
            right_on="date",
            direction="forward",
            allow_exact_matches=True,
            suffixes=("", "_price"),
        )
        merged["ticker"] = ticker
        aligned_parts.append(merged)

    if not aligned_parts:
        return pd.DataFrame(
            columns=[
                "ticker",
                "information_date",
                "date",
                "text",
                "article_count",
                "close",
                "forward_drawdown",
                "label",
            ]
        )

    result = pd.concat(aligned_parts, ignore_index=True)
    result = result.dropna(subset=["date", "label"]).copy()
    return result[
        [
            "ticker",
            "information_date",
            "date",
            "text",
            "article_count",
            "close",
            "forward_drawdown",
            "label",
        ]
    ].reset_index(drop=True)
