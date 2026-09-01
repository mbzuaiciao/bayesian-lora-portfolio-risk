import pandas as pd
import pytest

from bayesian_lora_portfolio_risk.data.builder import (
    aggregate_asset_day_news,
    align_news_to_trading_days,
    assign_information_date,
    build_price_labels,
)


def test_after_close_news_moves_to_next_calendar_day() -> None:
    news = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "timestamp": ["2024-01-02T20:30:00Z", "2024-01-02T22:30:00Z"],
            "text": ["before close", "after close"],
        }
    )

    assigned = assign_information_date(news)

    assert assigned.loc[0, "information_date"] == pd.Timestamp("2024-01-02")
    assert assigned.loc[1, "information_date"] == pd.Timestamp("2024-01-03")


def test_asset_day_aggregation_prevents_article_duplication() -> None:
    news = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "MSFT"],
            "information_date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-02"]),
            "text": ["headline one", "headline two", "other stock"],
        }
    )

    aggregated = aggregate_asset_day_news(news)
    apple = aggregated[aggregated["ticker"] == "AAPL"].iloc[0]

    assert apple["article_count"] == 2
    assert apple["text"] == "headline one\n\nheadline two"
    assert len(aggregated) == 2


def test_price_labels_are_computed_separately_per_ticker() -> None:
    prices = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT", "MSFT"],
            "date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                ]
            ),
            "close": [100.0, 80.0, 90.0, 100.0, 101.0, 102.0],
        }
    )

    labeled = build_price_labels(prices, horizon=1, threshold=0.10)

    apple_first = labeled[(labeled["ticker"] == "AAPL")].iloc[0]
    msft_first = labeled[(labeled["ticker"] == "MSFT")].iloc[0]
    assert apple_first["label"] == 1.0
    assert msft_first["label"] == 0.0


def test_weekend_news_aligns_to_next_trading_day() -> None:
    asset_days = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "information_date": pd.to_datetime(["2024-01-06"]),  # Saturday
            "text": ["weekend story"],
            "article_count": [1],
        }
    )
    prices = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "date": pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"]),
            "close": [100.0, 100.0, 80.0],
        }
    )
    labeled = build_price_labels(prices, horizon=1, threshold=0.10)

    result = align_news_to_trading_days(asset_days, labeled)

    assert len(result) == 1
    assert result.loc[0, "date"] == pd.Timestamp("2024-01-08")
    assert result.loc[0, "label"] == 1.0


def test_incomplete_future_window_is_removed_from_training_examples() -> None:
    asset_days = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "information_date": pd.to_datetime(["2024-01-09"]),
            "text": ["late sample"],
            "article_count": [1],
        }
    )
    prices = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "date": pd.to_datetime(["2024-01-08", "2024-01-09"]),
            "close": [100.0, 100.0],
        }
    )
    labeled = build_price_labels(prices, horizon=1, threshold=0.10)

    result = align_news_to_trading_days(asset_days, labeled)

    assert result.empty


def test_duplicate_price_dates_are_rejected() -> None:
    prices = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "close": [100.0, 101.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate"):
        build_price_labels(prices, horizon=1)
