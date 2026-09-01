from pathlib import Path

import pandas as pd
import pytest

from bayesian_lora_portfolio_risk.data.fnspid import (
    load_fnspid_news_slice,
    load_fnspid_price_history,
)


def test_load_fnspid_news_slice_filters_and_prefers_title(tmp_path: Path) -> None:
    path = tmp_path / "news.csv"
    pd.DataFrame(
        {
            "Date": ["2020-01-02 15:00:00", "2020-01-03 15:00:00", "2021-01-01 15:00:00"],
            "Stock_symbol": ["AAPL", "MSFT", "AAPL"],
            "Article_title": ["Apple title", "Microsoft title", "Later Apple title"],
            "Lsa_summary": ["Apple summary", "Microsoft summary", "Later Apple summary"],
            "Article": ["Apple body", "Microsoft body", "Later Apple body"],
        }
    ).to_csv(path, index=False)

    result = load_fnspid_news_slice(
        path,
        tickers=["aapl"],
        start="2020-01-01",
        end="2020-12-31 23:59:59",
        chunksize=1,
    )

    assert list(result["ticker"]) == ["AAPL"]
    assert list(result["text"]) == ["Apple title"]
    assert str(result.loc[0, "timestamp"].tz) == "UTC"


def test_load_fnspid_news_slice_falls_back_to_summary_then_article(tmp_path: Path) -> None:
    path = tmp_path / "news.csv"
    pd.DataFrame(
        {
            "Date": ["2020-01-02", "2020-01-03"],
            "Stock_symbol": ["AAPL", "AAPL"],
            "Article_title": [None, None],
            "Lsa_summary": ["Summary", None],
            "Article": ["Body one", "Body two"],
        }
    ).to_csv(path, index=False)

    result = load_fnspid_news_slice(path, tickers=["AAPL"], chunksize=2)
    assert list(result["text"]) == ["Summary", "Body two"]


def test_load_fnspid_price_history_normalizes_selected_tickers(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "Date": ["2020-01-02", "2020-01-03"],
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Volume": [10, 11],
        }
    ).to_csv(tmp_path / "aapl.csv", index=False)

    result = load_fnspid_price_history(tmp_path, ["AAPL"])
    assert list(result.columns) == ["ticker", "date", "close"]
    assert list(result["ticker"].unique()) == ["AAPL"]
    assert list(result["close"]) == [101.0, 102.0]


def test_load_fnspid_price_history_reports_missing_ticker(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="MSFT"):
        load_fnspid_price_history(tmp_path, ["MSFT"])
