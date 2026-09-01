import pandas as pd
import pytest

from bayesian_lora_portfolio_risk.data.audit import (
    summarize_model_dataset,
    ticker_label_summary,
    yearly_label_summary,
)


def _dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "MSFT"],
            "information_date": pd.to_datetime(["2023-12-29", "2024-01-02", "2024-01-02"]),
            "date": pd.to_datetime(["2023-12-29", "2024-01-02", "2024-01-02"]),
            "text": ["a", "longer", "news"],
            "article_count": [1, 2, 1],
            "label": [0.0, 1.0, 0.0],
        }
    )


def test_summarize_model_dataset_reports_core_statistics() -> None:
    summary = summarize_model_dataset(_dataset())

    assert summary["rows"] == 3
    assert summary["tickers"] == 2
    assert summary["start_date"] == "2023-12-29"
    assert summary["end_date"] == "2024-01-02"
    assert summary["positive_rate"] == pytest.approx(1 / 3)
    assert summary["mean_articles_per_day"] == pytest.approx(4 / 3)
    assert summary["median_text_chars"] == 4.0


def test_yearly_label_summary_preserves_temporal_class_balance() -> None:
    result = yearly_label_summary(_dataset())

    assert result["year"].tolist() == [2023, 2024]
    assert result["observations"].tolist() == [1, 2]
    assert result["positive_rate"].tolist() == [0.0, 0.5]


def test_ticker_label_summary_reports_per_asset_balance() -> None:
    result = ticker_label_summary(_dataset())

    apple = result[result["ticker"] == "AAPL"].iloc[0]
    msft = result[result["ticker"] == "MSFT"].iloc[0]
    assert apple["observations"] == 2
    assert apple["positive_rate"] == 0.5
    assert msft["positive_rate"] == 0.0


def test_model_ready_summary_rejects_nonbinary_labels() -> None:
    dataset = _dataset()
    dataset.loc[0, "label"] = 2.0

    with pytest.raises(ValueError, match="0/1"):
        summarize_model_dataset(dataset)
