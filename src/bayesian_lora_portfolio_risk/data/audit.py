"""Dataset audit summaries required before Milestone 0 model training."""

from __future__ import annotations

from typing import Any

import pandas as pd


def summarize_model_dataset(dataset: pd.DataFrame) -> dict[str, Any]:
    """Return core coverage and class-balance statistics for model-ready rows."""
    required = {
        "ticker",
        "information_date",
        "date",
        "text",
        "article_count",
        "label",
    }
    missing = required.difference(dataset.columns)
    if missing:
        raise ValueError(f"missing model-dataset columns: {sorted(missing)}")

    frame = dataset.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["information_date"] = pd.to_datetime(frame["information_date"], errors="raise")
    frame["text"] = frame["text"].fillna("").astype(str)

    if frame.empty:
        return {
            "rows": 0,
            "tickers": 0,
            "start_date": None,
            "end_date": None,
            "positive_rate": None,
            "mean_articles_per_day": None,
            "median_text_chars": None,
        }

    labels = pd.to_numeric(frame["label"], errors="raise")
    if not labels.isin([0.0, 1.0]).all():
        raise ValueError("label must contain only 0/1 values in model-ready data")

    return {
        "rows": int(len(frame)),
        "tickers": int(frame["ticker"].nunique()),
        "start_date": frame["date"].min().date().isoformat(),
        "end_date": frame["date"].max().date().isoformat(),
        "positive_rate": float(labels.mean()),
        "mean_articles_per_day": float(pd.to_numeric(frame["article_count"]).mean()),
        "median_text_chars": float(frame["text"].str.len().median()),
    }


def yearly_label_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    """Summarize observation count and positive-event rate by calendar year."""
    required = {"date", "label"}
    missing = required.difference(dataset.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    frame = dataset.loc[:, ["date", "label"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["label"] = pd.to_numeric(frame["label"], errors="raise")
    frame["year"] = frame["date"].dt.year

    result = (
        frame.groupby("year", sort=True)["label"]
        .agg(observations="size", positive_rate="mean")
        .reset_index()
    )
    result["observations"] = result["observations"].astype(int)
    return result


def ticker_label_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    """Summarize observation count and positive-event rate by ticker."""
    required = {"ticker", "label"}
    missing = required.difference(dataset.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    frame = dataset.loc[:, ["ticker", "label"]].copy()
    frame["label"] = pd.to_numeric(frame["label"], errors="raise")
    result = (
        frame.groupby("ticker", sort=True)["label"]
        .agg(observations="size", positive_rate="mean")
        .reset_index()
    )
    result["observations"] = result["observations"].astype(int)
    return result
