"""Hugging Face ingestion for a smaller FNSPID pilot dataset."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def load_hf_fnspid_news_slice(
    *,
    repo_id: str = "benstaf/FNSPID-filtered-nasdaq-100",
    split: str = "train",
    tickers: Iterable[str] | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Load a filtered FNSPID-derived Hugging Face split into normalized columns.

    This helper is intended for Milestone 0 pilot/audit work. The default repo is
    a third-party Nasdaq-100-filtered derivative of FNSPID, not the canonical
    release. Final experiments should be validated against the official FNSPID
    source.
    """
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive when provided")

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("install project dependencies with `uv sync`") from exc

    wanted = None
    if tickers is not None:
        wanted = {str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()}
        if not wanted:
            wanted = None

    start_ts = pd.Timestamp(start, tz="UTC") if start is not None else None
    end_ts = pd.Timestamp(end, tz="UTC") if end is not None else None
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError("start must not be after end")

    dataset = load_dataset(repo_id, split=split, streaming=True)
    records: list[dict[str, object]] = []

    for row in dataset:
        ticker = str(row.get("Stock_symbol", "")).strip().upper()
        if not ticker or (wanted is not None and ticker not in wanted):
            continue

        timestamp = pd.to_datetime(row.get("Date"), utc=True, errors="coerce")
        if pd.isna(timestamp):
            continue
        if start_ts is not None and timestamp < start_ts:
            continue
        if end_ts is not None and timestamp > end_ts:
            continue

        text = ""
        for key in (
            "Article_title",
            "Lsa_summary",
            "Luhn_summary",
            "Textrank_summary",
            "Lexrank_summary",
            "Article",
        ):
            value = row.get(key)
            if value is not None and str(value).strip():
                text = str(value).strip()
                break
        if not text:
            continue

        records.append({"ticker": ticker, "timestamp": timestamp, "text": text})
        if max_rows is not None and len(records) >= max_rows:
            break

    if not records:
        return pd.DataFrame(columns=["ticker", "timestamp", "text"])
    return pd.DataFrame.from_records(records)
