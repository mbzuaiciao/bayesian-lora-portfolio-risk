"""Hugging Face ingestion for a smaller FNSPID pilot dataset."""

from __future__ import annotations

import math
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

    When ``max_rows`` is set together with an explicit ticker list, the cap is
    distributed approximately evenly across requested tickers. This avoids a
    streaming-order artifact in which the first ticker in the dataset consumes
    the entire smoke-test budget.
    """
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive when provided")

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("install project dependencies with `uv sync`") from exc

    wanted: set[str] | None = None
    if tickers is not None:
        wanted = {str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()}
        if not wanted:
            wanted = None

    start_ts = pd.Timestamp(start, tz="UTC") if start is not None else None
    end_ts = pd.Timestamp(end, tz="UTC") if end is not None else None
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError("start must not be after end")

    per_ticker_cap: int | None = None
    counts: dict[str, int] = {}
    if max_rows is not None and wanted is not None:
        per_ticker_cap = max(1, math.ceil(max_rows / len(wanted)))
        counts = {ticker: 0 for ticker in wanted}

    dataset = load_dataset(repo_id, split=split, streaming=True)
    records: list[dict[str, object]] = []

    for row in dataset:
        ticker = str(row.get("Stock_symbol", "")).strip().upper()
        if not ticker or (wanted is not None and ticker not in wanted):
            continue
        if per_ticker_cap is not None and counts[ticker] >= per_ticker_cap:
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
        if per_ticker_cap is not None:
            counts[ticker] += 1
            if all(count >= per_ticker_cap for count in counts.values()):
                break
        elif max_rows is not None and len(records) >= max_rows:
            break

    if not records:
        return pd.DataFrame(columns=["ticker", "timestamp", "text"])

    frame = pd.DataFrame.from_records(records)
    if max_rows is not None and len(frame) > max_rows:
        # Preserve balance while respecting the exact global cap.
        frame = (
            frame.sort_values(["ticker", "timestamp"], kind="stable")
            .groupby("ticker", group_keys=False, sort=True)
            .head(per_ticker_cap if per_ticker_cap is not None else max_rows)
            .head(max_rows)
            .reset_index(drop=True)
        )
    return frame
