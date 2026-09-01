"""Chunked ingestion helpers for local FNSPID files.

The raw FNSPID release is large, so Milestone 0 starts from a filtered local
slice rather than loading the full news corpus into memory.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

_NEWS_ALIASES = {
    "date": "timestamp",
    "Date": "timestamp",
    "stock_symbol": "ticker",
    "Stock_symbol": "ticker",
    "Stock_symbol": "ticker",
    "Article_title": "title",
    "article_title": "title",
    "Article": "article",
    "article": "article",
    "Lsa_summary": "summary",
    "Luhn_summary": "summary",
    "Textrank_summary": "summary",
}


def _normalize_tickers(tickers: Iterable[str] | None) -> set[str] | None:
    if tickers is None:
        return None
    values = {str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()}
    return values or None


def _pick_column(columns: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def load_fnspid_news_slice(
    path: str | Path,
    *,
    tickers: Iterable[str] | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    chunksize: int = 100_000,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Read a filtered slice from a raw FNSPID news CSV in chunks.

    The output follows the repository's normalized news contract:
    ``ticker``, ``timestamp``, ``text``. Text preference is title, then summary,
    then article body. Date filtering is inclusive.
    """
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive when provided")

    wanted = _normalize_tickers(tickers)
    start_ts = pd.Timestamp(start, tz="UTC") if start is not None else None
    end_ts = pd.Timestamp(end, tz="UTC") if end is not None else None
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError("start must not be after end")

    pieces: list[pd.DataFrame] = []
    kept = 0

    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        ticker_col = _pick_column(chunk.columns, ("Stock_symbol", "stock_symbol", "ticker"))
        date_col = _pick_column(chunk.columns, ("Date", "date", "timestamp"))
        title_col = _pick_column(chunk.columns, ("Article_title", "article_title", "title"))
        summary_col = _pick_column(
            chunk.columns,
            ("Lsa_summary", "Luhn_summary", "Textrank_summary", "summary"),
        )
        article_col = _pick_column(chunk.columns, ("Article", "article", "text"))

        if ticker_col is None or date_col is None:
            raise ValueError("news CSV must contain ticker and date columns")
        if title_col is None and summary_col is None and article_col is None:
            raise ValueError("news CSV must contain title, summary, or article text")

        ticker = chunk[ticker_col].astype(str).str.strip().str.upper()
        timestamp = pd.to_datetime(chunk[date_col], utc=True, errors="coerce")

        text = pd.Series("", index=chunk.index, dtype="object")
        for column in (title_col, summary_col, article_col):
            if column is None:
                continue
            candidate = chunk[column].fillna("").astype(str).str.strip()
            text = text.where(text.str.len() > 0, candidate)

        mask = timestamp.notna() & ticker.ne("") & text.ne("")
        if wanted is not None:
            mask &= ticker.isin(wanted)
        if start_ts is not None:
            mask &= timestamp >= start_ts
        if end_ts is not None:
            mask &= timestamp <= end_ts

        part = pd.DataFrame(
            {
                "ticker": ticker[mask],
                "timestamp": timestamp[mask],
                "text": text[mask],
            }
        )
        if part.empty:
            continue

        if max_rows is not None:
            remaining = max_rows - kept
            part = part.head(remaining)
        pieces.append(part)
        kept += len(part)

        if max_rows is not None and kept >= max_rows:
            break

    if not pieces:
        return pd.DataFrame(columns=["ticker", "timestamp", "text"])
    return pd.concat(pieces, ignore_index=True)


def load_fnspid_price_history(
    root: str | Path,
    tickers: Iterable[str],
) -> pd.DataFrame:
    """Load selected per-ticker FNSPID price CSVs into normalized form.

    Files are resolved case-insensitively by stem and expected to contain at
    least Date and Close (or lowercase equivalents).
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(root_path)

    requested = _normalize_tickers(tickers)
    if requested is None:
        raise ValueError("at least one ticker is required")

    by_stem = {path.stem.upper(): path for path in root_path.glob("*.csv")}
    pieces: list[pd.DataFrame] = []
    missing: list[str] = []

    for ticker in sorted(requested):
        path = by_stem.get(ticker)
        if path is None:
            missing.append(ticker)
            continue
        frame = pd.read_csv(path, low_memory=False)
        date_col = _pick_column(frame.columns, ("Date", "date"))
        close_col = _pick_column(frame.columns, ("Close", "close", "Adj Close", "adj_close"))
        if date_col is None or close_col is None:
            raise ValueError(f"{path.name} must contain Date and Close columns")

        piece = pd.DataFrame(
            {
                "ticker": ticker,
                "date": pd.to_datetime(frame[date_col], errors="coerce"),
                "close": pd.to_numeric(frame[close_col], errors="coerce"),
            }
        ).dropna(subset=["date", "close"])
        pieces.append(piece)

    if missing:
        raise FileNotFoundError(f"missing price files for tickers: {missing}")
    if not pieces:
        return pd.DataFrame(columns=["ticker", "date", "close"])
    return pd.concat(pieces, ignore_index=True)
