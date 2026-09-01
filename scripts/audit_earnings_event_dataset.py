"""Audit the candidate earnings-event dataset before any modeling.

The script streams a manageable sample from Hugging Face, summarizes schema,
missingness, timestamp coverage, text lengths, and label distributions, and writes
human-readable example episodes for manual review.

Example:
    uv run python scripts/audit_earnings_event_dataset.py \
        --repo-id RudrakshNanavaty/earnings-call-data \
        --split train \
        --max-rows 20000 \
        --samples 12 \
        --output-dir artifacts/earnings_event_audit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="RudrakshNanavaty/earnings-call-data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-rows", type=int, default=20_000)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/earnings_event_audit"),
    )
    return parser.parse_args()


def _safe_len(value: object) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    return len(str(value))


def _numeric_summary(frame: pd.DataFrame, column: str) -> dict[str, float | int | None]:
    if column not in frame.columns:
        return {}
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "p05": float(values.quantile(0.05)),
        "p50": float(values.quantile(0.50)),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    if args.max_rows <= 0 or args.samples <= 0:
        raise ValueError("max-rows and samples must be positive")

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install project dependencies with `uv sync`") from exc

    print(f"Streaming up to {args.max_rows} rows from {args.repo_id} [{args.split}]...")
    dataset = load_dataset(args.repo_id, split=args.split, streaming=True)
    records: list[dict[str, object]] = []
    for row in dataset:
        records.append(dict(row))
        if len(records) >= args.max_rows:
            break

    if not records:
        raise SystemExit("No rows were loaded from the candidate dataset")

    frame = pd.DataFrame.from_records(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for column in ("date", "earnings_date", "next_qtr_date"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")

    text_columns = [
        column
        for column in ("earnings_transcript", "press_release_8k_body", "press_release_ex991", "press_release_ex992")
        if column in frame.columns
    ]
    for column in text_columns:
        frame[f"{column}_chars"] = frame[column].map(_safe_len)

    missingness = (
        frame.isna().mean().sort_values(ascending=False).rename("missing_fraction").reset_index()
    )
    missingness.columns = ["column", "missing_fraction"]
    missingness.to_csv(args.output_dir / "missingness.csv", index=False)

    text_summary: dict[str, dict[str, float | int]] = {}
    for column in text_columns:
        lengths = frame[f"{column}_chars"]
        nonempty = lengths[lengths > 0]
        text_summary[column] = {
            "nonempty_rows": int((lengths > 0).sum()),
            "nonempty_fraction": float((lengths > 0).mean()),
            "median_chars_nonempty": float(nonempty.median()) if not nonempty.empty else 0.0,
            "p95_chars_nonempty": float(nonempty.quantile(0.95)) if not nonempty.empty else 0.0,
        }

    summary: dict[str, object] = {
        "repo_id": args.repo_id,
        "split": args.split,
        "rows_loaded": int(len(frame)),
        "columns": list(frame.columns),
        "unique_symbols": int(frame["symbol"].nunique()) if "symbol" in frame.columns else None,
        "date_min": str(frame["date"].min()) if "date" in frame.columns else None,
        "date_max": str(frame["date"].max()) if "date" in frame.columns else None,
        "text": text_summary,
        "move_1d": _numeric_summary(frame, "move_1d"),
        "move_30d": _numeric_summary(frame, "move_30d"),
        "move_next_qtr": _numeric_summary(frame, "move_next_qtr"),
    }

    if "move_1d_direction" in frame.columns:
        counts = frame["move_1d_direction"].value_counts(dropna=False)
        summary["move_1d_direction_counts"] = {str(k): int(v) for k, v in counts.items()}
    if "sentiment_label" in frame.columns:
        counts = frame["sentiment_label"].value_counts(dropna=False)
        summary["sentiment_label_counts"] = {str(k): int(v) for k, v in counts.items()}

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    sample_n = min(args.samples, len(frame))
    sampled = frame.sample(n=sample_n, random_state=args.seed).copy()
    sample_columns = [
        column
        for column in (
            "episode_id",
            "symbol",
            "company_name",
            "year",
            "quarter",
            "date",
            "earnings_date",
            "sector",
            "move_1d",
            "move_30d",
            "move_next_qtr",
            "move_1d_direction",
            "price_momentum_30d",
            "price_momentum_90d",
            "pct_from_52w_high_pt",
        )
        if column in sampled.columns
    ]
    sampled[sample_columns].to_csv(args.output_dir / "sampled_episodes.csv", index=False)

    lines = [
        "# Earnings-event dataset audit",
        "",
        f"Repository: `{args.repo_id}`",
        f"Rows inspected: {len(frame)}",
        "",
        "Manual questions for each episode:",
        "",
        "1. Does the transcript clearly correspond to the named company and quarter?",
        "2. Is the event timestamp precise enough to define the first tradable post-event price?",
        "3. Does the text appear complete rather than truncated or duplicated?",
        "4. Does the provided return label look numerically plausible from the OHLCV anchors?",
        "5. Are any structured fields obviously post-event information that must be excluded from model inputs?",
        "",
    ]

    for _, row in sampled.iterrows():
        symbol = row.get("symbol", "NA")
        date = row.get("date", "NA")
        lines.extend([f"## {symbol} — {date}", ""])
        for column in sample_columns:
            lines.append(f"- {column}: {row.get(column)}")
        for column in text_columns:
            text = "" if pd.isna(row.get(column)) else str(row.get(column))
            lines.extend(["", f"### {column}", "", text[:5000] if text else "[EMPTY]", ""])

    (args.output_dir / "sampled_episodes.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
