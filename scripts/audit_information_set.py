"""Inspect whether sampled asset-day information sets are meaningful for the downside task.

The audit samples positive and negative rows from a model-ready dataset and writes
both a compact CSV and a human-readable Markdown report. It is intended to answer
whether the aggregated text plausibly contains information relevant to the later
realized downside event before any further model work.

Example:
    uv run python scripts/audit_information_set.py \
        --dataset artifacts/fnspid_slice_full/model_ready_volnorm.csv \
        --label-column normalized_label \
        --samples-per-class 12 \
        --seed 7 \
        --output-dir artifacts/information_set_audit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--label-column", default="normalized_label")
    parser.add_argument("--samples-per-class", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-text-chars", type=int, default=3000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/information_set_audit"),
    )
    return parser.parse_args()


def _sample_class(frame: pd.DataFrame, label_column: str, label: int, n: int, seed: int) -> pd.DataFrame:
    pool = frame[frame[label_column] == label]
    if pool.empty:
        return pool.copy()
    return pool.sample(n=min(n, len(pool)), random_state=seed)


def _fmt_float(value: object, digits: int = 4) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a simple Markdown table without pandas' optional tabulate dependency."""
    columns = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def main() -> None:
    args = parse_args()
    if not args.dataset.is_file():
        raise SystemExit(f"Dataset not found: {args.dataset}")
    if args.samples_per_class <= 0:
        raise ValueError("samples-per-class must be positive")
    if args.max_text_chars <= 0:
        raise ValueError("max-text-chars must be positive")

    frame = pd.read_csv(args.dataset)
    required = {
        "ticker",
        "information_date",
        "date",
        "text",
        "article_count",
        "forward_drawdown",
        args.label_column,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise SystemExit(f"Dataset is missing required columns: {sorted(missing)}")

    frame["information_date"] = pd.to_datetime(frame["information_date"], errors="raise")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame[args.label_column] = pd.to_numeric(frame[args.label_column], errors="raise").astype(int)
    frame["text"] = frame["text"].fillna("").astype(str)
    frame["alignment_days"] = (frame["date"] - frame["information_date"]).dt.days
    frame["text_chars"] = frame["text"].str.len()

    positive = _sample_class(frame, args.label_column, 1, args.samples_per_class, args.seed)
    negative = _sample_class(frame, args.label_column, 0, args.samples_per_class, args.seed + 1)
    sampled = pd.concat([positive, negative], ignore_index=True)
    sampled = sampled.sort_values([args.label_column, "date", "ticker"], ascending=[False, True, True])

    preferred = [
        "ticker",
        "information_date",
        "date",
        "alignment_days",
        "article_count",
        "text_chars",
        "forward_drawdown",
        "trailing_vol",
        "normalized_score",
        args.label_column,
        "text",
    ]
    columns = [column for column in preferred if column in sampled.columns]
    sampled = sampled[columns].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "sampled_asset_days.csv"
    md_path = args.output_dir / "sampled_asset_days.md"
    summary_path = args.output_dir / "alignment_summary.csv"

    sampled.to_csv(csv_path, index=False)

    alignment_summary = (
        frame.groupby(args.label_column)
        .agg(
            observations=("ticker", "size"),
            mean_alignment_days=("alignment_days", "mean"),
            max_alignment_days=("alignment_days", "max"),
            mean_articles=("article_count", "mean"),
            median_text_chars=("text_chars", "median"),
            mean_forward_drawdown=("forward_drawdown", "mean"),
        )
        .reset_index()
    )
    alignment_summary.to_csv(summary_path, index=False)

    lines: list[str] = [
        "# Information-set audit",
        "",
        f"Dataset: `{args.dataset}`",
        "",
        "Use this report to judge each sampled row on four questions:",
        "",
        "1. Is the text genuinely about the named asset?",
        "2. Could the text plausibly contain information relevant to the later downside outcome?",
        "3. Is the information date aligned conservatively to the trading date?",
        "4. Does the realized drawdown look related to the information set, or more like a later macro/idiosyncratic shock?",
        "",
        "## Alignment summary",
        "",
        _markdown_table(alignment_summary),
        "",
    ]

    for _, row in sampled.iterrows():
        label = int(row[args.label_column])
        lines.extend(
            [
                f"## {'Positive' if label else 'Negative'} — {row['ticker']} — {pd.Timestamp(row['date']).date()}",
                "",
                f"- information_date: {pd.Timestamp(row['information_date']).date()}",
                f"- trading_date: {pd.Timestamp(row['date']).date()}",
                f"- alignment_days: {int(row['alignment_days'])}",
                f"- article_count: {int(row['article_count'])}",
                f"- forward_drawdown: {_fmt_float(row['forward_drawdown'])}",
            ]
        )
        if "trailing_vol" in sampled.columns:
            lines.append(f"- trailing_vol: {_fmt_float(row.get('trailing_vol'))}")
        if "normalized_score" in sampled.columns:
            lines.append(f"- normalized_score: {_fmt_float(row.get('normalized_score'))}")
        lines.extend(["", "### Text", ""])
        text = str(row["text"])
        if len(text) > args.max_text_chars:
            text = text[: args.max_text_chars] + "\n\n[TRUNCATED FOR AUDIT]"
        lines.extend([text, ""])

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(alignment_summary.to_string(index=False))
    print(f"\nSampled {len(positive)} positive and {len(negative)} negative asset-days.")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")
    print(f"Alignment summary: {summary_path}")


if __name__ == "__main__":
    main()
