"""Audit consistency of earnings-event return labels against OHLC anchors.

The script streams a sample from an earnings-event Hugging Face dataset and
reconstructs simple post-event returns from the provided D-1 / D+1 / D+30 price
anchors. It does not assume that the dataset creator's label definitions are
correct; instead it quantifies agreement and surfaces discrepant episodes for
manual inspection.

Example:
    uv run python scripts/audit_earnings_label_consistency.py \
        --repo-id RudrakshNanavaty/earnings-call-data \
        --split train \
        --max-rows 20000 \
        --samples 20 \
        --output-dir artifacts/earnings_label_audit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-rows", type=int, default=20_000)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/earnings_label_audit"),
    )
    return parser.parse_args()


def _safe_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _rel_error(a: pd.Series, b: pd.Series) -> pd.Series:
    denom = b.abs().clip(lower=1e-8)
    return (a - b).abs() / denom


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

    rows: list[dict[str, object]] = []
    wanted = [
        "episode_id",
        "symbol",
        "company_name",
        "date",
        "earnings_date",
        "d_minus_1_close",
        "d_plus_1_open",
        "d_plus_1_close",
        "d_plus_30_close",
        "move_1d",
        "move_30d",
        "gap_open_d1",
        "move_1d_direction",
    ]
    for index, row in enumerate(dataset):
        rows.append({key: row.get(key) for key in wanted})
        if index + 1 >= args.max_rows:
            break

    if not rows:
        raise SystemExit("No rows loaded")

    frame = pd.DataFrame.from_records(rows)
    numeric = [
        "d_minus_1_close",
        "d_plus_1_open",
        "d_plus_1_close",
        "d_plus_30_close",
        "move_1d",
        "move_30d",
        "gap_open_d1",
    ]
    for column in numeric:
        frame[column] = frame[column].map(_safe_float)

    frame["reconstructed_close_to_close_1d"] = (
        frame["d_plus_1_close"] / frame["d_minus_1_close"] - 1.0
    )
    frame["reconstructed_gap_open_d1"] = (
        frame["d_plus_1_open"] / frame["d_minus_1_close"] - 1.0
    )
    frame["reconstructed_close_to_close_30d"] = (
        frame["d_plus_30_close"] / frame["d_minus_1_close"] - 1.0
    )

    valid_1d = frame.dropna(subset=["move_1d", "reconstructed_close_to_close_1d"]).copy()
    valid_30d = frame.dropna(subset=["move_30d", "reconstructed_close_to_close_30d"]).copy()
    valid_gap = frame.dropna(subset=["gap_open_d1", "reconstructed_gap_open_d1"]).copy()

    def agreement_summary(
        data: pd.DataFrame,
        supplied: str,
        reconstructed: str,
    ) -> dict[str, float | int]:
        if data.empty:
            return {"count": 0}
        diff = data[supplied] - data[reconstructed]
        return {
            "count": int(len(data)),
            "mean_abs_error": float(diff.abs().mean()),
            "median_abs_error": float(diff.abs().median()),
            "p95_abs_error": float(diff.abs().quantile(0.95)),
            "correlation": float(data[[supplied, reconstructed]].corr().iloc[0, 1]),
            "sign_agreement": float(
                np.mean(np.sign(data[supplied].to_numpy()) == np.sign(data[reconstructed].to_numpy()))
            ),
        }

    summary = {
        "repo_id": args.repo_id,
        "split": args.split,
        "rows_loaded": int(len(frame)),
        "move_1d_vs_close_to_close": agreement_summary(
            valid_1d, "move_1d", "reconstructed_close_to_close_1d"
        ),
        "move_30d_vs_close_to_close": agreement_summary(
            valid_30d, "move_30d", "reconstructed_close_to_close_30d"
        ),
        "gap_open_d1": agreement_summary(
            valid_gap, "gap_open_d1", "reconstructed_gap_open_d1"
        ),
    }

    if not valid_1d.empty:
        valid_1d["move_1d_abs_error"] = (
            valid_1d["move_1d"] - valid_1d["reconstructed_close_to_close_1d"]
        ).abs()
    if not valid_30d.empty:
        valid_30d["move_30d_abs_error"] = (
            valid_30d["move_30d"] - valid_30d["reconstructed_close_to_close_30d"]
        ).abs()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )

    columns = [
        "episode_id",
        "symbol",
        "company_name",
        "date",
        "earnings_date",
        "d_minus_1_close",
        "d_plus_1_open",
        "d_plus_1_close",
        "d_plus_30_close",
        "move_1d",
        "reconstructed_close_to_close_1d",
        "gap_open_d1",
        "reconstructed_gap_open_d1",
        "move_30d",
        "reconstructed_close_to_close_30d",
        "move_1d_direction",
    ]
    frame[[column for column in columns if column in frame.columns]].to_csv(
        args.output_dir / "reconstructed_returns.csv", index=False
    )

    discrepancy_frames: list[pd.DataFrame] = []
    if not valid_1d.empty:
        discrepancy_frames.append(valid_1d.nlargest(args.samples, "move_1d_abs_error"))
    if not valid_30d.empty:
        discrepancy_frames.append(valid_30d.nlargest(args.samples, "move_30d_abs_error"))
    if discrepancy_frames:
        discrepancies = pd.concat(discrepancy_frames, ignore_index=True).drop_duplicates(
            subset=["episode_id"], keep="first"
        )
        discrepancies.to_csv(args.output_dir / "largest_discrepancies.csv", index=False)

    print(json.dumps(summary, indent=2, allow_nan=True))
    print(f"\nArtifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
