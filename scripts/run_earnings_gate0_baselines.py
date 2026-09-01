"""Run Gate 0 structured and transcript baselines for earnings downside events.

The target is a large negative next-session close-to-close earnings reaction.
The downside threshold is estimated from the training period only and frozen for
the chronological test period.

Example:
    uv run python scripts/run_earnings_gate0_baselines.py \
        --repo-id RudrakshNanavaty/earnings-call-data \
        --split train \
        --max-rows 20000 \
        --train-end 2021-12-31 \
        --test-start 2022-01-01 \
        --test-end 2023-12-31 \
        --target-positive-rate 0.20 \
        --output artifacts/earnings_gate0_2022_2023.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-rows", type=int, default=20_000)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--test-end", required=True)
    parser.add_argument("--target-positive-rate", type=float, default=0.20)
    parser.add_argument("--max-features", type=int, default=30_000)
    parser.add_argument("--max-text-chars", type=int, default=30_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/earnings_gate0_baselines.json"),
    )
    return parser.parse_args()


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    result = {
        "brier": float(brier_score_loss(y, p)),
        "nll": float(log_loss(y, np.column_stack([1.0 - p, p]), labels=[0, 1])),
    }
    result["auroc"] = float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan")
    return result


def _trim_transcript(text: object, max_chars: int) -> str:
    value = "" if text is None else str(text)
    if len(value) <= max_chars:
        return value
    head = max_chars // 2
    tail = max_chars - head
    return value[:head] + "\n\n[...TRUNCATED MIDDLE...]\n\n" + value[-tail:]


def _log1p_nonnegative(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.log1p(np.clip(array, a_min=0.0, a_max=None))


def main() -> None:
    args = parse_args()
    if args.max_rows <= 0:
        raise ValueError("max-rows must be positive")
    if not 0.0 < args.target_positive_rate < 0.5:
        raise ValueError("target-positive-rate must be between 0 and 0.5")
    if args.max_features <= 0 or args.max_text_chars <= 0:
        raise ValueError("max-features and max-text-chars must be positive")

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install project dependencies with `uv sync`") from exc

    wanted = [
        "episode_id",
        "symbol",
        "date",
        "earnings_transcript",
        "move_1d",
        "price_momentum_30d",
        "price_momentum_90d",
        "pct_from_52w_high_pt",
        "avg_volume_20d",
    ]

    print(f"Streaming up to {args.max_rows} rows from {args.repo_id} [{args.split}]...", flush=True)
    dataset = load_dataset(args.repo_id, split=args.split, streaming=True)
    rows: list[dict[str, object]] = []
    for index, row in enumerate(dataset):
        rows.append({key: row.get(key) for key in wanted})
        if (index + 1) % 2500 == 0:
            print(f"  loaded {index + 1}/{args.max_rows} rows", flush=True)
        if index + 1 >= args.max_rows:
            break

    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        raise SystemExit("No rows loaded")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    numeric_columns = [
        "move_1d",
        "price_momentum_30d",
        "price_momentum_90d",
        "pct_from_52w_high_pt",
        "avg_volume_20d",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["earnings_transcript"] = frame["earnings_transcript"].map(
        lambda value: _trim_transcript(value, args.max_text_chars)
    )
    frame = frame.dropna(subset=["date", "move_1d"]).copy()
    frame = frame[frame["earnings_transcript"].str.len() > 0].copy()

    train_end = pd.Timestamp(args.train_end)
    test_start = pd.Timestamp(args.test_start)
    test_end = pd.Timestamp(args.test_end)
    if not (train_end < test_start <= test_end):
        raise ValueError("require train-end < test-start <= test-end")

    train = frame[frame["date"] <= train_end].copy()
    test = frame[(frame["date"] >= test_start) & (frame["date"] <= test_end)].copy()
    if train.empty or test.empty:
        raise SystemExit("Train or test split is empty")

    threshold = float(train["move_1d"].quantile(args.target_positive_rate))
    train["label"] = (train["move_1d"] <= threshold).astype(int)
    test["label"] = (test["move_1d"] <= threshold).astype(int)

    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()
    constant_p = np.full(len(test), float(np.mean(y_train)))

    structured_features = [
        "price_momentum_30d",
        "price_momentum_90d",
        "pct_from_52w_high_pt",
        "avg_volume_20d",
    ]

    structured_preprocessor = ColumnTransformer(
        transformers=[
            (
                "regular",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                structured_features[:3],
            ),
            (
                "volume",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("log", FunctionTransformer(_log1p_nonnegative, validate=False)),
                        ("scale", StandardScaler()),
                    ]
                ),
                ["avg_volume_20d"],
            ),
        ]
    )
    structured_model = Pipeline(
        [
            ("preprocess", structured_preprocessor),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )
    structured_model.fit(train[structured_features], y_train)
    structured_p = structured_model.predict_proba(test[structured_features])[:, 1]

    print("Fitting TF-IDF transcript baseline...", flush=True)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.98,
        max_features=args.max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    x_text_train = vectorizer.fit_transform(train["earnings_transcript"].tolist())
    x_text_test = vectorizer.transform(test["earnings_transcript"].tolist())

    text_model = LogisticRegression(max_iter=2000, solver="liblinear")
    text_model.fit(x_text_train, y_train)
    text_p = text_model.predict_proba(x_text_test)[:, 1]

    structured_transformer = structured_preprocessor.fit(train[structured_features])
    x_struct_train = structured_transformer.transform(train[structured_features])
    x_struct_test = structured_transformer.transform(test[structured_features])
    if not sparse.issparse(x_struct_train):
        x_struct_train = sparse.csr_matrix(x_struct_train)
        x_struct_test = sparse.csr_matrix(x_struct_test)

    x_combined_train = sparse.hstack([x_text_train, x_struct_train], format="csr")
    x_combined_test = sparse.hstack([x_text_test, x_struct_test], format="csr")
    combined_model = LogisticRegression(max_iter=2000, solver="liblinear")
    combined_model.fit(x_combined_train, y_train)
    combined_p = combined_model.predict_proba(x_combined_test)[:, 1]

    results = {
        "repo_id": args.repo_id,
        "split": args.split,
        "rows_loaded": int(len(rows)),
        "rows_after_filtering": int(len(frame)),
        "train_end": args.train_end,
        "test_start": args.test_start,
        "test_end": args.test_end,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "target": "move_1d downside tail",
        "target_positive_rate_requested": args.target_positive_rate,
        "training_threshold_move_1d": threshold,
        "train_positive_rate": float(np.mean(y_train)),
        "test_positive_rate": float(np.mean(y_test)),
        "max_text_chars": args.max_text_chars,
        "tfidf_features": int(len(vectorizer.get_feature_names_out())),
        "constant": _metrics(y_test, constant_p),
        "structured_only": _metrics(y_test, structured_p),
        "tfidf_text_only": _metrics(y_test, text_p),
        "tfidf_text_plus_structured": _metrics(y_test, combined_p),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, allow_nan=True))
    print(f"\nArtifact written to {args.output}")


if __name__ == "__main__":
    main()
