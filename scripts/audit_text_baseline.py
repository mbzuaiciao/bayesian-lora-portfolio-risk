"""Audit whether simple text features add signal beyond trailing volatility.

This gate runs before LoRA. It compares:
- constant prevalence,
- trailing volatility only,
- TF-IDF text only,
- TF-IDF text plus trailing volatility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--label-column", default="normalized_label")
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--test-end", required=True)
    parser.add_argument("--max-features", type=int, default=20_000)
    parser.add_argument("--min-df", type=int, default=3)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("artifacts/text_baseline.json"))
    return parser.parse_args()


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "brier": float(brier_score_loss(y, p)),
        "nll": float(log_loss(y, np.column_stack([1.0 - p, p]), labels=[0, 1])),
        "auroc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan"),
    }


def fit_predict(x_train, y_train: np.ndarray, x_test) -> np.ndarray:
    model = LogisticRegression(max_iter=2000)
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


def main() -> None:
    args = parse_args()
    if not args.dataset.is_file():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    frame = pd.read_csv(args.dataset)
    required = {"date", "text", "trailing_vol", args.label_column}
    missing = required.difference(frame.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["text"] = frame["text"].fillna("").astype(str)
    frame["trailing_vol"] = pd.to_numeric(frame["trailing_vol"], errors="coerce")
    frame[args.label_column] = pd.to_numeric(frame[args.label_column], errors="raise").astype(int)
    frame = frame.dropna(subset=["trailing_vol"]).copy()

    train_end = pd.Timestamp(args.train_end)
    test_start = pd.Timestamp(args.test_start)
    test_end = pd.Timestamp(args.test_end)
    if not (train_end < test_start <= test_end):
        raise ValueError("require train-end < test-start <= test-end")

    train = frame[frame["date"] <= train_end].copy()
    test = frame[(frame["date"] >= test_start) & (frame["date"] <= test_end)].copy()
    if train.empty or test.empty:
        raise SystemExit("Train or test split is empty.")

    y_train = train[args.label_column].to_numpy()
    y_test = test[args.label_column].to_numpy()

    results: dict[str, object] = {
        "label_column": args.label_column,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_positive_rate": float(y_train.mean()),
        "test_positive_rate": float(y_test.mean()),
    }

    constant_p = np.full(len(test), float(y_train.mean()))
    results["constant"] = metrics(y_test, constant_p)

    scaler = StandardScaler()
    vol_train = scaler.fit_transform(train[["trailing_vol"]])
    vol_test = scaler.transform(test[["trailing_vol"]])
    results["volatility_only"] = metrics(y_test, fit_predict(vol_train, y_train, vol_test))

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, args.ngram_max),
        min_df=args.min_df,
        max_features=args.max_features,
        sublinear_tf=True,
    )
    text_train = vectorizer.fit_transform(train["text"])
    text_test = vectorizer.transform(test["text"])
    results["vocabulary_size"] = int(len(vectorizer.vocabulary_))
    results["text_only"] = metrics(y_test, fit_predict(text_train, y_train, text_test))

    combined_train = hstack([text_train, csr_matrix(vol_train)], format="csr")
    combined_test = hstack([text_test, csr_matrix(vol_test)], format="csr")
    results["text_plus_volatility"] = metrics(
        y_test,
        fit_predict(combined_train, y_train, combined_test),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=True) + "\n")
    print(json.dumps(results, indent=2, allow_nan=True))
    print(f"\nArtifact written to {args.output}")


if __name__ == "__main__":
    main()
