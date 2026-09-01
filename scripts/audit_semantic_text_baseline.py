"""Audit whether pretrained financial semantics add signal beyond volatility.

This is the final cheap gate before LoRA. It freezes a pretrained FinBERT encoder,
extracts one embedding per asset-day text observation, and fits logistic-regression
heads on chronological train/test splits.

Example:
    uv run python scripts/audit_semantic_text_baseline.py \
        --dataset artifacts/fnspid_slice_full/model_ready_volnorm.csv \
        --label-column normalized_label \
        --train-end 2021-12-31 \
        --test-start 2022-01-01 \
        --test-end 2022-12-31 \
        --output artifacts/semantic_text_baseline_volnorm_2022.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--label-column", default="normalized_label")
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--test-end", required=True)
    parser.add_argument("--model-name", default="ProsusAI/finbert")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/semantic_text_baseline.json")
    )
    return parser.parse_args()


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    result = {
        "brier": float(brier_score_loss(y, p)),
        "nll": float(log_loss(y, np.column_stack([1.0 - p, p]), labels=[0, 1])),
    }
    result["auroc"] = float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan")
    return result


def _resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but is not available")
        return torch.device("mps")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


def _embed_texts(
    texts: list[str],
    *,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            pooled = _mean_pool(output.last_hidden_state, encoded["attention_mask"])
            pieces.append(pooled.float().cpu().numpy())
    if not pieces:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(pieces, axis=0)


def main() -> None:
    args = parse_args()
    if not args.dataset.is_file():
        raise SystemExit(f"Dataset not found: {args.dataset}")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.max_length <= 0:
        raise ValueError("max-length must be positive")

    frame = pd.read_csv(args.dataset)
    required = {"date", "text", "trailing_vol", args.label_column}
    missing = required.difference(frame.columns)
    if missing:
        raise SystemExit(f"Dataset missing required columns: {sorted(missing)}")

    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["text"] = frame["text"].fillna("").astype(str)
    frame["trailing_vol"] = pd.to_numeric(frame["trailing_vol"], errors="raise")
    frame[args.label_column] = pd.to_numeric(frame[args.label_column], errors="raise").astype(int)
    frame = frame[(frame["text"].str.len() > 0) & frame["trailing_vol"].notna()].copy()

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

    device = _resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)

    train_embeddings = _embed_texts(
        train["text"].tolist(),
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    test_embeddings = _embed_texts(
        test["text"].tolist(),
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    results: dict[str, object] = {
        "label_column": args.label_column,
        "model_name": args.model_name,
        "device": str(device),
        "embedding_dim": int(train_embeddings.shape[1]),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_positive_rate": float(np.mean(y_train)),
        "test_positive_rate": float(np.mean(y_test)),
    }

    constant_p = np.full(len(test), float(np.mean(y_train)))
    results["constant"] = _metrics(y_test, constant_p)

    vol_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    vol_model.fit(train[["trailing_vol"]], y_train)
    vol_p = vol_model.predict_proba(test[["trailing_vol"]])[:, 1]
    results["volatility_only"] = _metrics(y_test, vol_p)

    semantic_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    semantic_model.fit(train_embeddings, y_train)
    semantic_p = semantic_model.predict_proba(test_embeddings)[:, 1]
    results["semantic_text_only"] = _metrics(y_test, semantic_p)

    train_combined = np.column_stack(
        [train_embeddings, train["trailing_vol"].to_numpy(dtype=float)]
    )
    test_combined = np.column_stack(
        [test_embeddings, test["trailing_vol"].to_numpy(dtype=float)]
    )
    combined_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    combined_model.fit(train_combined, y_train)
    combined_p = combined_model.predict_proba(test_combined)[:, 1]
    results["semantic_text_plus_volatility"] = _metrics(y_test, combined_p)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=True) + "\n")
    print(json.dumps(results, indent=2, allow_nan=True))
    print(f"\nArtifact written to {args.output}")


if __name__ == "__main__":
    main()
