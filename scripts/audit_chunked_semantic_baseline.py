"""Audit chunked FinBERT document embeddings against volatility.

This script tests whether earlier semantic baselines underperformed because a
single FinBERT window discarded most of the relevance-filtered document. Each
asset-day document is tokenized into overlapping chunks, encoded independently,
and mean pooled across chunks before fitting logistic-regression classifiers.

Example:
    uv run python scripts/audit_chunked_semantic_baseline.py \
        --dataset artifacts/fnspid_relevance_filtered/model_ready_volnorm.csv \
        --label-column normalized_label \
        --train-end 2021-12-31 \
        --test-start 2022-01-01 \
        --test-end 2022-12-31 \
        --output artifacts/chunked_semantic_baseline_2022.json
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--label-column", default="normalized_label")
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--test-end", required=True)
    parser.add_argument("--model-name", default="ProsusAI/finbert")
    parser.add_argument("--chunk-tokens", type=int, default=384)
    parser.add_argument("--chunk-stride", type=int, default=64)
    parser.add_argument("--max-chunks", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/chunked_semantic_baseline.json"),
    )
    return parser.parse_args()


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    result = {
        "brier": float(brier_score_loss(y, p)),
        "nll": float(log_loss(y, np.column_stack([1.0 - p, p]), labels=[0, 1])),
    }
    result["auroc"] = float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan")
    return result


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def _chunk_text(
    text: str,
    tokenizer: object,
    *,
    chunk_tokens: int,
    stride: int,
    max_chunks: int,
) -> list[dict[str, list[int]]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
    )
    token_ids: list[int] = encoded["input_ids"]
    if not token_ids:
        token_ids = [tokenizer.unk_token_id]

    content_width = chunk_tokens - tokenizer.num_special_tokens_to_add(pair=False)
    if content_width <= 0:
        raise ValueError("chunk-tokens is too small for tokenizer special tokens")
    step = content_width - stride
    if step <= 0:
        raise ValueError("chunk-stride must be smaller than effective chunk width")

    chunks: list[dict[str, list[int]]] = []
    start = 0
    while start < len(token_ids) and len(chunks) < max_chunks:
        piece = token_ids[start : start + content_width]
        with_special = tokenizer.prepare_for_model(
            piece,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=True,
        )
        chunks.append(
            {
                "input_ids": with_special["input_ids"],
                "attention_mask": with_special["attention_mask"],
            }
        )
        if start + content_width >= len(token_ids):
            break
        start += step
    return chunks


def _encode_documents(
    texts: list[str],
    *,
    tokenizer: object,
    model: torch.nn.Module,
    device: torch.device,
    chunk_tokens: int,
    stride: int,
    max_chunks: int,
    batch_size: int,
) -> tuple[np.ndarray, list[int]]:
    all_embeddings: list[np.ndarray] = []
    chunk_counts: list[int] = []

    model.eval()
    for text in texts:
        chunks = _chunk_text(
            text,
            tokenizer,
            chunk_tokens=chunk_tokens,
            stride=stride,
            max_chunks=max_chunks,
        )
        chunk_counts.append(len(chunks))
        chunk_embeddings: list[torch.Tensor] = []

        for offset in range(0, len(chunks), batch_size):
            batch = chunks[offset : offset + batch_size]
            padded = tokenizer.pad(batch, padding=True, return_tensors="pt")
            padded = {key: value.to(device) for key, value in padded.items()}
            with torch.inference_mode():
                outputs = model(**padded)
                pooled = _mean_pool(outputs.last_hidden_state, padded["attention_mask"])
            chunk_embeddings.append(pooled.detach().cpu())

        document_embedding = torch.cat(chunk_embeddings, dim=0).mean(dim=0)
        all_embeddings.append(document_embedding.numpy())

    return np.stack(all_embeddings), chunk_counts


def main() -> None:
    args = parse_args()
    if not args.dataset.is_file():
        raise SystemExit(f"Dataset not found: {args.dataset}")
    if args.chunk_tokens <= 8:
        raise ValueError("chunk-tokens must be greater than 8")
    if args.chunk_stride < 0:
        raise ValueError("chunk-stride must be non-negative")
    if args.max_chunks <= 0 or args.batch_size <= 0:
        raise ValueError("max-chunks and batch-size must be positive")

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install project dependencies with `uv sync`") from exc

    frame = pd.read_csv(args.dataset)
    required = {"date", "text", "trailing_vol", args.label_column}
    missing = required.difference(frame.columns)
    if missing:
        raise SystemExit(f"Dataset is missing required columns: {sorted(missing)}")

    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame[args.label_column] = pd.to_numeric(frame[args.label_column], errors="raise").astype(int)
    frame["trailing_vol"] = pd.to_numeric(frame["trailing_vol"], errors="coerce")
    frame["text"] = frame["text"].fillna("").astype(str)
    frame = frame.dropna(subset=["trailing_vol"]).copy()

    train_end = pd.Timestamp(args.train_end)
    test_start = pd.Timestamp(args.test_start)
    test_end = pd.Timestamp(args.test_end)
    if not (train_end < test_start <= test_end):
        raise ValueError("require train-end < test-start <= test-end")

    train = frame[frame["date"] <= train_end].copy()
    test = frame[(frame["date"] >= test_start) & (frame["date"] <= test_end)].copy()
    if train.empty or test.empty:
        raise SystemExit("Train or test split is empty")

    device = _device()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)

    train_x, train_chunks = _encode_documents(
        train["text"].tolist(),
        tokenizer=tokenizer,
        model=model,
        device=device,
        chunk_tokens=args.chunk_tokens,
        stride=args.chunk_stride,
        max_chunks=args.max_chunks,
        batch_size=args.batch_size,
    )
    test_x, test_chunks = _encode_documents(
        test["text"].tolist(),
        tokenizer=tokenizer,
        model=model,
        device=device,
        chunk_tokens=args.chunk_tokens,
        stride=args.chunk_stride,
        max_chunks=args.max_chunks,
        batch_size=args.batch_size,
    )

    y_train = train[args.label_column].to_numpy()
    y_test = test[args.label_column].to_numpy()
    vol_train = train[["trailing_vol"]].to_numpy()
    vol_test = test[["trailing_vol"]].to_numpy()

    constant_p = np.full(len(test), float(np.mean(y_train)))

    vol_model = Pipeline(
        [("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]
    )
    vol_model.fit(vol_train, y_train)
    vol_p = vol_model.predict_proba(vol_test)[:, 1]

    text_model = Pipeline(
        [("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000))]
    )
    text_model.fit(train_x, y_train)
    text_p = text_model.predict_proba(test_x)[:, 1]

    combined_train = np.column_stack([train_x, vol_train])
    combined_test = np.column_stack([test_x, vol_test])
    combined_model = Pipeline(
        [("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000))]
    )
    combined_model.fit(combined_train, y_train)
    combined_p = combined_model.predict_proba(combined_test)[:, 1]

    results = {
        "label_column": args.label_column,
        "model_name": args.model_name,
        "device": str(device),
        "embedding_dim": int(train_x.shape[1]),
        "chunk_tokens": args.chunk_tokens,
        "chunk_stride": args.chunk_stride,
        "max_chunks": args.max_chunks,
        "mean_train_chunks": float(np.mean(train_chunks)),
        "median_train_chunks": float(np.median(train_chunks)),
        "fraction_train_hitting_max_chunks": float(np.mean(np.asarray(train_chunks) == args.max_chunks)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_positive_rate": float(np.mean(y_train)),
        "test_positive_rate": float(np.mean(y_test)),
        "constant": _metrics(y_test, constant_p),
        "volatility_only": _metrics(y_test, vol_p),
        "chunked_semantic_text_only": _metrics(y_test, text_p),
        "chunked_semantic_text_plus_volatility": _metrics(y_test, combined_p),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=True) + "\n")
    print(json.dumps(results, indent=2, allow_nan=True))
    print(f"\nArtifact written to {args.output}")


if __name__ == "__main__":
    main()
