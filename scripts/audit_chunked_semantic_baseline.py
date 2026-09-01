"""Audit chunked FinBERT document embeddings against volatility.

This script tests whether earlier semantic baselines underperformed because a
single FinBERT window discarded most of the relevance-filtered document. Each
asset-day document is tokenized into overlapping chunks, encoded in batches
across documents, and mean pooled back to one embedding per document before
fitting logistic-regression classifiers.

Example:
    uv run python scripts/audit_chunked_semantic_baseline.py \
        --dataset artifacts/fnspid_relevance_filtered/model_ready_volnorm.csv \
        --label-column normalized_label \
        --train-end 2021-12-31 \
        --test-start 2022-01-01 \
        --test-end 2022-12-31 \
        --max-chunks 4 \
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
    parser.add_argument("--max-chunks", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=25)
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
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if not token_ids:
        if unk_id is None:
            raise ValueError("tokenizer returned no tokens and has no unk_token_id")
        token_ids = [int(unk_id)]

    cls_id = getattr(tokenizer, "cls_token_id", None)
    sep_id = getattr(tokenizer, "sep_token_id", None)
    if cls_id is None or sep_id is None:
        raise ValueError("BERT-style tokenizer must expose cls_token_id and sep_token_id")

    content_width = chunk_tokens - 2
    if content_width <= 0:
        raise ValueError("chunk-tokens is too small for BERT special tokens")
    step = content_width - stride
    if step <= 0:
        raise ValueError("chunk-stride must be smaller than effective chunk width")

    chunks: list[dict[str, list[int]]] = []
    start = 0
    while start < len(token_ids) and len(chunks) < max_chunks:
        piece = token_ids[start : start + content_width]
        input_ids = [int(cls_id), *[int(token_id) for token_id in piece], int(sep_id)]
        chunks.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
        if start + content_width >= len(token_ids):
            break
        start += step
    return chunks


def _encode_documents(
    texts: list[str],
    *,
    split_name: str,
    tokenizer: object,
    model: torch.nn.Module,
    device: torch.device,
    chunk_tokens: int,
    stride: int,
    max_chunks: int,
    batch_size: int,
    progress_every: int,
) -> tuple[np.ndarray, list[int]]:
    """Encode chunks in batches across documents, then mean-pool per document."""
    flat_chunks: list[dict[str, list[int]]] = []
    chunk_doc_ids: list[int] = []
    chunk_counts: list[int] = []

    print(f"Preparing {split_name} chunks for {len(texts)} documents...", flush=True)
    for doc_id, text in enumerate(texts):
        chunks = _chunk_text(
            text,
            tokenizer,
            chunk_tokens=chunk_tokens,
            stride=stride,
            max_chunks=max_chunks,
        )
        chunk_counts.append(len(chunks))
        flat_chunks.extend(chunks)
        chunk_doc_ids.extend([doc_id] * len(chunks))
        if progress_every > 0 and (doc_id + 1) % 500 == 0:
            print(
                f"  prepared {doc_id + 1}/{len(texts)} {split_name} documents "
                f"({len(flat_chunks)} chunks)",
                flush=True,
            )

    if not flat_chunks:
        raise ValueError(f"No chunks produced for {split_name} split")

    hidden_size = int(getattr(model.config, "hidden_size"))
    sums = np.zeros((len(texts), hidden_size), dtype=np.float32)
    counts = np.zeros(len(texts), dtype=np.int32)
    total_batches = (len(flat_chunks) + batch_size - 1) // batch_size
    print(
        f"Encoding {split_name}: {len(flat_chunks)} chunks in {total_batches} batches "
        f"on {device}...",
        flush=True,
    )

    model.eval()
    for batch_index, offset in enumerate(range(0, len(flat_chunks), batch_size), start=1):
        batch = flat_chunks[offset : offset + batch_size]
        doc_ids = chunk_doc_ids[offset : offset + batch_size]
        padded = tokenizer.pad(batch, padding=True, return_tensors="pt")
        padded = {key: value.to(device) for key, value in padded.items()}
        with torch.inference_mode():
            outputs = model(**padded)
            pooled = _mean_pool(outputs.last_hidden_state, padded["attention_mask"])
        pooled_np = pooled.detach().cpu().numpy().astype(np.float32, copy=False)

        for row, doc_id in zip(pooled_np, doc_ids, strict=True):
            sums[doc_id] += row
            counts[doc_id] += 1

        if progress_every > 0 and (
            batch_index == 1
            or batch_index % progress_every == 0
            or batch_index == total_batches
        ):
            percent = 100.0 * batch_index / total_batches
            print(
                f"  {split_name}: batch {batch_index}/{total_batches} ({percent:.1f}%)",
                flush=True,
            )

    if np.any(counts == 0):
        raise RuntimeError(f"Some {split_name} documents received no encoded chunks")
    embeddings = sums / counts[:, None]
    return embeddings, chunk_counts


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
        split_name="train",
        tokenizer=tokenizer,
        model=model,
        device=device,
        chunk_tokens=args.chunk_tokens,
        stride=args.chunk_stride,
        max_chunks=args.max_chunks,
        batch_size=args.batch_size,
        progress_every=args.progress_every,
    )
    test_x, test_chunks = _encode_documents(
        test["text"].tolist(),
        split_name="test",
        tokenizer=tokenizer,
        model=model,
        device=device,
        chunk_tokens=args.chunk_tokens,
        stride=args.chunk_stride,
        max_chunks=args.max_chunks,
        batch_size=args.batch_size,
        progress_every=args.progress_every,
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
        "fraction_train_hitting_max_chunks": float(
            np.mean(np.asarray(train_chunks) == args.max_chunks)
        ),
        "mean_test_chunks": float(np.mean(test_chunks)),
        "median_test_chunks": float(np.median(test_chunks)),
        "fraction_test_hitting_max_chunks": float(
            np.mean(np.asarray(test_chunks) == args.max_chunks)
        ),
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
