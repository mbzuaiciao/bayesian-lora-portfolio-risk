"""Run a chunked frozen-FinBERT baseline for earnings downside events.

This Gate 0.5 experiment keeps the same chronological split, downside-tail target,
and structured features as ``run_earnings_gate0_baselines.py`` while replacing
TF-IDF with frozen chunked FinBERT document embeddings. Chunks from many
transcripts are globally batched for efficient MPS/CUDA execution.

Example:
    uv run python scripts/run_earnings_chunked_semantic_baseline.py \
        --repo-id RudrakshNanavaty/earnings-call-data \
        --split train \
        --max-rows 20000 \
        --train-end 2021-12-31 \
        --test-start 2022-01-01 \
        --test-end 2023-12-31 \
        --target-positive-rate 0.20 \
        --chunk-tokens 384 \
        --chunk-stride 64 \
        --max-chunks 4 \
        --batch-size 16 \
        --output artifacts/earnings_chunked_semantic_2022_2023.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-rows", type=int, default=20_000)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--test-end", required=True)
    parser.add_argument("--target-positive-rate", type=float, default=0.20)
    parser.add_argument("--model-name", default="ProsusAI/finbert")
    parser.add_argument("--chunk-tokens", type=int, default=384)
    parser.add_argument("--chunk-stride", type=int, default=64)
    parser.add_argument("--max-chunks", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/earnings_chunked_semantic_baseline.json"),
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
        verbose=False,
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
    step = content_width - stride
    if content_width <= 0:
        raise ValueError("chunk-tokens is too small for BERT special tokens")
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
    flat_chunks: list[dict[str, list[int]]] = []
    doc_ids: list[int] = []
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
        doc_ids.extend([doc_id] * len(chunks))
        if (doc_id + 1) % 500 == 0:
            print(
                f"  prepared {doc_id + 1}/{len(texts)} {split_name} documents "
                f"({len(flat_chunks)} chunks)",
                flush=True,
            )

    hidden_size = int(getattr(model.config, "hidden_size"))
    sums = torch.zeros((len(texts), hidden_size), dtype=torch.float32)
    counts = torch.zeros(len(texts), dtype=torch.float32)
    total_batches = (len(flat_chunks) + batch_size - 1) // batch_size
    print(
        f"Encoding {split_name}: {len(flat_chunks)} chunks in {total_batches} batches on {device}...",
        flush=True,
    )

    model.eval()
    for batch_index, offset in enumerate(range(0, len(flat_chunks), batch_size), start=1):
        batch = flat_chunks[offset : offset + batch_size]
        batch_doc_ids = doc_ids[offset : offset + batch_size]
        padded = tokenizer.pad(batch, padding=True, return_tensors="pt")
        padded = {key: value.to(device) for key, value in padded.items()}
        with torch.inference_mode():
            outputs = model(**padded)
            pooled = _mean_pool(outputs.last_hidden_state, padded["attention_mask"]).detach().cpu()

        for row_index, doc_id in enumerate(batch_doc_ids):
            sums[doc_id] += pooled[row_index]
            counts[doc_id] += 1.0

        if batch_index == 1 or batch_index % progress_every == 0 or batch_index == total_batches:
            print(
                f"  {split_name}: batch {batch_index}/{total_batches} "
                f"({batch_index / total_batches:.1%})",
                flush=True,
            )

    embeddings = sums / counts.clamp_min(1.0).unsqueeze(1)
    return embeddings.numpy(), chunk_counts


def _structured_arrays(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    return pipe.fit_transform(train[features]), pipe.transform(test[features])


def main() -> None:
    args = parse_args()
    if args.max_rows <= 0:
        raise ValueError("max-rows must be positive")
    if not 0.0 < args.target_positive_rate < 0.5:
        raise ValueError("target-positive-rate must be between 0 and 0.5")
    if args.max_chunks <= 0 or args.batch_size <= 0 or args.progress_every <= 0:
        raise ValueError("max-chunks, batch-size, and progress-every must be positive")

    try:
        from datasets import load_dataset
        from transformers import AutoModel, AutoTokenizer
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
    frame["earnings_transcript"] = frame["earnings_transcript"].fillna("").astype(str)
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

    device = _device()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)

    train_x, train_chunks = _encode_documents(
        train["earnings_transcript"].tolist(),
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
        test["earnings_transcript"].tolist(),
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

    structured_features = [
        "price_momentum_30d",
        "price_momentum_90d",
        "pct_from_52w_high_pt",
        "avg_volume_20d",
    ]
    x_struct_train, x_struct_test = _structured_arrays(train, test, structured_features)

    constant_p = np.full(len(test), float(np.mean(y_train)))
    structured_model = LogisticRegression(max_iter=2000)
    structured_model.fit(x_struct_train, y_train)
    structured_p = structured_model.predict_proba(x_struct_test)[:, 1]

    semantic_model = Pipeline(
        [("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000))]
    )
    semantic_model.fit(train_x, y_train)
    semantic_p = semantic_model.predict_proba(test_x)[:, 1]

    combined_train = np.column_stack([train_x, x_struct_train])
    combined_test = np.column_stack([test_x, x_struct_test])
    combined_model = Pipeline(
        [("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000))]
    )
    combined_model.fit(combined_train, y_train)
    combined_p = combined_model.predict_proba(combined_test)[:, 1]

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
        "model_name": args.model_name,
        "device": str(device),
        "embedding_dim": int(train_x.shape[1]),
        "chunk_tokens": args.chunk_tokens,
        "chunk_stride": args.chunk_stride,
        "max_chunks": args.max_chunks,
        "mean_train_chunks": float(np.mean(train_chunks)),
        "median_train_chunks": float(np.median(train_chunks)),
        "fraction_train_hitting_max_chunks": float(np.mean(np.asarray(train_chunks) == args.max_chunks)),
        "mean_test_chunks": float(np.mean(test_chunks)),
        "median_test_chunks": float(np.median(test_chunks)),
        "fraction_test_hitting_max_chunks": float(np.mean(np.asarray(test_chunks) == args.max_chunks)),
        "constant": _metrics(y_test, constant_p),
        "structured_only": _metrics(y_test, structured_p),
        "chunked_semantic_text_only": _metrics(y_test, semantic_p),
        "chunked_semantic_text_plus_structured": _metrics(y_test, combined_p),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, allow_nan=True))
    print(f"\nArtifact written to {args.output}")


if __name__ == "__main__":
    main()
