"""Train a standard LoRA-adapted FinBERT baseline for earnings downside events.

This experiment keeps the same downside-tail target and chronological test split as
Gate 0 / Gate 0.5, then adapts FinBERT with PEFT LoRA on transcript chunks.
Document-level probabilities are obtained by averaging chunk probabilities.

The training period is split chronologically again so that the final fraction is a
validation set used only for model selection / diagnostics. The test period is never
used for threshold selection or training.

Example:
    uv run python scripts/run_earnings_lora_baseline.py \
        --repo-id RudrakshNanavaty/earnings-call-data \
        --split train \
        --max-rows 20000 \
        --train-end 2021-12-31 \
        --test-start 2022-01-01 \
        --test-end 2023-12-31 \
        --target-positive-rate 0.20 \
        --chunk-tokens 384 \
        --chunk-stride 64 \
        --train-max-chunks 2 \
        --eval-max-chunks 4 \
        --batch-size 8 \
        --epochs 2 \
        --learning-rate 2e-4 \
        --output artifacts/earnings_lora_2022_2023.json \
        --adapter-dir artifacts/earnings_lora_adapter
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from torch.utils.data import DataLoader, Dataset


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
    parser.add_argument("--train-max-chunks", type=int, default=2)
    parser.add_argument("--eval-max-chunks", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/earnings_lora_baseline.json"),
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("artifacts/earnings_lora_adapter"),
    )
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    result = {
        "brier": float(brier_score_loss(y, p)),
        "nll": float(log_loss(y, np.column_stack([1.0 - p, p]), labels=[0, 1])),
    }
    result["auroc"] = float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan")
    return result


def _chunk_text(
    text: str,
    tokenizer: Any,
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
    if not token_ids:
        unk_id = tokenizer.unk_token_id
        if unk_id is None:
            raise ValueError("tokenizer returned no tokens and has no unk_token_id")
        token_ids = [int(unk_id)]

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
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
        ids = [int(cls_id), *[int(token_id) for token_id in piece], int(sep_id)]
        chunks.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
        if start + content_width >= len(token_ids):
            break
        start += step
    return chunks


class ChunkDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: Any,
        *,
        chunk_tokens: int,
        stride: int,
        max_chunks: int,
    ) -> None:
        self.rows: list[dict[str, Any]] = []
        for doc_id, row in enumerate(frame.itertuples(index=False)):
            chunks = _chunk_text(
                str(row.earnings_transcript),
                tokenizer,
                chunk_tokens=chunk_tokens,
                stride=stride,
                max_chunks=max_chunks,
            )
            for chunk in chunks:
                self.rows.append(
                    {
                        **chunk,
                        "label": int(row.label),
                        "doc_id": doc_id,
                    }
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def _make_collate(tokenizer: Any):
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        features = [
            {"input_ids": row["input_ids"], "attention_mask": row["attention_mask"]}
            for row in batch
        ]
        padded = tokenizer.pad(features, padding=True, return_tensors="pt")
        padded["labels"] = torch.tensor([row["label"] for row in batch], dtype=torch.long)
        padded["doc_ids"] = torch.tensor([row["doc_id"] for row in batch], dtype=torch.long)
        return padded

    return collate


def _predict_documents(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, Any]],
    *,
    device: torch.device,
    n_documents: int,
) -> np.ndarray:
    sums: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            doc_ids = batch.pop("doc_ids").tolist()
            batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            for doc_id, prob in zip(doc_ids, probs, strict=True):
                sums[int(doc_id)] += float(prob)
                counts[int(doc_id)] += 1
    return np.asarray([sums[i] / max(counts[i], 1) for i in range(n_documents)], dtype=float)


def main() -> None:
    args = parse_args()
    if args.max_rows <= 0 or args.batch_size <= 0 or args.epochs <= 0:
        raise ValueError("max-rows, batch-size, and epochs must be positive")
    if not 0.0 < args.target_positive_rate < 0.5:
        raise ValueError("target-positive-rate must be between 0 and 0.5")
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("validation-fraction must be between 0 and 0.5")
    if args.train_max_chunks <= 0 or args.eval_max_chunks <= 0:
        raise ValueError("chunk counts must be positive")

    _seed_everything(args.seed)

    try:
        from datasets import load_dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install project dependencies with `uv sync`") from exc

    wanted = ["date", "earnings_transcript", "move_1d"]
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
    frame["move_1d"] = pd.to_numeric(frame["move_1d"], errors="coerce")
    frame["earnings_transcript"] = frame["earnings_transcript"].fillna("").astype(str)
    frame = frame.dropna(subset=["date", "move_1d"]).copy()
    frame = frame[frame["earnings_transcript"].str.len() > 0].copy()
    frame = frame.sort_values("date").reset_index(drop=True)

    train_end = pd.Timestamp(args.train_end)
    test_start = pd.Timestamp(args.test_start)
    test_end = pd.Timestamp(args.test_end)
    if not (train_end < test_start <= test_end):
        raise ValueError("require train-end < test-start <= test-end")

    pretest = frame[frame["date"] <= train_end].copy()
    test = frame[(frame["date"] >= test_start) & (frame["date"] <= test_end)].copy()
    if pretest.empty or test.empty:
        raise SystemExit("Train or test split is empty")

    threshold = float(pretest["move_1d"].quantile(args.target_positive_rate))
    pretest["label"] = (pretest["move_1d"] <= threshold).astype(int)
    test["label"] = (test["move_1d"] <= threshold).astype(int)

    split_index = int(math.floor(len(pretest) * (1.0 - args.validation_fraction)))
    train = pretest.iloc[:split_index].copy().reset_index(drop=True)
    valid = pretest.iloc[split_index:].copy().reset_index(drop=True)
    test = test.reset_index(drop=True)

    device = _device()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["query", "value"],
        bias="none",
    )
    model = get_peft_model(base_model, lora_config).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Device={device}; trainable={trainable_params:,}/{total_params:,} "
        f"({trainable_params / total_params:.3%})",
        flush=True,
    )

    train_ds = ChunkDataset(
        train,
        tokenizer,
        chunk_tokens=args.chunk_tokens,
        stride=args.chunk_stride,
        max_chunks=args.train_max_chunks,
    )
    valid_ds = ChunkDataset(
        valid,
        tokenizer,
        chunk_tokens=args.chunk_tokens,
        stride=args.chunk_stride,
        max_chunks=args.eval_max_chunks,
    )
    test_ds = ChunkDataset(
        test,
        tokenizer,
        chunk_tokens=args.chunk_tokens,
        stride=args.chunk_stride,
        max_chunks=args.eval_max_chunks,
    )
    collate = _make_collate(tokenizer)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(round(total_steps * args.warmup_fraction))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history: list[dict[str, Any]] = []
    best_valid_auc = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch.pop("doc_ids")
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.detach().cpu())

            if step == 1 or step % args.progress_every == 0 or step == len(train_loader):
                print(
                    f"epoch {epoch}/{args.epochs} step {step}/{len(train_loader)} "
                    f"loss={running_loss / step:.4f}",
                    flush=True,
                )

        valid_p = _predict_documents(
            model,
            valid_loader,
            device=device,
            n_documents=len(valid),
        )
        valid_metrics = _metrics(valid["label"].to_numpy(), valid_p)
        epoch_record = {
            "epoch": epoch,
            "train_chunk_loss": running_loss / max(len(train_loader), 1),
            "validation": valid_metrics,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, indent=2), flush=True)

        if valid_metrics["auroc"] > best_valid_auc:
            best_valid_auc = valid_metrics["auroc"]
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
                if "lora_" in name or "modules_to_save" in name
            }

    if best_state is not None:
        current = model.state_dict()
        current.update(best_state)
        model.load_state_dict(current, strict=False)

    valid_p = _predict_documents(model, valid_loader, device=device, n_documents=len(valid))
    test_p = _predict_documents(model, test_loader, device=device, n_documents=len(test))

    args.adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.adapter_dir)
    tokenizer.save_pretrained(args.adapter_dir)

    results = {
        "repo_id": args.repo_id,
        "split": args.split,
        "rows_loaded": int(len(rows)),
        "rows_after_filtering": int(len(frame)),
        "train_end": args.train_end,
        "test_start": args.test_start,
        "test_end": args.test_end,
        "pretest_rows": int(len(pretest)),
        "train_rows": int(len(train)),
        "validation_rows": int(len(valid)),
        "test_rows": int(len(test)),
        "validation_fraction": args.validation_fraction,
        "target": "move_1d downside tail",
        "target_positive_rate_requested": args.target_positive_rate,
        "training_threshold_move_1d": threshold,
        "train_positive_rate": float(train["label"].mean()),
        "validation_positive_rate": float(valid["label"].mean()),
        "test_positive_rate": float(test["label"].mean()),
        "model_name": args.model_name,
        "device": str(device),
        "chunk_tokens": args.chunk_tokens,
        "chunk_stride": args.chunk_stride,
        "train_max_chunks": args.train_max_chunks,
        "eval_max_chunks": args.eval_max_chunks,
        "train_chunks": int(len(train_ds)),
        "validation_chunks": int(len(valid_ds)),
        "test_chunks": int(len(test_ds)),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
        "history": history,
        "best_validation_auroc": float(best_valid_auc),
        "validation": _metrics(valid["label"].to_numpy(), valid_p),
        "test": _metrics(test["label"].to_numpy(), test_p),
        "reference_auroc": {
            "tfidf_text_only": 0.6118578653180728,
            "frozen_chunked_finbert_text_only": 0.5788952409608344,
        },
        "adapter_dir": str(args.adapter_dir),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, allow_nan=True))
    print(f"\nArtifact written to {args.output}")
    print(f"Adapter written to {args.adapter_dir}")


if __name__ == "__main__":
    main()
