"""Train Bayesian LoRA FinBERT and evaluate whether epistemic uncertainty is useful.

The deterministic Stage-2 winner (r=16, lr=2e-4) provides the default capacity
and optimizer settings. Hyperparameters are still selected only from pre-test
validation data. The untouched 2022-2023 test period is used for final diagnostics.

The primary research question is not whether Bayesian LoRA raises AUROC. It is
whether posterior uncertainty identifies unreliable predictions and temporal shift.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Reuse the audited transcript chunking / document aggregation code from the
# deterministic experiment so only the adaptation method changes.
import run_earnings_lora_baseline as det
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from bayesian_lora_portfolio_risk.uncertainty.bayesian_lora import (
    count_variational_parameters,
    inject_bert_attention_bayesian_lora,
    model_kl_divergence,
    set_posterior_sampling,
)
from bayesian_lora_portfolio_risk.uncertainty.metrics import (
    error_rate_by_uncertainty_quantile,
    expected_calibration_error,
    selective_risk_curve,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="RudrakshNanavaty/earnings-call-data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-rows", type=int, default=20_000)
    parser.add_argument("--train-end", default="2021-12-31")
    parser.add_argument("--test-start", default="2022-01-01")
    parser.add_argument("--test-end", default="2023-12-31")
    parser.add_argument("--target-positive-rate", type=float, default=0.20)
    parser.add_argument("--model-name", default="ProsusAI/finbert")
    parser.add_argument("--chunk-tokens", type=int, default=384)
    parser.add_argument("--chunk-stride", type=int, default=64)
    parser.add_argument("--train-max-chunks", type=int, default=2)
    parser.add_argument("--eval-max-chunks", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--posterior-init-std", type=float, default=0.01)
    parser.add_argument("--prior-std", type=float, default=1.0)
    parser.add_argument("--kl-beta", type=float, default=0.1)
    parser.add_argument("--mc-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/earnings_bayesian_lora.json"),
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("artifacts/earnings_bayesian_lora_predictions.csv"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/earnings_bayesian_lora_state.pt"),
    )
    return parser.parse_args()


def load_frame(args: argparse.Namespace) -> pd.DataFrame:
    from datasets import load_dataset

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
    frame = frame.dropna(subset=["date", "move_1d"])
    frame = frame[frame["earnings_transcript"].str.len() > 0]
    return frame.sort_values("date").reset_index(drop=True)


def split_frame(frame: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    train_end = pd.Timestamp(args.train_end)
    test_start = pd.Timestamp(args.test_start)
    test_end = pd.Timestamp(args.test_end)
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
    return train, valid, test.reset_index(drop=True), threshold


def document_probabilities(model: torch.nn.Module, loader: DataLoader[Any], *, device: torch.device, n_documents: int) -> np.ndarray:
    return det._predict_documents(model, loader, device=device, n_documents=n_documents)


def posterior_predict(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    *,
    device: torch.device,
    n_documents: int,
    mc_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples: list[np.ndarray] = []
    set_posterior_sampling(model, True)
    for index in range(mc_samples):
        print(f"posterior sample {index + 1}/{mc_samples}", flush=True)
        samples.append(document_probabilities(model, loader, device=device, n_documents=n_documents))
    matrix = np.stack(samples, axis=0)
    return matrix.mean(axis=0), matrix.var(axis=0), matrix


def risk_at_coverages(correct: np.ndarray, uncertainty: np.ndarray) -> dict[str, float]:
    coverage, risk = selective_risk_curve(correct, uncertainty)
    result: dict[str, float] = {}
    for target in (0.5, 0.7, 0.8, 0.9, 1.0):
        index = int(np.argmin(np.abs(coverage - target)))
        result[f"coverage_{target:.1f}"] = float(risk[index])
    return result


def main() -> None:
    args = parse_args()
    if args.mc_samples < 2:
        raise ValueError("mc-samples must be at least 2")
    det._seed_everything(args.seed)

    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    frame = load_frame(args)
    train, valid, test, threshold = split_frame(frame, args)
    device = det._device()
    if args.require_cuda and device.type != "cuda":
        raise SystemExit("CUDA required but unavailable")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2, ignore_mismatched_sizes=True
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    injected = inject_bert_attention_bayesian_lora(
        model,
        rank=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        init_std=args.posterior_init_std,
        prior_std=args.prior_std,
    )
    # The task-specific binary classifier is not Bayesian in this first experiment;
    # keeping it deterministic isolates uncertainty in the LoRA adaptation itself.
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    model.to(device)

    train_ds = det.ChunkDataset(train, tokenizer, chunk_tokens=args.chunk_tokens, stride=args.chunk_stride, max_chunks=args.train_max_chunks)
    valid_ds = det.ChunkDataset(valid, tokenizer, chunk_tokens=args.chunk_tokens, stride=args.chunk_stride, max_chunks=args.eval_max_chunks)
    test_ds = det.ChunkDataset(test, tokenizer, chunk_tokens=args.chunk_tokens, stride=args.chunk_stride, max_chunks=args.eval_max_chunks)
    collate = det._make_collate(tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    variational = count_variational_parameters(model)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(
        f"Device={device}; injected={len(injected)} projections; variational={variational['total']:,}; trainable={trainable:,}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(round(total_steps * args.warmup_fraction)),
        num_training_steps=total_steps,
    )

    history: list[dict[str, Any]] = []
    best_auc = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    dataset_size = len(train_ds)

    for epoch in range(1, args.epochs + 1):
        model.train()
        set_posterior_sampling(model, True)
        total_loss = pred_loss_sum = scaled_kl_sum = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch.pop("doc_ids")
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            pred_loss = outputs.loss
            raw_kl = model_kl_divergence(model)
            scaled_kl = (args.kl_beta / dataset_size) * raw_kl
            loss = pred_loss + scaled_kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach().cpu())
            pred_loss_sum += float(pred_loss.detach().cpu())
            scaled_kl_sum += float(scaled_kl.detach().cpu())
            if step == 1 or step % args.progress_every == 0 or step == len(train_loader):
                print(
                    f"epoch {epoch}/{args.epochs} step {step}/{len(train_loader)} "
                    f"loss={total_loss / step:.4f} pred={pred_loss_sum / step:.4f} kl={scaled_kl_sum / step:.4f}",
                    flush=True,
                )

        model.eval()
        set_posterior_sampling(model, False)
        valid_p = document_probabilities(model, valid_loader, device=device, n_documents=len(valid))
        valid_metrics = det._metrics(valid["label"].to_numpy(), valid_p)
        record = {
            "epoch": epoch,
            "train_total_loss": total_loss / len(train_loader),
            "train_pred_loss": pred_loss_sum / len(train_loader),
            "train_scaled_kl": scaled_kl_sum / len(train_loader),
            "validation_posterior_mean": valid_metrics,
        }
        history.append(record)
        print(json.dumps(record, indent=2), flush=True)
        if valid_metrics["auroc"] > best_auc:
            best_auc = float(valid_metrics["auroc"])
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.checkpoint)

    model.eval()
    set_posterior_sampling(model, False)
    valid_mean_det = document_probabilities(model, valid_loader, device=device, n_documents=len(valid))
    test_mean_det = document_probabilities(model, test_loader, device=device, n_documents=len(test))
    valid_det_metrics = det._metrics(valid["label"].to_numpy(), valid_mean_det)
    test_det_metrics = det._metrics(test["label"].to_numpy(), test_mean_det)

    test_mean, test_var, _ = posterior_predict(
        model,
        test_loader,
        device=device,
        n_documents=len(test),
        mc_samples=args.mc_samples,
    )
    y_test = test["label"].to_numpy()
    posterior_metrics = det._metrics(y_test, test_mean)
    posterior_metrics["ece"] = expected_calibration_error(y_test, test_mean)
    predicted = test_mean >= 0.5
    correct = predicted == y_test
    incorrect = (~correct).astype(int)
    error_detection_auc = float(roc_auc_score(incorrect, test_var)) if np.unique(incorrect).size == 2 else float("nan")
    quantile_error = error_rate_by_uncertainty_quantile(correct, test_var, n_quantiles=5)

    annual: dict[str, dict[str, float]] = {}
    for year in sorted(test["date"].dt.year.unique()):
        mask = test["date"].dt.year.to_numpy() == year
        metrics = det._metrics(y_test[mask], test_mean[mask])
        annual[str(int(year))] = {
            **metrics,
            "mean_epistemic_variance": float(np.mean(test_var[mask])),
            "positive_rate": float(np.mean(y_test[mask])),
            "n": int(np.sum(mask)),
        }

    predictions = pd.DataFrame(
        {
            "date": test["date"],
            "label": y_test,
            "move_1d": test["move_1d"].to_numpy(),
            "posterior_mean": test_mean,
            "epistemic_variance": test_var,
            "posterior_mean_deterministic": test_mean_det,
            "correct_at_0.5": correct,
        }
    )
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.predictions_output, index=False)

    result = {
        "repo_id": args.repo_id,
        "rows_loaded": args.max_rows,
        "train_rows": len(train),
        "validation_rows": len(valid),
        "test_rows": len(test),
        "training_threshold_move_1d": threshold,
        "device": str(device),
        "model_name": args.model_name,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "kl_beta": args.kl_beta,
        "prior_std": args.prior_std,
        "posterior_init_std": args.posterior_init_std,
        "mc_samples": args.mc_samples,
        "injected_projections": len(injected),
        "variational_parameters": variational,
        "history": history,
        "best_validation_auroc_posterior_mean": best_auc,
        "validation_posterior_mean_deterministic": valid_det_metrics,
        "test_posterior_mean_deterministic": test_det_metrics,
        "test_posterior_predictive": posterior_metrics,
        "uncertainty": {
            "mean_epistemic_variance": float(np.mean(test_var)),
            "median_epistemic_variance": float(np.median(test_var)),
            "error_detection_auroc": error_detection_auc,
            "error_rate_by_uncertainty_quintile_low_to_high": quantile_error.tolist(),
            "selective_risk": risk_at_coverages(correct, test_var),
            "annual": annual,
        },
        "reference_auroc": {
            "tfidf": 0.6118578653180728,
            "frozen_finbert": 0.5788952409608344,
            "deterministic_lora_r16_stage2_validation_mean": 0.5989710816542116,
            "deterministic_lora_r16_stage2_test_mean": 0.5873258805784758,
        },
        "predictions_output": str(args.predictions_output),
        "checkpoint": str(args.checkpoint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Artifact written to {args.output}")
    print(f"Predictions written to {args.predictions_output}")


if __name__ == "__main__":
    main()
