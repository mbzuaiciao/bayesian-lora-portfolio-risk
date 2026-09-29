"""Run Stage 2 reproducibility checks for promising earnings LoRA configs.

Runs two Stage-1 finalists (r=8 and r=16, both lr=2e-4) across three seeds.
The launcher is resume-aware: if a run JSON already exists and is readable, it
is reused rather than rerun. A summary grouped by configuration is rewritten
after each completed/recovered run.

Hyperparameter conclusions must be based on validation AUROC. Test metrics are
reported only for diagnosis/final reporting and are never used to choose a winner.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

TFIDF_REFERENCE_AUROC = 0.6118578653180728
FROZEN_FINBERT_REFERENCE_AUROC = 0.5788952409608344


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="RudrakshNanavaty/earnings-call-data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-rows", type=int, default=20_000)
    parser.add_argument("--train-end", default="2021-12-31")
    parser.add_argument("--test-start", default="2022-01-01")
    parser.add_argument("--test-end", default="2023-12-31")
    parser.add_argument("--target-positive-rate", type=float, default=0.20)
    parser.add_argument("--chunk-tokens", type=int, default=384)
    parser.add_argument("--chunk-stride", type=int, default=64)
    parser.add_argument("--train-max-chunks", type=int, default=2)
    parser.add_argument("--eval-max-chunks", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 777])
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/earnings_lora_stage2_repro"),
    )
    return parser.parse_args()


def _cuda_available() -> bool:
    probe = subprocess.run(
        [sys.executable, "-c", "import torch; print(int(torch.cuda.is_available()))"],
        check=True,
        capture_output=True,
        text=True,
    )
    return probe.stdout.strip() == "1"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _row_from_result(
    *,
    tag: str,
    rank: int,
    learning_rate: float,
    seed: int,
    output: Path,
    adapter_dir: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    validation = result["validation"]
    test = result["test"]
    return {
        "tag": tag,
        "lora_r": rank,
        "learning_rate": learning_rate,
        "seed": seed,
        "epochs": result.get("epochs"),
        "train_max_chunks": result.get("train_max_chunks"),
        "validation_auroc": validation["auroc"],
        "validation_brier": validation["brier"],
        "validation_nll": validation["nll"],
        "test_auroc": test["auroc"],
        "test_brier": test["brier"],
        "test_nll": test["nll"],
        "delta_test_vs_frozen_finbert": test["auroc"] - FROZEN_FINBERT_REFERENCE_AUROC,
        "delta_test_vs_tfidf": test["auroc"] - TFIDF_REFERENCE_AUROC,
        "artifact": str(output),
        "adapter_dir": str(adapter_dir),
    }


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def _stdev(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _write_summary(output_dir: Path, rows: list[dict[str, Any]], runs_total: int) -> None:
    grouped: list[dict[str, Any]] = []
    keys = sorted({(int(row["lora_r"]), float(row["learning_rate"])) for row in rows})
    for rank, learning_rate in keys:
        members = [
            row
            for row in rows
            if int(row["lora_r"]) == rank and float(row["learning_rate"]) == learning_rate
        ]
        val_auc = [float(row["validation_auroc"]) for row in members]
        test_auc = [float(row["test_auroc"]) for row in members]
        grouped.append(
            {
                "config": f"r{rank}_lr{learning_rate:.0e}".replace("+", ""),
                "lora_r": rank,
                "learning_rate": learning_rate,
                "seeds_completed": len(members),
                "validation_auroc_mean": _mean(val_auc),
                "validation_auroc_std": _stdev(val_auc),
                "validation_auroc_min": min(val_auc),
                "validation_auroc_max": max(val_auc),
                "test_auroc_mean": _mean(test_auc),
                "test_auroc_std": _stdev(test_auc),
                "runs": sorted(members, key=lambda row: int(row["seed"])),
            }
        )

    grouped.sort(key=lambda row: row["validation_auroc_mean"], reverse=True)
    payload = {
        "selection_metric": "mean_validation_auroc_across_seeds",
        "important_note": (
            "Use validation AUROC across seeds to compare configurations. "
            "Test AUROC is diagnostic/final-reporting only and must not drive selection."
        ),
        "tfidf_reference_auroc": TFIDF_REFERENCE_AUROC,
        "frozen_finbert_reference_auroc": FROZEN_FINBERT_REFERENCE_AUROC,
        "runs_completed": len(rows),
        "runs_total": runs_total,
        "configs": grouped,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.require_cuda and not _cuda_available():
        raise SystemExit("CUDA is required but torch.cuda.is_available() is false")

    configs = [
        {"rank": 8, "learning_rate": 2e-4},
        {"rank": 16, "learning_rate": 2e-4},
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    runs_total = len(configs) * len(args.seeds)
    run_index = 0

    for config in configs:
        rank = int(config["rank"])
        learning_rate = float(config["learning_rate"])
        for seed in args.seeds:
            run_index += 1
            base = f"r{rank}_lr{learning_rate:.0e}".replace("+", "")
            tag = f"{base}_seed{seed}"
            output = args.output_dir / f"{tag}.json"
            adapter_dir = args.output_dir / f"adapter_{tag}"

            if output.exists():
                try:
                    result = _load_json(output)
                    row = _row_from_result(
                        tag=tag,
                        rank=rank,
                        learning_rate=learning_rate,
                        seed=seed,
                        output=output,
                        adapter_dir=adapter_dir,
                        result=result,
                    )
                    rows.append(row)
                    print(
                        f"[{run_index}/{runs_total}] {tag} already complete; skipping",
                        flush=True,
                    )
                    _write_summary(args.output_dir, rows, runs_total)
                    continue
                except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                    print(
                        f"[{run_index}/{runs_total}] {tag} artifact unreadable; rerunning",
                        flush=True,
                    )

            command = [
                sys.executable,
                "scripts/run_earnings_lora_baseline.py",
                "--repo-id",
                args.repo_id,
                "--split",
                args.split,
                "--max-rows",
                str(args.max_rows),
                "--train-end",
                args.train_end,
                "--test-start",
                args.test_start,
                "--test-end",
                args.test_end,
                "--target-positive-rate",
                str(args.target_positive_rate),
                "--chunk-tokens",
                str(args.chunk_tokens),
                "--chunk-stride",
                str(args.chunk_stride),
                "--train-max-chunks",
                str(args.train_max_chunks),
                "--eval-max-chunks",
                str(args.eval_max_chunks),
                "--batch-size",
                str(args.batch_size),
                "--epochs",
                str(args.epochs),
                "--learning-rate",
                str(learning_rate),
                "--lora-r",
                str(rank),
                "--lora-alpha",
                str(2 * rank),
                "--seed",
                str(seed),
                "--output",
                str(output),
                "--adapter-dir",
                str(adapter_dir),
            ]

            print(
                f"\n=== Stage 2 run {run_index}/{runs_total}: "
                f"r={rank}, lr={learning_rate:g}, seed={seed} ===",
                flush=True,
            )
            subprocess.run(command, check=True)
            result = _load_json(output)
            rows.append(
                _row_from_result(
                    tag=tag,
                    rank=rank,
                    learning_rate=learning_rate,
                    seed=seed,
                    output=output,
                    adapter_dir=adapter_dir,
                    result=result,
                )
            )
            _write_summary(args.output_dir, rows, runs_total)

    _write_summary(args.output_dir, rows, runs_total)
    summary = _load_json(args.output_dir / "summary.json")
    print("\n=== Stage 2 reproducibility summary ===")
    for position, config in enumerate(summary["configs"], start=1):
        print(
            f"{position}. {config['config']:14s} "
            f"val_auc={config['validation_auroc_mean']:.4f}±{config['validation_auroc_std']:.4f} "
            f"test_auc={config['test_auroc_mean']:.4f}±{config['test_auroc_std']:.4f}"
        )
    print(f"Summary written to {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
