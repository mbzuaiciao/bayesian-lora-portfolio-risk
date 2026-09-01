"""Launch a small deterministic LoRA hyperparameter sweep.

Stage 1 intentionally tunes only learning rate and LoRA rank while keeping the
rest of the earnings downside experiment fixed. Each configuration delegates to
``run_earnings_lora_baseline.py`` and writes its own JSON artifact and adapter.
A compact leaderboard is written after all runs finish.

Designed primarily for a CUDA machine (Lightning AI / GCP), although it can run
elsewhere if ``--require-cuda`` is omitted.
"""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/earnings_lora_sweep_stage1"),
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


def _load_result(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    if args.require_cuda and not _cuda_available():
        raise SystemExit("CUDA is required for this sweep but torch.cuda.is_available() is false")

    # Small first-stage grid: enough to test whether the initial r=8, lr=2e-4 run
    # was simply under-tuned without spending heavily on context/seed sweeps yet.
    learning_rates = [5e-5, 1e-4, 2e-4]
    ranks = [4, 8, 16]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []

    total = len(learning_rates) * len(ranks)
    run_index = 0
    for rank in ranks:
        for learning_rate in learning_rates:
            run_index += 1
            tag = f"r{rank}_lr{learning_rate:.0e}".replace("+", "")
            output = args.output_dir / f"{tag}.json"
            adapter_dir = args.output_dir / f"adapter_{tag}"

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
                str(args.seed),
                "--output",
                str(output),
                "--adapter-dir",
                str(adapter_dir),
            ]

            print(
                f"\n=== Sweep run {run_index}/{total}: r={rank}, lr={learning_rate:g} ===",
                flush=True,
            )
            subprocess.run(command, check=True)
            result = _load_result(output)
            test_metrics = result["test"]
            validation_metrics = result["validation"]
            runs.append(
                {
                    "tag": tag,
                    "lora_r": rank,
                    "learning_rate": learning_rate,
                    "epochs": args.epochs,
                    "train_max_chunks": args.train_max_chunks,
                    "validation_auroc": validation_metrics["auroc"],
                    "test_auroc": test_metrics["auroc"],
                    "test_brier": test_metrics["brier"],
                    "test_nll": test_metrics["nll"],
                    "delta_vs_frozen_finbert": test_metrics["auroc"]
                    - FROZEN_FINBERT_REFERENCE_AUROC,
                    "delta_vs_tfidf": test_metrics["auroc"] - TFIDF_REFERENCE_AUROC,
                    "artifact": str(output),
                    "adapter_dir": str(adapter_dir),
                }
            )

            leaderboard = sorted(runs, key=lambda row: row["validation_auroc"], reverse=True)
            partial = {
                "selection_metric": "validation_auroc",
                "tfidf_reference_auroc": TFIDF_REFERENCE_AUROC,
                "frozen_finbert_reference_auroc": FROZEN_FINBERT_REFERENCE_AUROC,
                "runs_completed": len(runs),
                "runs_total": total,
                "leaderboard": leaderboard,
            }
            (args.output_dir / "leaderboard.json").write_text(
                json.dumps(partial, indent=2) + "\n", encoding="utf-8"
            )

    leaderboard = sorted(runs, key=lambda row: row["validation_auroc"], reverse=True)
    final = {
        "selection_metric": "validation_auroc",
        "important_note": (
            "Hyperparameters are ranked by validation AUROC. Test AUROC is reported for diagnosis "
            "but must not be used to choose the winning configuration."
        ),
        "tfidf_reference_auroc": TFIDF_REFERENCE_AUROC,
        "frozen_finbert_reference_auroc": FROZEN_FINBERT_REFERENCE_AUROC,
        "runs_completed": len(runs),
        "runs_total": total,
        "leaderboard": leaderboard,
    }
    leaderboard_path = args.output_dir / "leaderboard.json"
    leaderboard_path.write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")

    print("\n=== Stage-1 leaderboard (selected by validation AUROC) ===")
    for position, row in enumerate(leaderboard, start=1):
        print(
            f"{position:2d}. {row['tag']:14s} "
            f"val_auc={row['validation_auroc']:.4f} "
            f"test_auc={row['test_auroc']:.4f} "
            f"delta_vs_tfidf={row['delta_vs_tfidf']:+.4f}"
        )
    print(f"\nLeaderboard written to {leaderboard_path}")


if __name__ == "__main__":
    main()
