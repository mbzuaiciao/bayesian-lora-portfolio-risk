"""Run a small KL-beta sweep for the earnings Bayesian-LoRA experiment.

Selection discipline:
- configurations are ranked ONLY by pre-2022 validation posterior-mean AUROC;
- test metrics emitted by the underlying experiment remain diagnostic only;
- the sweep also reports the final KL/prediction-loss ratio so we can see whether
  the ELBO is still dominated by the KL term.

This is intentionally a small sweep. The first beta=0.1 pilot showed useful
uncertainty ordering but a scaled KL roughly 19x the predictive loss. The default
betas below bracket the region where KL and predictive loss should be comparable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--betas", nargs="+", type=float, default=[0.001, 0.003, 0.01])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--mc-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/earnings_bayesian_lora_beta_sweep"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/earnings_bayesian_lora_beta_sweep/summary.json"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun configurations even if their JSON artifact already exists.",
    )
    return parser.parse_args()


def beta_tag(beta: float) -> str:
    return f"beta_{beta:g}".replace(".", "p").replace("-", "m")


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for beta in args.betas:
        tag = beta_tag(beta)
        output = args.output_dir / f"{tag}.json"
        predictions = args.output_dir / f"{tag}_predictions.csv"
        checkpoint = args.output_dir / f"{tag}_state.pt"

        if output.exists() and not args.force:
            print(f"[resume] {tag}: using {output}", flush=True)
        else:
            command = [
                sys.executable,
                "scripts/run_earnings_bayesian_lora.py",
                "--batch-size",
                str(args.batch_size),
                "--epochs",
                str(args.epochs),
                "--lora-r",
                str(args.lora_r),
                "--learning-rate",
                str(args.learning_rate),
                "--kl-beta",
                str(beta),
                "--mc-samples",
                str(args.mc_samples),
                "--seed",
                str(args.seed),
                "--output",
                str(output),
                "--predictions-output",
                str(predictions),
                "--checkpoint",
                str(checkpoint),
            ]
            if args.require_cuda:
                command.append("--require-cuda")
            print(f"[run] {tag}: {' '.join(command)}", flush=True)
            subprocess.run(command, check=True)

        result = load_json(output)
        final_epoch = result["history"][-1]
        pred_loss = float(final_epoch["train_pred_loss"])
        scaled_kl = float(final_epoch["train_scaled_kl"])
        records.append(
            {
                "kl_beta": beta,
                "best_validation_auroc_posterior_mean": float(
                    result["best_validation_auroc_posterior_mean"]
                ),
                "final_train_pred_loss": pred_loss,
                "final_train_scaled_kl": scaled_kl,
                "final_kl_to_pred_loss_ratio": scaled_kl / pred_loss if pred_loss else None,
                "diagnostic_test_posterior_predictive_auroc": float(
                    result["test_posterior_predictive"]["auroc"]
                ),
                "artifact": str(output),
            }
        )

        # Persist partial progress so an interrupted VM/session can resume safely.
        partial = {
            "selection_metric": "best_validation_auroc_posterior_mean",
            "important_note": (
                "Select beta only from validation AUROC and ELBO diagnostics. "
                "Test metrics are diagnostic and must not drive selection."
            ),
            "configs_completed": len(records),
            "configs_total": len(args.betas),
            "configs": sorted(
                records,
                key=lambda row: row["best_validation_auroc_posterior_mean"],
                reverse=True,
            ),
        }
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(partial, indent=2) + "\n")

    summary = load_json(args.summary)
    winner = summary["configs"][0] if summary["configs"] else None
    summary["selected_by_validation"] = winner
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Summary written to {args.summary}", flush=True)


if __name__ == "__main__":
    main()
