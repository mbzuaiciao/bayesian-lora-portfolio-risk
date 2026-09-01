# GPU LoRA sweep

The first deterministic LoRA run improved test AUROC only modestly over frozen FinBERT:

- frozen chunked FinBERT: 0.5789
- first LoRA run: 0.5834
- TF-IDF transcript baseline: 0.6119

This does not yet justify Bayesian LoRA. Before adding posterior uncertainty, run a small deterministic tuning stage on CUDA.

## Stage 1: tune rank and learning rate

Keep the target, chronological split, chunking policy, and random seed fixed. Sweep only:

- LoRA rank: 4, 8, 16
- learning rate: 5e-5, 1e-4, 2e-4
- epochs: 4
- train chunks: 2
- eval chunks: 4

Use `scripts/run_earnings_lora_sweep.py`. The launcher ranks configurations by **validation AUROC**, never by test AUROC. Test metrics are included only as a diagnostic.

Recommended CUDA command:

```bash
uv run python scripts/run_earnings_lora_sweep.py \
  --require-cuda \
  --batch-size 32 \
  --epochs 4 \
  --output-dir artifacts/earnings_lora_sweep_stage1
```

If GPU memory is insufficient, reduce `--batch-size` to 16 or 8.

The launcher writes every run separately and continuously refreshes:

```text
artifacts/earnings_lora_sweep_stage1/leaderboard.json
```

This makes an interrupted cloud job recoverable: completed run artifacts remain on disk.

## Decision after Stage 1

Do not automatically run a large sweep.

Proceed to Stage 2 only if the best validation-selected configuration shows a meaningful improvement over the initial LoRA setup and appears plausibly competitive with the TF-IDF benchmark.

Stage 2 should then test only a small number of questions, for example:

1. 2 versus 4 transcript chunks during training;
2. one or two nearby learning rates around the Stage-1 winner;
3. multiple random seeds for the final deterministic configuration.

Only after establishing a competitive deterministic LoRA baseline should the project move to Bayesian LoRA and epistemic-uncertainty experiments.

## Cloud workflow

For Lightning AI or Google Cloud, the minimal workflow is:

```bash
git clone https://github.com/mbzuaiciao/bayesian-lora-portfolio-risk.git
cd bayesian-lora-portfolio-risk
uv sync
hf auth login
nvidia-smi
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

Then launch the Stage-1 command above.

A Hugging Face read token is sufficient for the public dataset/model downloads used here.
