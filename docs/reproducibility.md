# Environment Setup and Reproducibility Guide

This guide describes environment setup, experiment and result provenance, expected local prediction artifacts, and the exact commands connecting each experimental stage of the completed study.

---

## 1. Environment Setup

The repository uses [`uv`](https://docs.astral.sh/uv/) for fast, deterministic Python environment management.

### Prerequisites
- Python >= 3.11 (Python 3.11 or 3.12 recommended; 3.14 tested)
- Git
- `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh` or `brew install uv`)

### Installation
Clone the repository and install the frozen environment:
```bash
git clone https://github.com/mbzuaiciao/bayesian-lora-portfolio-risk.git
cd bayesian-lora-portfolio-risk
uv sync
```

---

## 2. Consistency & Reproducibility Verification

The repository supports two verification tiers via `scripts/verify_reproducibility.py`:

### Tier 1: Clean-Clone Verification (Default, Fast, CI-safe)
To establish that all committed result summaries, distribution-shift audits, and downstream decision metrics are internally consistent without needing any local artifacts, run:

```bash
uv run python scripts/verify_reproducibility.py
```

This automated clean-clone check:
1. Validates that all 27 committed result summary JSON files under `results/` are non-empty and well-formed.
2. Verifies statistical invariants across development, holdout, concentration-control, model-risk penalty, and subgroup shift summaries.
3. Confirms that all mathematical relationships (e.g., $\Delta\text{AUROC} = \text{AUROC}_{\text{aug}} - \text{AUROC}_{\text{base}}$) hold exactly across committed outputs.
4. Requires **zero** untracked local artifacts, passing reliably in CI environments from a fresh `git clone`.

### Tier 2: Deep Local-Artifact Recomputation (`--deep`)
When local prediction artifacts are present in `artifacts/` (e.g. from local inference runs or research storage), run:

```bash
uv run python scripts/verify_reproducibility.py --deep
```

This deep check recomputes:
1. Out-of-sample error detection AUROCs and selective risk metrics directly from raw Monte Carlo prediction CSVs for all 3 seeds (42, 123, 777).
2. Effective position counts ($N_{\text{eff}}$), $\gamma$-bisection concentration matching, and equal-gross allocation metrics, verifying exact bitwise/floating-point reproduction of committed research results.

You can also run the full test suite, linter, and type checker:
```bash
uv run pytest
uv run ruff check .
uv run pyright
```

---

## 3. Experiment and Result Provenance

The study proceeded across sequential stages, each freezing preceding hyperparameters and evaluation splits before advancing:

```
[Phase 0: Daily News Pilot (FNSPID)]
   │ Rejected: text signal failed the incremental-information gate beyond trailing volatility
   ▼
[Stage 1: Earnings-Event Formulation & Gate 0]
   │ 1-day downside move: r^{(1)} <= tau (tau frozen on pre-2022 training quantile at 20%)
   │ Passed Gate 0 (TF-IDF AUROC ~0.612, frozen FinBERT AUROC ~0.579)
   ▼
[Stage 2: Deterministic LoRA Hyperparameter Selection]
   │ Grid search over rank in {4, 8, 16} and lr in {5e-5, 1e-4, 2e-4}
   │ Multi-seed replication selected rank=16, lr=2e-4 on pre-2022 validation AUROC
   ▼
[Stage 3: Bayesian LoRA Adaptation & KL-Beta Calibration]
   │ Evaluated KL beta in {0.001, 0.003, 0.01}; beta=0.003 selected on validation ELBO/AUROC
   │ Verified across seeds (42, 123, 777) with 32 Monte Carlo posterior samples
   ▼
[Stage 4: Confidence Uncertainty Control (Primary Falsification 1)]
   │ Compared epistemic variance against deterministic confidence: 1 - 2*|p - 0.5|
   │ Result: Ordinary confidence is at least as strong for error detection
   ▼
[Stage 5: Downstream Portfolio Allocation & Concentration Control (Primary Falsification 2)]
   │ Apparent epistemic sizing advantage explained by increased concentration
   │ Under effective-N matched concentration, epistemic allocation underperforms confidence
   ▼
[Stage 6: Model-Risk Penalty Selection (Primary Falsification 3)]
   │ Exponential penalty exp(-lambda * u_epi); lambda chosen via calibration grid
   │ Selected lambda* = 0.0 across all seeds and evaluation periods
   ▼
[Stage 7: Distribution-Shift Subgroups (Primary Falsification 4)]
   │ Unseen companies (N=35), sparse-history (N=47), and underrepresented sectors (N=204)
   │ Clear negative result on the largest subgroup (sectors); search stopped to avoid fishing
```

---

## 4. Expected Artifacts vs. Committed Results

| Location | Contents | Committed to Git? | Status / Provenance |
|---|---|---|---|
| `results/` | Small summary JSONs for all sweep stages, Bayesian repro, concentration controls, model risk penalties, shift subgroups, and sensitivity analysis. | **Yes** | Fully committed. Serves as ground truth for all numerical tables in docs. Clean-clone verification checks this directory. |
| `artifacts/earnings_bayesian_lora_uncertainty_control/` | `beta_0p003_seed_{42,123,777}_mc32_predictions.csv` | **No** (Local artifact) | Excluded from git via `.gitignore`. Generated locally during model inference. Used for optional deep verification (`--deep`). |
| `artifacts/earnings_bayesian_lora_final_holdout/` | `beta_0p003_seed_{42,123,777}_mc32_predictions.csv` | **No** (Local artifact) | Excluded from git via `.gitignore`. Generated locally during holdout inference. Used for optional deep verification (`--deep`). |
| `artifacts/*_state.pt`, `*.safetensors` | Model weights, PyTorch checkpoints, and full raw text dumps (> 100MB). | **No** (Local only) | Intentionally not tracked in git due to file size constraints. Reconstructible via training scripts if GPU available. |

---

### 4.1 Calibration Protocol Evolution (FALSIFY-04 Remediation)

A methodological distinction exists across the portfolio allocation scripts regarding how epistemic reliability $1 - F_{\text{cal}}(\sigma^2_{\text{epi}})$ is calibrated:

1. **Initial Exploratory Pooled Calibration Protocol** (`analyze_earnings_event_risk_allocation.py`):
   - In the initial raw-sizing exploratory study, epistemic variance was normalized using an empirical CDF reference distribution $F_{\text{dev}}(\cdot)$ fit across the entire pooled 2022–2023 development sample ($N=2,369$).
   - This single pooled calibration reference was then applied to score the 2024–2025 final holdout. While chronologically preceding the holdout, this pooled approach did not evaluate 2023 out-of-sample or segment the development period sequentially.

2. **Sequential / Expanding-Window Calibration Protocol** (`analyze_earnings_event_risk_allocation_concentration_control.py` & `analyze_earnings_model_risk_penalty.py`):
   - To strictly guard against temporal leakage and reflect real-world sequential deployment, later allocation scripts adopted a rigorous expanding-window calibration:
     - **Period 1 (2023 Evaluation)**: The epistemic calibration distribution $F_{2022}(\cdot)$ and target effective position counts $N_{\text{eff}}$ were fit strictly on 2022 data ($N=1,188$).
     - **Period 2 (2024–2025 Holdout Evaluation)**: The calibration distribution $F_{\text{pre-2024}}(\cdot)$ was fit on the expanding pre-2024 window (pooled 2022–2023 data, $N=2,369$).
   - Under this sequential protocol, at no point are future evaluation periods or in-period outcomes used to normalize epistemic quantiles or set concentration matching targets. Both protocols yield consistent scientific conclusions: concentration matching and model-risk penalty optimizations confirm zero robust incremental value from epistemic variance ($\lambda^* = 0.0$).

---

## 5. Exact Commands Connecting Experimental Stages

### Re-running Analysis from Prediction Artifacts (Fast, CPU-only)

1. **Incremental Epistemic Value Meta-Model:**
   ```bash
   uv run python scripts/analyze_earnings_epistemic_incremental_value.py \
       --input-dir artifacts/earnings_bayesian_lora_uncertainty_control \
       --output results/earnings_bayesian_lora/incremental_epistemic_value.json
   ```

2. **Downstream Event-Risk Allocation (Raw Sizing):**
   ```bash
   uv run python scripts/analyze_earnings_event_risk_allocation.py \
       --development-dir artifacts/earnings_bayesian_lora_uncertainty_control \
       --holdout-dir artifacts/earnings_bayesian_lora_final_holdout \
       --output results/earnings_event_risk_allocation/summary.json
   ```

3. **Concentration-Matched Allocation Control:**
   ```bash
   uv run python scripts/analyze_earnings_event_risk_allocation_concentration_control.py \
       --development-dir artifacts/earnings_bayesian_lora_uncertainty_control \
       --holdout-dir artifacts/earnings_bayesian_lora_final_holdout \
       --output results/earnings_event_risk_allocation_concentration_control/summary.json
   ```

4. **Model-Risk Penalty Optimization:**
   ```bash
   uv run python scripts/analyze_earnings_model_risk_penalty.py \
       --development-dir artifacts/earnings_bayesian_lora_uncertainty_control \
       --holdout-dir artifacts/earnings_bayesian_lora_final_holdout \
       --output results/earnings_model_risk_penalty/summary.json
   ```

5. **Distribution-Shift Analyses (Requires streaming Hugging Face metadata):**
   ```bash
   # Unseen company shift
   uv run python scripts/analyze_earnings_unseen_company_shift.py \
       --output results/earnings_unseen_company_shift/summary.json

   # Sparse history shift
   uv run python scripts/analyze_earnings_sparse_history_shift.py \
       --output results/earnings_sparse_history_shift/summary.json

   # Underrepresented sector shift
   uv run python scripts/analyze_earnings_underrepresented_sector_shift.py \
       --output results/earnings_underrepresented_sector_shift/summary.json
   ```

6. **Methodological Sensitivity Analysis ($2 \times 2$ Factorial Cross, FALSIFY-02 & FALSIFY-03):**
   ```bash
   uv run python scripts/analyze_meta_model_sensitivity.py \
       --output results/earnings_meta_model_sensitivity/summary.json
   ```
   *Evaluates scaled vs. unscaled features across 2022-only vs. 2022–2023 pooled training on all predefined evaluation populations. See [meta_model_sensitivity_analysis.md](file:///Users/xq/Documents/GitHub/bayesian-lora-portfolio-risk/docs/meta_model_sensitivity_analysis.md).*

---

### End-to-End Model Training Pipeline (Requires GPU/CUDA)

*Note: As noted in `README.md`, existing numerical results are frozen. Running these commands will retrain models on GPU from scratch.*

1. **Gate 0 Baselines:**
   ```bash
   uv run python scripts/run_earnings_gate0_baselines.py \
       --repo-id RudrakshNanavaty/earnings-call-data \
       --train-end 2021-12-31 --test-start 2022-01-01 --test-end 2023-12-31
   ```

2. **Deterministic LoRA Sweep (Stage 1 & 2):**
   ```bash
   uv run python scripts/run_earnings_lora_sweep.py --require-cuda
   uv run python scripts/run_earnings_lora_stage2_repro.py --require-cuda
   ```

3. **Bayesian LoRA Training & Uncertainty Control:**
   ```bash
   uv run python scripts/run_earnings_bayesian_lora_uncertainty_control.py \
       --kl-beta 0.003 --mc-samples 32 --seeds 42 123 777 --require-cuda
   ```

4. **Bayesian LoRA Final Holdout Confirmation:**
   ```bash
   uv run python scripts/run_earnings_bayesian_lora_final_holdout.py \
       --kl-beta 0.003 --mc-samples 32 --seeds 42 123 777 --require-cuda
   ```
