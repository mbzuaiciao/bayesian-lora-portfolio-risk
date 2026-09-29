# Computational Research Audit Log

This document records the comprehensive audit conducted on this repository as a completed computational research study whose headline finding is a controlled negative result.

---

## 1. Audit Scope & Methodology

- **Constraint:** The scientific conclusions, temporal splits, stopping rule, existing result artifacts, hyperparameter choices, subgroup definitions, and evaluation protocol are **frozen**. No new tuning, subgroup searches, or empirical parameter shifts were performed.
- **Objectives:**
  1. Map every claim in `README.md` and `docs/research_spec.md` to supporting code and committed artifacts.
  2. Identify conclusion-critical code with inadequate test coverage and implement robust synthetic-fixture test suites.
  3. Verify mathematical/statistical invariants: data leakage, label construction, Bayesian uncertainty calculations, confidence controls, concentration matching ($N_{\text{eff}}$), model-risk penalties, subgroup definitions, seed aggregation, and numerical stability.
  4. Refactor pure analysis logic into importable modules without altering behavior.
  5. Build a lightweight, CPU-friendly verification command establishing repository consistency without retraining.
  6. Add CI for unit tests, Ruff, and Pyright.
  7. Document environment setup, artifact provenance, and exact stage-by-stage reproduction commands.

---

## 2. Issues Discovered & Fixes Made

| Issue # | Component | Description of Issue | Resolution / Fix |
|---|---|---|---|
| **ISS-01** | `src/.../uncertainty/bayesian_lora.py` | Line length violations (> 100 chars) in `VariationalGaussianParameter.__init__` and `inject_bert_attention_bayesian_lora`. | Wrapped parameter signatures and boolean conditions across multiple lines. |
| **ISS-02** | `src/.../data/fnspid.py` | Duplicate dictionary key literal `"Stock_symbol"` in `_NEWS_ALIASES`. | Removed duplicate key literal. |
| **ISS-03** | `src/.../data/hf_news.py` | Import block un-sorted (`from collections.abc import Iterable` before `import math`). | Sorted imports according to PEP 8 / Ruff standards. |
| **ISS-04** | Conclusion-Critical Decision Logic | Core analysis functions for confidence uncertainty, empirical CDF percentiles, effective position count ($N_{\text{eff}}$), concentration power matching ($\gamma$-bisection), allocation downside metrics, and subgroup partitioning were embedded inside monolithic script files without direct unit tests. | Extracted pure analysis functions into `src/bayesian_lora_portfolio_risk/uncertainty/evaluation.py` while preserving existing script behavior exactly. |
| **ISS-05** | Test Suite Coverage | Existing tests only covered basic pilot ingestion and 4 simple scalar uncertainty metrics. Conclusion-critical calculations (Bayesian LoRA sampling, KL divergence, confidence controls, concentration matching, model-risk penalty, and subgroup shift logic) had 0 unit test coverage. | Added 19 comprehensive tests across 4 new test modules (`test_evaluation_metrics.py`, `test_bayesian_lora.py`, `test_leakage_and_invariants.py`, `test_results_and_stability.py`). Total passing tests increased from 23 to 42. |
| **ISS-06** | Continuous Integration | No GitHub Actions workflow existed to enforce test execution, linting, or type checking on push/PR. | Added `.github/workflows/ci.yml` running unit tests, Ruff, Pyright, and the consistency verifier on Python 3.12. |
| **ISS-07** | Lint & Type Checker Config | `pyproject.toml` lacked per-file ignore rules for legacy executable scripts and Pyright configuration for the project packages. | Updated `pyproject.toml` with `per-file-ignores` for scripts and standard Pyright configuration. |

---

## 3. Invariants Verified

1. **Temporal & Data Leakage Invariants:**
   - Pre-2022 train split ($\le 2021-12-31$), 2022–2023 development diagnostic ($2022-01-01$ to $2023-12-31$), and 2024–2025 final holdout ($2024-01-01$ to $2025-12-31$) are strictly non-overlapping and disjoint.
   - Downside threshold $\tau$ is strictly estimated from pre-test data and held frozen during all evaluation periods.
   - Subgroup definitions (unseen companies, sparse history $\le 4$ episodes, underrepresented sectors $\le 25\text{th}$ percentile) are derived solely from pre-2022 history without using post-2022 labels or return outcomes.
2. **Bayesian Uncertainty & Confidence Controls:**
   - Bayesian epistemic variance $\sigma^2_{\text{epi}}$ is strictly positive and non-zero across posterior MC samples.
   - Confidence uncertainty $u_{\text{conf}} = 1 - 2|p - 0.5|$ is symmetric around $0.5$ and bounded in $[0, 1]$.
   - Out-of-sample error detection AUROC and selective risk curves show confidence is consistently superior or equivalent to epistemic uncertainty across all three seeds.
3. **Concentration Matching & Model-Risk Penalty:**
   - Power transformation $w^\gamma$ strictly preserves ranking while monotonically adjusting $N_{\text{eff}} = 1 / \sum w_i^2$.
   - Bisection convergence achieves target $N_{\text{eff}}$ without utilizing forward period returns.
   - In both 2023 and 2024–2025 evaluation periods, model-risk penalty calibration selects $\lambda^* = 0.0$ across 100% of seeds.
4. **Numerical Stability & Serialization:**
   - Verified handling of boundary conditions: $p \in \{0.0, 1.0, 10^{-15}\}$, near-zero weights, zero-movement return vectors, and flat uniform weight vectors.
   - All 27 committed result summaries and prediction CSVs validate against schema requirements.

---

## 4. Verification of Items Requiring Local/Absent Checkpoints

- **Model Checkpoints (`*.pt`, `adapter_model.safetensors`):** Large checkpoint files and raw token cache files (> 100MB) are intentionally excluded from git tracking to keep repository clone size minimal.
- **Verification Technique:** Because the completed Monte Carlo prediction files (`beta_0p003_seed_{42,123,777}_mc32_predictions.csv`) for both development and holdout are committed under `artifacts/`, all downstream statistical, economic, and distribution-shift results were verified directly against these prediction arrays. The lightweight verification script `scripts/verify_reproducibility.py` validates this complete calculation chain in under 5 seconds on CPU.
