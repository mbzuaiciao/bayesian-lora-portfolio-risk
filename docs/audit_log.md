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

- **Model Checkpoints & Prediction Artifacts (`*.pt`, `adapter_model.safetensors`, `*predictions.csv`):** Large checkpoint files and generated prediction CSVs (> 100MB) are intentionally excluded from git tracking via `.gitignore` to keep repository clone size minimal.
- **Verification Technique:** Committed research summaries under `results/` serve as frozen ground truth. The verification script `scripts/verify_reproducibility.py` provides clean-clone verification by default (evaluating committed JSON schemas and statistical invariants), and optional deep verification (`--deep`) when prediction CSVs are present locally in `artifacts/`.

---

## 5. Remediation of Adversarial Audit Findings (FALSIFY-01 to FALSIFY-06)

Following an adversarial audit of the computational appendix, six specific findings were identified and remediated without altering any frozen historical numerical outcomes:

| Finding ID | Description | Remediation | Verification |
|---|---|---|---|
| **FALSIFY-01** | Prediction CSVs in `artifacts/` are gitignored, but verification script and docs asserted they were committed, breaking clean-clone CI. | Separated clean-clone verification (default) from optional deep recomputation (`--deep` / `--local-artifacts`) in `scripts/verify_reproducibility.py`. Corrected documentation in `docs/reproducibility.md` and `docs/audit_log.md`. CI runs clean-clone mode. | Clean clone simulation passes 100% with no `artifacts/` directory. `--deep` passes when artifacts are present. |
| **FALSIFY-02** | Predictor scaling discrepancy: `StandardScaler` used in `analyze_earnings_epistemic_incremental_value.py`, while unscaled predictors were used elsewhere. | Retained all historical scripts and results untouched. Implemented dedicated sensitivity analysis script `scripts/analyze_meta_model_sensitivity.py` evaluating the full $2 \times 2$ grid (scaled vs unscaled $\times$ 2022-only vs 2022–2023 pooled). Documented in `docs/meta_model_sensitivity_analysis.md`. | Verified headline negative result holds under both scaled and unscaled predictors ($\Delta\text{AUROC} \le 0.0007$ on holdout). |
| **FALSIFY-03** | Training window discrepancy: meta-models fit on 2022-only vs 2022–2023 pooled data across scripts. | Evaluated both training windows across all predefined evaluation populations in `scripts/analyze_meta_model_sensitivity.py` without tuning or choosing a preferred model. | Verified conclusions are invariant to training window choice. |
| **FALSIFY-04** | Calibration protocol discrepancy: initial exploratory allocation used pooled 2022–2023 reference CDF, while later concentration controls used sequential expanding-window calibration. | Explicitly documented the protocol evolution and rationale in Section 4.1 of `docs/reproducibility.md`. Both protocols yield identical qualitative conclusions ($\lambda^* = 0.0$). | Protocol distinction documented clearly without modifying committed results. |
| **FALSIFY-05** | Test coupling: `test_results_and_stability.py` combined pure mathematical unit tests with frozen artifact regression snapshot checks without distinguishing them. | Strictly decoupled mathematical tests into `tests/test_numerical_stability.py` (property checks, precision limits, boundary conditions) and refactored `tests/test_results_and_stability.py` into explicitly documented artifact snapshot invariants. | All 50 pytest tests pass independently. |
| **FALSIFY-06** | Allocation crash on degenerate all-zero raw scores (e.g. $p \equiv 0.5$ everywhere where confidence reliability is zero). | Enhanced `normalize_same_gross`, `compute_allocation_weights`, and `allocation_metrics` in `src/.../uncertainty/evaluation.py` with `fallback_equal=True` parameter returning uniform equal weights of 1.0. Added unit tests in `tests/test_evaluation_metrics.py`. | Degenerate edge cases return uniform weights safely; all non-zero historical calculations remain strictly identical. |

---

## 6. Pyright Static Typing Audit & Warning Classification

A strict static typing audit was performed across all 20 tracked Python modules via `uv run pyright`.

### Audit Summary
- **Errors:** 0
- **Warnings:** 47
- **Information:** 0

### Warning Classification

| Category | Count | Source Files | Description & Root Cause | Correctness / Safety Assessment |
|---|---|---|---|---|
| **Pandas Union-Expansion Friction** | 36 | `src/bayesian_lora_portfolio_risk/data/audit.py` (L50) | Calling `.mean()` on `pd.to_numeric(frame["article_count"])`. The `pandas-stubs` type definition returns a broad 13-type union (`float \| int \| Number \| NAType \| NaTType \| Timestamp \| Timedelta \| ExtensionArray \| Series \| ...`). Pyright checks attribute access on every union member and warns for non-Series branches (`reportAttributeAccessIssue`). | **Benign stub limitation.** At runtime, `pd.to_numeric` on a Series always produces a Series of numeric values. |
| **Pandas Method Overload Resolution** | 5 | `audit.py` (L41), `builder.py` (L153), `hf_news.py` (L67) | Overloaded returns for `.isin().all()`, `sort_values("date")`, and `pd.to_datetime(...)`. The stub signatures define overloaded unions containing `BooleanArray` and `ExtensionArray` (`reportCallIssue`, `reportArgumentType`). | **Benign stub limitation.** Standard idiom in pandas data manipulation. |
| **Pandas Series Indexing Union Types** | 3 | `src/bayesian_lora_portfolio_risk/uncertainty/evaluation.py` (L250-253) | Boolean indexing `pre_sector_counts[mask]` produces a union `Series \| ndarray` in stubs, triggering warnings when accessing `.quantile()`, `.index`, or converting `.index` to a set (`reportAttributeAccessIssue`, `reportReturnType`). | **Benign stub limitation.** `pre_sector_counts` is guaranteed to be a `pd.Series`. |
| **Test Parameter Narrowing Friction** | 3 | `tests/test_leakage_and_invariants.py` (L83, 88, 95) | Passing column slice `df["symbol"]` (typed as `Series \| DataFrame` by `DataFrame.__getitem__` stubs) into helper functions expecting `pd.Series` (`reportArgumentType`). | **Benign test stub typing.** Single column string key always yields a `pd.Series`. |
| **Actionable Correctness Bugs** | 0 | None | No functional or numerical defects identified. | **None.** |

### Policy Compliance
In strict adherence to the release constraints:
- Pyright configuration was **not** weakened.
- No blanket `# type: ignore` annotations were added.
- No unjustified type casts were introduced.
- No numerical or scientific code was modified merely to satisfy external type stubs.


