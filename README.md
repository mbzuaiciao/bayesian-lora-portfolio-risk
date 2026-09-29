# Bayesian LoRA Portfolio Risk

Research repository studying whether epistemic uncertainty from Bayesian LoRA on financial text improves downstream financial-risk decisions beyond deterministic model confidence.

## Current research result

The earnings-event study now supports a fairly clear negative result:

1. The deterministic text signal is useful.
2. Bayesian epistemic uncertainty is reproducibly associated with harder / more error-prone examples.
3. Ordinary classifier confidence is at least as useful for error detection.
4. An apparent allocation advantage from epistemic sizing disappears after matching portfolio concentration.
5. When epistemic uncertainty is added as a separate model-risk penalty on top of confidence, the selected penalty strength is zero across all tested seeds and time periods.
6. Targeted distribution-shift tests on unseen companies, sparse-history companies, and underrepresented sectors do not show robust incremental value beyond confidence.

The current conclusion is therefore:

> Bayesian-LoRA epistemic uncertainty is diagnostically meaningful, but in this financial earnings-event task ordinary deterministic confidence captures essentially all of the practically useful information tested here.

This is treated as a controlled negative result rather than as evidence that Bayesian uncertainty is never useful.

## Research Artifact Status

> [!IMPORTANT]
> **Study Status: Complete, Audited, and Frozen**
>
> 1. **Completed & Closed Study:** The 2022–2025 research study is finalized and closed. Under the study's pre-specified stopping rule, further subgroup searches, allocation transformations, or parameter tuning on the inspected 2022–2025 sample are intentionally out of scope.
> 2. **Controlled Negative Result:** The headline conclusion is a controlled negative result: while Bayesian-LoRA epistemic uncertainty contains a real diagnostic signal correlated with prediction difficulty, it provides no robust incremental decision or error-detection value beyond ordinary deterministic classifier confidence ($1 - 2|p - 0.5|$) under the tested protocols, concentration-matched portfolios, model-risk penalty optimizations, or distribution-shift subgroups.
> 3. **Adversarial Audit & Remediation:** The computational appendix underwent an adversarial scientific audit. All findings (**FALSIFY-01** through **FALSIFY-06**) were remediated or explicitly documented in [docs/audit_log.md](docs/audit_log.md) without altering frozen historical results or committed numerical outputs.
> 4. **Dual Verification Paths:**
>    - **Clean-Clone Verification (Default):** Runs via `uv run python scripts/verify_reproducibility.py` and requires zero untracked prediction CSVs or checkpoints, passing reliably in clean-clone CI environments.
>    - **Deep Local Verification:** When local prediction artifacts are present in `artifacts/`, `uv run python scripts/verify_reproducibility.py --deep` recomputes AUROCs and portfolio allocations directly from raw Monte Carlo prediction files.
> 5. **Methodological Sensitivity Analysis:** The discovered meta-model discrepancies (predictor scaling and 2022-only vs. 2022–2023 training windows) were tested under a complete $2 \times 2$ factorial cross in [docs/meta_model_sensitivity_analysis.md](docs/meta_model_sensitivity_analysis.md), confirming that the headline negative result is completely invariant to these choices ($\Delta\text{AUROC} \le 0.0007$ on holdout).
>
> For full details, see the [Research Specification](docs/research_spec.md), [Claims-to-Code Mapping](docs/claims_to_code_mapping.md), [Reproducibility Guide](docs/reproducibility.md), [Audit Log](docs/audit_log.md), and [Sensitivity Analysis](docs/meta_model_sensitivity_analysis.md).

## Experimental progression

### FNSPID pilot

The initial asset-day news formulation was rejected because text did not add robust information beyond structured market features after controlling for ticker and volatility effects.

### Earnings-event task

The project then moved to long-form earnings documents with a fixed downside-event target. This task passed the deterministic information gate and supported standard-LoRA and Bayesian-LoRA experiments.

### Bayesian uncertainty tests

The analysis includes:

- posterior predictive evaluation,
- calibration metrics,
- error-vs-uncertainty diagnostics,
- selective-risk analysis,
- multi-seed replication,
- confidence controls,
- incremental-value tests,
- later-period holdout evaluation.

### Economic decision tests

Downstream exploratory analyses include:

- equal-gross event-risk allocation,
- rolling temporal robustness,
- concentration matching,
- incremental epistemic model-risk penalties.

The concentration and model-risk controls are important falsification tests: they show that the apparent economic advantage of epistemic sizing is not robust once ordinary confidence and portfolio concentration are properly controlled.

### Distribution-shift tests

The final robustness stage tested whether epistemic uncertainty becomes more useful under forms of model unfamiliarity.

- **Strict unseen companies:** a small positive 2024-2025 hint, but only 35 examples per seed and mixed earlier-period evidence.
- **Sparse-history companies:** mixed results; the positive hint did not remain stable when the subgroup was broadened.
- **Underrepresented sectors:** the largest shift subgroup, with 133 examples per seed in 2023 and 204 per seed in 2024-2025, produced a clear negative result. Confidence remained stronger overall, and adding epistemic uncertainty did not improve the error meta-model consistently.

Further subgroup search is intentionally stopped here to avoid post-hoc subgroup fishing.

## Interpretation

The result is narrower and more useful than either “Bayesian uncertainty works” or “Bayesian uncertainty is useless.”

In this setup, Bayesian parameter uncertainty contains a real diagnostic signal, but that signal appears largely redundant with ordinary predictive confidence for the downstream decisions and distribution shifts tested.

A genuinely new follow-up should therefore use a separately pre-specified task or dataset with a stronger external shift, rather than continuing to slice the existing sample.

## Repository guide

- `docs/research_spec.md` — full research framing and experiment chain.
- `docs/distribution_shift_results.md` — consolidated distribution-shift evidence and stopping rationale.
- `docs/claims_to_code_mapping.md` — claim-by-claim mapping between research spec and supporting artifacts.
- `docs/reproducibility.md` — environment setup, experiment provenance, and exact commands connecting experimental stages.
- `docs/audit_log.md` — adversarial audit remediation log (FALSIFY-01 to FALSIFY-06).
- `docs/meta_model_sensitivity_analysis.md` — 2x2 sensitivity analysis on meta-model specifications.
- `results/` — small reproducible result summaries.
- `scripts/verify_reproducibility.py` — lightweight verification command to audit repository consistency without retraining.

Large prediction files and checkpoints remain local artifacts.
