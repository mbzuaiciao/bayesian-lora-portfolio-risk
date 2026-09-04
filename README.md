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
- `results/` — small reproducible result summaries.

Large prediction files and checkpoints remain local artifacts.
