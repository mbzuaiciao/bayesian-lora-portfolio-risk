# Bayesian LoRA Portfolio Risk

Research repository studying whether epistemic uncertainty from Bayesian LoRA on financial text improves downstream financial-risk decisions beyond deterministic model confidence.

## Current research result

The project now has a fairly clear empirical result on the earnings-event task:

1. The deterministic text signal is useful.
2. Bayesian epistemic uncertainty is reproducibly higher on harder / more error-prone examples.
3. Ordinary classifier confidence is at least as useful for error detection.
4. An apparent allocation advantage from epistemic sizing disappears after matching portfolio concentration.
5. When epistemic uncertainty is added as a separate model-risk penalty on top of confidence, the selected penalty strength is zero across all tested seeds and time periods.

The current conclusion is therefore:

> Epistemic uncertainty is diagnostically meaningful, but in this setup it does not provide clear incremental decision value beyond ordinary confidence.

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

## Next question

The next hypothesis is whether epistemic uncertainty becomes incrementally useful under genuine distribution shift, for example:

- unseen companies or sectors,
- regime shifts,
- sparse-data subsets,
- temporally distant evaluation windows,
- document-source or style shifts.

Any such experiment should test epistemic uncertainty against ordinary confidence directly and should be pre-specified before inspecting outcomes.

## Repository guide

See `docs/research_spec.md` for the full research framing and current conclusions. Small reproducible result summaries are tracked under `results/`; large prediction files and checkpoints remain local artifacts.
