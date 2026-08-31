# Milestone 0 — Uncertainty Before Portfolios

## Objective

Establish whether Bayesian LoRA produces useful epistemic uncertainty on a timestamped financial-text prediction task before any portfolio optimizer is introduced.

## Experimental contract

Hold constant:

- base language model,
- tokenizer,
- training examples,
- prediction target,
- train/validation/test split,
- evaluation code.

Vary only the adaptation method:

1. Standard LoRA.
2. Bayesian LoRA.

## Required datasets

Construct three splits:

- Train: earlier period.
- ID test: same broad regime/domain as training.
- Shift test: later period or deliberately shifted document/domain regime.

All documents must respect point-in-time availability. No future text or future-derived features may enter an earlier example.

## Required metrics

### Predictive performance

- Accuracy
- F1
- AUROC when appropriate

### Probabilistic quality

- Negative log likelihood
- Brier score
- Expected calibration error
- Reliability curve data

### Uncertainty usefulness

- Mean uncertainty on correct predictions
- Mean uncertainty on incorrect predictions
- Error rate by uncertainty quantile
- Risk-coverage curve / selective prediction

### Distribution shift

- Change in predictive performance from ID to shift
- Change in calibration from ID to shift
- Change in mean uncertainty from ID to shift

## Go / no-go criteria

Proceed to portfolio integration only if the Bayesian model shows most of the following:

1. Predictive performance roughly comparable to deterministic LoRA.
2. Better calibration than deterministic LoRA.
3. Higher uncertainty on incorrect than correct predictions.
4. Monotonic or near-monotonic increase in error rate across uncertainty quantiles.
5. Meaningful increase in uncertainty on the shifted test set.

If uncertainty fails these tests, diagnose the Bayesian approximation, target, or shift construction before adding downstream complexity.

## Non-goals

Milestone 0 does not include:

- portfolio optimization,
- backtested trading strategies,
- transaction costs,
- expected-return forecasting,
- multi-agent systems,
- elaborate financial dashboards.

The milestone asks one question only: **does the uncertainty mean anything?**
