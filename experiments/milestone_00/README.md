# Milestone 0 — Uncertainty Before Portfolios

## Objective

Establish whether Bayesian LoRA produces useful epistemic uncertainty on a timestamped financial-text prediction task before any portfolio optimizer is introduced.

## Locked first task

Use FNSPID financial news and stock-price histories to predict whether an asset experiences a material drawdown during the next 20 trading days.

For asset i on date t:

\[
DD_{i,t}^{(20)} = \min_{1 \le k \le 20}\left(\frac{P_{i,t+k}}{P_{i,t}} - 1\right)
\]

and

\[
y_{i,t} = \mathbf{1}[DD_{i,t}^{(20)} \le -\delta].
\]

Start with \(\delta=10\%\). Any threshold adjustment must be based only on training-period class balance and documented before test evaluation.

See `docs/dataset_and_task.md` for the full data contract.

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

## Required splits

Construct three chronological splits:

- Train: earlier period.
- ID validation/test: later observations still near the training distribution.
- Shift test: a strictly later period or materially different market regime.

All documents must respect point-in-time availability. No future text or future-derived features may enter an earlier example. Ambiguous same-day article timing is handled conservatively by assigning the text to the next trading-day information set.

## Required metrics

### Predictive performance

- Accuracy
- F1
- AUROC

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
