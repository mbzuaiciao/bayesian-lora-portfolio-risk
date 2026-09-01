# Milestone 0 — Uncertainty Before Portfolios

## Objective

Establish whether Bayesian LoRA produces useful model uncertainty on a timestamped, event-centered financial-text prediction task before any portfolio optimizer is introduced.

## Current task family: earnings episodes

The preferred observation unit is one company earnings event:

\[
D_{i,q} = \text{earnings-call transcript and/or earnings release for company i, quarter q}.
\]

The first target should be one post-event risk outcome, for example:

- abnormal downside return,
- maximum drawdown over a fixed post-event window, or
- realized-volatility increase.

The target must begin strictly after the disclosure event. Pre-event structured variables may be used as baselines, but no post-event information may leak into the feature set.

## Gate 0 — deterministic text must add information

Before standard LoRA or Bayesian LoRA training, compare:

1. constant baseline,
2. simple pre-event structured baseline,
3. frozen semantic text baseline,
4. semantic text + structured baseline.

Proceed only if the text representation adds credible incremental information beyond the structured baseline on a chronological test period.

## FNSPID pilot — no-go result

The initial task used FNSPID daily financial news and future volatility-normalized drawdown. It was intentionally stopped before LoRA training after failing Gate 0.

The falsification sequence included:

1. **Stock-identity audit.** The absolute 20-day 10% drawdown label had a strong ticker shortcut. Volatility normalization reduced ticker-only AUROC from roughly 0.74 to roughly 0.54.
2. **Information-set audit.** Many ticker-associated observations were generic or irrelevant headlines. A relevance filter rejected obvious mismatches and increased median asset-day text length from roughly 164 characters to roughly 9,260 characters by retaining substantive article text.
3. **Semantic baseline.** Relevance-filtered single-window FinBERT text still failed to improve on trailing volatility.
4. **Horizon sweep.** Three-, five-, ten-, and twenty-day normalized targets all showed non-positive incremental value from text relative to volatility.
5. **Longer-context audit.** Four-chunk FinBERT improved the combined model somewhat but remained substantially below volatility alone (approximately 0.616 versus 0.696 AUROC on the 2022 diagnostic split), with worse probabilistic metrics.

Conclusion:

> Generic daily FNSPID news bundles do not provide robust incremental information over recent volatility for predicting volatility-normalized future drawdown in this experimental setup.

This is a rejected operationalization, not a claim that financial language is generally uninformative.

The FNSPID code and artifacts are retained because they document a useful negative result and demonstrate the project's falsification discipline.

## Experimental contract after Gate 0

Once an earnings-event task passes the deterministic text gate, hold constant:

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

Construct chronological splits:

- Train: earlier events.
- ID validation/test: later observations still near the training distribution.
- Shift test: a strictly later period or materially different market regime.

All documents and structured variables must respect point-in-time availability.

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

Proceed to portfolio integration only if:

1. the deterministic earnings-text signal adds information beyond simple pre-event structured variables,
2. Bayesian-LoRA predictive performance is roughly comparable to deterministic LoRA,
3. Bayesian-LoRA calibration is better or at least competitive,
4. uncertainty is higher on errors than on correct predictions,
5. error rate generally increases with uncertainty,
6. uncertainty responds meaningfully to the shifted test set.

## Non-goals

Milestone 0 does not include:

- portfolio optimization,
- backtested trading strategies,
- transaction costs,
- expected-return forecasting,
- multi-agent systems,
- elaborate financial dashboards.

The milestone asks two ordered questions only:

1. **Does the financial document contain incremental predictive information?**
2. **If so, does Bayesian model uncertainty mean anything?**
