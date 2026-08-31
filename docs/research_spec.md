# Research Specification

## Research question

Can epistemic uncertainty from a Bayesian-adapted financial language model improve portfolio decisions beyond the information contained in its deterministic NLP prediction alone?

The project compares:

1. Classical portfolio estimation.
2. Classical estimation augmented with a deterministic financial-NLP signal.
3. Classical estimation augmented with a Bayesian-NLP signal and epistemic uncertainty.

The central scientific object is not portfolio return. It is whether model uncertainty is decision-relevant.

## Two-stream view

### Classical stream

Market data produces estimates of expected return and covariance:

\[
\hat\mu_t, \hat\Sigma_t.
\]

### Text stream

Timestamped documents associated with asset i at time t are mapped to a risk prediction and, under Bayesian LoRA, uncertainty:

\[
D_{i,t} \rightarrow (\bar r_{i,t}, u_{i,t}).
\]

The long-run project studies how these can be combined before portfolio allocation.

## Central hypothesis

Bayesian-LoRA uncertainty should identify cases in which a financial-text prediction is less trustworthy, particularly under temporal or domain distribution shift. If uncertainty is informative, conditioning downstream decisions on it should improve robustness relative to using the point prediction alone.

## Initial NLP task

Use a timestamped supervised financial prediction problem with an economically interpretable target. Candidate first targets include:

- future drawdown threshold,
- future volatility increase,
- earnings deterioration,
- credit-spread widening,
- rating downgrade.

The first implementation should choose one target only.

## Experimental hierarchy

### Experiment 1 — Is the uncertainty useful?

Compare standard LoRA and Bayesian LoRA on the same task, data, and base model. Evaluate predictive quality, calibration, error-uncertainty association, and behavior under distribution shift.

### Experiment 2 — Is the text signal incrementally useful?

Test whether the deterministic NLP signal contains information beyond classical structured market variables.

### Experiment 3 — Does uncertainty improve decisions?

Only after Experiments 1 and 2 succeed, test uncertainty-aware portfolio risk adjustment.

## Critical ablation for the final portfolio stage

The key comparison will be Bayesian posterior mean only versus Bayesian posterior mean plus uncertainty. This separates any benefit of Bayesian training from the incremental value of uncertainty itself.

## Falsification criteria

The project should be reconsidered if:

1. Bayesian uncertainty is not associated with model error.
2. Uncertainty does not respond meaningfully to temporal/domain shift.
3. The NLP signal adds no information beyond standard financial variables.
4. Any downstream result disappears under realistic costs or alternate time periods.

## Contribution target

A strong result would show that epistemic uncertainty from parameter-efficient adaptation identifies unreliable financial-NLP signals under shift and can be translated into more robust downstream financial decisions.
