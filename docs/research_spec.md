# Research Specification

## Research question

Can model uncertainty from a Bayesian-adapted financial language model improve downstream financial decisions beyond the information contained in its deterministic NLP prediction alone?

The project compares:

1. Classical financial estimation.
2. Classical estimation augmented with a deterministic financial-NLP signal.
3. Classical estimation augmented with a Bayesian-NLP signal and model uncertainty.

The central scientific object is not portfolio return. It is whether model uncertainty is decision-relevant.

## Two-stream view

### Classical stream

Market data produces estimates of expected return and covariance:

\[
\hat\mu_t, \hat\Sigma_t.
\]

### Text stream

A substantive, point-in-time financial document associated with asset i at event time t is mapped to a risk prediction and, under Bayesian LoRA, model uncertainty:

\[
D_{i,t} \rightarrow (\bar r_{i,t}, u_{i,t}).
\]

The long-run project studies how these can be combined before portfolio allocation.

## Central hypothesis

Bayesian-LoRA uncertainty should identify cases in which a financial-text prediction is less trustworthy, particularly under temporal or domain distribution shift. If uncertainty is informative, conditioning downstream decisions on it should improve robustness relative to using the point prediction alone.

## Current Milestone 0 task family: earnings events

Milestone 0 now uses event-centered financial documents rather than generic daily news. The preferred first observation unit is one company earnings episode:

\[
D_{i,q} = \text{earnings-call transcript and/or earnings release for company i, quarter q}.
\]

Candidate downstream targets include:

- abnormal downside return after the earnings event,
- volatility increase after the earnings event,
- maximum post-event drawdown over a fixed horizon.

The first implementation should choose one target only and should normalize the target using information available before the event where appropriate.

The core scientific question is:

> Does uncertainty in a language model's interpretation of a substantive financial disclosure tell us when its financial-risk signal should not be trusted?

## FNSPID pilot: falsified operationalization

The initial pilot used FNSPID daily news aggregated to asset-day observations and predicted volatility-normalized future drawdown. This operationalization was rejected before LoRA training because the deterministic text signal failed the incremental-information gate.

Diagnostics included:

- ticker-only structured baselines,
- volatility normalization to remove stock-identity shortcuts,
- relevance filtering of ticker-associated news,
- replacement of headline-like inputs with substantially longer article text,
- horizon sweeps at 3, 5, 10, and 20 trading days,
- single-window FinBERT embeddings,
- chunked FinBERT document embeddings.

Across these variants, text did not add robust predictive information beyond trailing volatility. This does not imply that financial NLP is uninformative in general; it rejects the specific mapping from generic daily news bundles to future normalized drawdown for this project.

The FNSPID pilot remains in the repository as a documented negative result and falsification example.

## Experimental hierarchy

### Experiment 0 — Does the deterministic text signal add information?

Before LoRA or Bayesian modeling, establish that the selected earnings document task contains incremental information beyond simple pre-event structured variables such as recent volatility and returns.

This is now a hard gate.

### Experiment 1 — Is the uncertainty useful?

Compare standard LoRA and Bayesian LoRA on the same task, data, and base model. Evaluate predictive quality, calibration, error-uncertainty association, and behavior under distribution shift.

### Experiment 2 — Does uncertainty improve decisions?

Only after the deterministic signal and uncertainty gates succeed, test uncertainty-aware portfolio risk adjustment.

## Critical ablation for the final portfolio stage

The key comparison will be Bayesian posterior mean only versus Bayesian posterior mean plus uncertainty. This separates any benefit of Bayesian training from the incremental value of uncertainty itself.

## Falsification criteria

The project should be reconsidered or the task should be redesigned if:

1. The deterministic NLP signal adds no information beyond standard pre-event financial variables.
2. Bayesian uncertainty is not associated with model error.
3. Uncertainty does not respond meaningfully to temporal/domain shift.
4. Any downstream result disappears under realistic costs or alternate time periods.

## Contribution target

A strong result would show that model uncertainty from parameter-efficient adaptation identifies unreliable financial-NLP signals under shift and can be translated into more robust downstream financial decisions.
