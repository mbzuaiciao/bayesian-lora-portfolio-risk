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

Bayesian-LoRA uncertainty may identify cases in which a financial-text prediction is less trustworthy, particularly under temporal or domain distribution shift. The key question is not merely whether uncertainty correlates with error, but whether it adds decision-relevant information beyond ordinary model confidence.

## Current Milestone 0 task family: earnings events

Milestone 0 uses event-centered financial documents rather than generic daily news. The first observation unit is one company earnings episode:

\[
D_{i,q} = \text{earnings-call transcript and/or earnings release for company i, quarter q}.
\]

The implemented target is a downside earnings-event label defined from the subsequent one-day market move, with the threshold fixed from the pre-2022 training period.

The core scientific question is:

> Does uncertainty in a language model's interpretation of a substantive financial disclosure tell us when its financial-risk signal should not be trusted, beyond what ordinary confidence already tells us?

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

## Earnings-event result chain

### Experiment 0 — Deterministic text signal

The earnings-event task passed the deterministic-information gate. TF-IDF text and FinBERT features contained meaningful predictive information for the downside-event target, and LoRA adaptation produced a usable text model.

### Experiment 1 — Bayesian uncertainty

Bayesian LoRA produced epistemic uncertainty that was reproducibly associated with model errors across seeds and on a later 2024-2025 holdout. High-uncertainty examples had materially higher error rates, and selective prediction based on epistemic uncertainty improved over using all examples.

However, ordinary deterministic confidence was at least as useful for error detection. In both the 2022-2023 development diagnostics and the later 2024-2025 holdout, epistemic uncertainty did not consistently outperform confidence. Incremental error-prediction models using confidence plus epistemic uncertainty showed essentially no additional value from the epistemic term.

### Experiment 2 — Downstream allocation bridge

An exploratory allocation analysis initially suggested that epistemic-adjusted sizing reduced downside-event exposure relative to confidence-adjusted sizing at equal gross exposure. This pattern was stable across 2023 and 2024-2025.

A concentration-control analysis then showed that the apparent advantage was largely explained by the epistemic rule creating a more concentrated portfolio. When confidence and epistemic allocation rules were matched on effective number of positions, epistemic allocation no longer outperformed confidence and was generally worse on the main downside-risk metrics.

A further model-risk-penalty experiment tested whether epistemic uncertainty added value on top of confidence after concentration matching. The calibration procedure selected an epistemic penalty strength of zero for every seed and evaluation period. In this setup, the optimal policy was therefore to ignore the additional epistemic penalty once confidence was already included.

## Current conclusion

The project currently supports the following conclusion:

> Bayesian epistemic uncertainty is diagnostically meaningful: it identifies harder earnings-event predictions. But in the present task, it does not provide clear incremental decision value beyond ordinary model confidence.

This is a substantive negative result rather than a failed experiment. Multiple independent controls point to the same conclusion:

- epistemic uncertainty correlates with error,
- confidence is at least as strong for error detection,
- the apparent allocation advantage disappears under concentration matching,
- an explicit epistemic model-risk penalty is optimally set to zero.

## Next hypothesis: distribution shift

The most natural remaining hypothesis is that epistemic uncertainty may become more useful under genuine distribution shift rather than ordinary temporal continuation.

Candidate tests include:

- unseen companies or sectors,
- regime-shift periods,
- sparse-data subsets,
- temporally distant evaluation windows,
- document-style or source shifts.

Any next experiment should be pre-specified before inspecting its outcomes and should evaluate whether epistemic uncertainty provides incremental information beyond confidence under shift.

## Falsification criteria

The project should be reconsidered or the task should be redesigned if:

1. The deterministic NLP signal adds no information beyond standard pre-event financial variables.
2. Bayesian uncertainty is not associated with model error.
3. Epistemic uncertainty adds no incremental information beyond confidence under the shift conditions where it is hypothesized to matter.
4. Any downstream result disappears under concentration controls, realistic costs, or alternate time periods.

## Contribution target

A strong positive result would show that Bayesian parameter uncertainty identifies unreliable financial-NLP signals under genuine shift and improves downstream decisions beyond ordinary confidence. A strong negative result is also informative if carefully controlled experiments show that confidence already captures the decision-relevant information that Bayesian uncertainty was intended to add.
