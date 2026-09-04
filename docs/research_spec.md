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

## Current task family: earnings events

The project uses event-centered financial documents rather than generic daily news. The observation unit is one company earnings episode:

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

Across these variants, text did not add robust predictive information beyond trailing volatility. This rejects the specific mapping from generic daily news bundles to future normalized drawdown for this project.

## Earnings-event result chain

### Experiment 0 — Deterministic text signal

The earnings-event task passed the deterministic-information gate. TF-IDF text and FinBERT features contained meaningful predictive information for the downside-event target, and LoRA adaptation produced a usable text model.

### Experiment 1 — Bayesian uncertainty

Bayesian LoRA produced epistemic uncertainty that was reproducibly associated with model errors across seeds and on a later 2024-2025 holdout. High-uncertainty examples had materially higher error rates, and selective prediction based on epistemic uncertainty improved over using all examples.

However, ordinary deterministic confidence was at least as useful for error detection. In both the 2022-2023 development diagnostics and the later 2024-2025 holdout, epistemic uncertainty did not consistently outperform confidence. Incremental error-prediction models using confidence plus epistemic uncertainty showed essentially no additional value from the epistemic term.

### Experiment 2 — Downstream allocation bridge

An exploratory allocation analysis initially suggested that epistemic-adjusted sizing reduced downside-event exposure relative to confidence-adjusted sizing at equal gross exposure. This pattern was stable across 2023 and 2024-2025.

A concentration-control analysis then showed that the apparent advantage was largely explained by the epistemic rule creating a more concentrated portfolio. When confidence and epistemic allocation rules were matched on effective number of positions, epistemic allocation no longer outperformed confidence and was generally worse on the main downside-risk metrics.

A further model-risk-penalty experiment tested whether epistemic uncertainty added value on top of confidence after concentration matching. The calibration procedure selected an epistemic penalty strength of zero for every seed and evaluation period.

### Experiment 3 — Distribution shift

The final hypothesis was that Bayesian epistemic uncertainty might become incrementally useful under forms of model unfamiliarity.

Three pre-specified subgroup tests were run:

1. **Strict unseen-company shift:** symbols absent from all usable pre-2022 rows. The 2024-2025 subgroup gave a small positive incremental signal, but there were only 35 examples per seed and the 2023 subgroup contained only 12 examples per seed.
2. **Sparse-history company shift:** companies with at most four usable pre-2022 earnings episodes. The result was mixed: 2023 was negative; 2024-2025 showed only a weak positive hint.
3. **Underrepresented-sector shift:** sectors at or below the 25th percentile of pre-2022 episode counts. This produced 133 examples per seed in 2023 and 204 per seed in 2024-2025. In both periods, epistemic uncertainty was weaker than confidence on average, and confidence-plus-epistemic meta-models did not improve robustly over confidence alone.

The larger sector-shift test therefore does not support the distribution-shift rescue hypothesis. Further slicing of the same sample is intentionally stopped to avoid post-hoc subgroup fishing. See `docs/distribution_shift_results.md` for the consolidated evidence.

## Current conclusion

The project supports the following conclusion:

> Bayesian-LoRA epistemic uncertainty is diagnostically meaningful: it identifies harder earnings-event predictions. But in the present task, ordinary deterministic confidence captures essentially all of the practically useful information tested here, including under the evaluated distribution shifts.

This is a substantive negative result rather than a failed experiment. Multiple independent controls point in the same direction:

- epistemic uncertainty correlates with error,
- confidence is at least as strong for error detection,
- incremental error-prediction value beyond confidence is near zero,
- the apparent allocation advantage disappears under concentration matching,
- an explicit epistemic model-risk penalty is optimally set to zero,
- broader distribution-shift tests do not restore a robust incremental advantage.

## Stopping rule for the current study

No additional subgroup thresholds or allocation transformations should be tuned on the already-inspected 2022-2025 sample. The current study is considered complete as a controlled negative-result investigation.

A future continuation should begin as a genuinely new experiment with a separately pre-specified task or dataset, ideally involving an external source, a stronger domain shift, or a different uncertainty approximation.

## Falsification criteria and outcome

The original project was to be reconsidered if:

1. The deterministic NLP signal added no information beyond standard pre-event financial variables.
2. Bayesian uncertainty was not associated with model error.
3. Epistemic uncertainty added no incremental information beyond confidence under the shift conditions where it was hypothesized to matter.
4. Any downstream result disappeared under concentration controls or alternate periods.

The earnings-event task passed criteria 1 and 2 but failed to establish the desired incremental effect in criteria 3 and 4. This is the central empirical result.

## Contribution target

The contribution is a controlled falsification study showing that Bayesian parameter uncertainty can be diagnostically meaningful without providing incremental decision value beyond deterministic confidence. The repository documents both positive diagnostics and the controls that eliminate the stronger economic claim.
