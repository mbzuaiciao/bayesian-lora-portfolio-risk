# Bayesian LoRA Portfolio Risk — One-Page Code Walkthrough

## Research question

Does Bayesian-LoRA epistemic uncertainty add useful information beyond ordinary model confidence, and can that improve downstream portfolio-risk decisions?

## End-to-end flow

`earnings transcript -> FinBERT -> Bayesian LoRA on attention query/value projections -> posterior samples -> predictive mean + epistemic variance -> error/confidence controls -> portfolio-risk experiments`

## 1. Bayesian LoRA implementation

**File:** `src/bayesian_lora_portfolio_risk/uncertainty/bayesian_lora.py`

What to explain verbally:

- Standard LoRA learns fixed low-rank matrices.
- Here the LoRA parameters are treated probabilistically: each parameter has a learned mean and scale.
- During posterior sampling, the code draws LoRA weights from those learned distributions using reparameterization.
- The training objective is prediction loss plus a KL penalty that regularizes the approximate posterior toward the prior.
- Bayesian adaptation is injected into FinBERT attention **query** and **value** projections; the task classifier remains deterministic.

Simple verbal summary:

> Instead of learning one LoRA solution, I learn a distribution over plausible LoRA solutions. Repeated forward passes with sampled LoRA weights give a distribution of predictions.

## 2. Main Bayesian training and inference

**File:** `scripts/run_earnings_bayesian_lora.py`

What happens:

1. Stream earnings-call transcripts and the one-day post-earnings move.
2. Split chronologically; the downside label threshold is determined from pre-test training data.
3. Chunk long transcripts and aggregate chunk predictions at the document level.
4. Train Bayesian LoRA with prediction loss + scaled KL.
5. At inference, run multiple posterior samples.
6. For each document compute:
   - **posterior mean** = average predicted downside probability;
   - **epistemic variance** = variance of that probability across posterior samples.

The selected setup after validation experiments was approximately:

- LoRA rank: 16
- KL beta: 0.003
- learning rate: 2e-4
- posterior samples for final diagnostics: 32
- seeds: 42, 123, 777

## 3. Result that originally looked promising

The uncertainty ordering was reproducible: high-epistemic-uncertainty predictions were more error-prone than low-uncertainty predictions, and selective prediction using the least-uncertain cases reduced error.

The important interpretation is:

> The Bayesian uncertainty signal is not random; it identifies harder examples.

## 4. Critical control: ordinary confidence

**Files:**

- `scripts/run_earnings_bayesian_lora_uncertainty_control.py`
- `scripts/analyze_earnings_epistemic_incremental_value.py`

Ordinary confidence uncertainty is based on how close the model probability is to 0.5.

The key control asks:

> Does epistemic variance tell us something about errors that ordinary confidence does not already tell us?

Result:

- epistemic uncertainty predicts errors;
- ordinary confidence is generally as good or slightly better;
- adding epistemic uncertainty to a confidence-only error model gives little or no robust incremental improvement.

## 5. Portfolio / uncertainty-fusion experiments

**Files:**

- `scripts/analyze_earnings_event_risk_allocation.py`
- `scripts/analyze_earnings_event_risk_allocation_concentration_control.py`
- `scripts/analyze_earnings_model_risk_penalty.py`

Initial uncertainty-adjusted sizing looked encouraging, but the epistemic rule also produced a more concentrated portfolio.

After matching concentration, the apparent advantage disappeared or reversed. A separate model-risk penalty experiment selected an epistemic penalty strength of zero across all tested seeds and periods.

Interpretation:

> Once ordinary confidence and portfolio concentration are controlled, the current experiments do not show clear incremental economic value from epistemic uncertainty.

## 6. Distribution-shift checks

**Doc:** `docs/distribution_shift_results.md`

Tested:

- unseen companies,
- sparse-history companies,
- underrepresented sectors.

Tiny unseen/sparse subsets showed a few positive hints, but the broader sector-shift test was negative. I stopped further subgroup tuning on the inspected sample to avoid post-hoc fishing.

## Current conclusion

> Bayesian-LoRA epistemic uncertainty is diagnostically meaningful, but in this earnings-event task it has not shown clear incremental decision value beyond ordinary deterministic confidence.

## Questions to discuss with Praneeth

1. Is the current Bayesian-LoRA posterior parameterization the right one for this application?
2. Is predictive variance across posterior samples the right epistemic quantity to fuse into portfolio risk?
3. Should uncertainty enter allocation as shrinkage, a separate model-risk term, covariance inflation, or something else?
4. Would a task with stronger distribution shift by construction be a better testbed?
5. Which parts of the code or experimental design should be simplified before the next stage?
