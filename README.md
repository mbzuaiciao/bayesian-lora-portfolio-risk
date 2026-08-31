# Bayesian LoRA Portfolio Risk

Research repository for studying whether epistemic uncertainty from Bayesian LoRA on financial text can improve downstream portfolio risk decisions beyond deterministic NLP signals.

## Milestone 0: uncertainty before portfolios

The first milestone deliberately excludes portfolio optimization. We first test whether Bayesian LoRA produces useful uncertainty on a timestamped financial-text prediction task.

Success requires:

1. Comparable predictive performance to standard LoRA.
2. Better probabilistic calibration.
3. Higher uncertainty on examples the model gets wrong.
4. Higher uncertainty under temporal or domain distribution shift.

Only after these conditions hold do we use the uncertainty signal to adjust classical portfolio risk estimates.

## Initial comparison

- Standard LoRA
- Bayesian LoRA
- Same base model
- Same training data
- Same financial prediction target
- In-distribution and shifted test sets

## Key evaluation metrics

- Accuracy / F1
- Negative log likelihood
- Brier score
- Expected calibration error
- Reliability curves
- Risk-coverage / selective prediction
- Error-vs-uncertainty relationship
- Uncertainty shift between in-distribution and out-of-distribution data

See `docs/research_spec.md` for the full research framing and `experiments/milestone_00/README.md` for the first experiment contract.
