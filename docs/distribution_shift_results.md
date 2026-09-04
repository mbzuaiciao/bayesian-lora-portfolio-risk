# Distribution-shift results

## Question

After the aggregate earnings-event experiments found that Bayesian-LoRA epistemic uncertainty was diagnostically meaningful but did not add clear value beyond ordinary confidence, the project tested a narrower hypothesis:

> Does epistemic uncertainty become incrementally useful under distribution shift?

The comparison remained strict throughout: higher raw uncertainty on shifted examples was not sufficient. The relevant criterion was whether epistemic uncertainty improved error prediction **beyond deterministic confidence**.

These are targeted robustness analyses rather than pristine new holdouts because the underlying 2022-2025 dates had already been inspected in earlier aggregate experiments. The subgroup definitions themselves were fixed before inspecting their subgroup outcomes.

## 1. Strict unseen-company shift

### Definition

A company is unseen if its symbol does not occur in any usable pre-2022 row.

### Sample size

- 2023: 12 unseen-company examples per seed.
- 2024-2025: 35 unseen-company examples per seed.

### Result

The 2024-2025 unseen-company subgroup showed a small positive incremental signal:

- mean confidence+epistemic minus confidence-only AUROC: approximately +0.0178,
- confidence+epistemic improved AUROC in 3/3 seeds,
- confidence+epistemic improved log loss in 3/3 seeds.

However, the subgroup was too small for a reliable conclusion. The 2023 result was mixed and unstable across seeds.

Interpretation: suggestive, but underpowered.

## 2. Sparse-history company shift

### Definition

A company is sparse-history if it had at most four usable earnings episodes before 2022. This threshold was fixed before inspecting subgroup outcomes.

### Sample size

- 2023: 20 sparse-history examples per seed.
- 2024-2025: 47 sparse-history examples per seed.

### Result

2023 was negative:

- mean epistemic-minus-confidence error-detection AUROC: approximately -0.0622,
- mean incremental meta-model AUROC delta: approximately -0.0400.

2024-2025 showed only a weak positive hint:

- raw epistemic error detection remained slightly worse than confidence, about -0.0081 AUROC,
- mean confidence+epistemic meta-model AUROC delta: approximately +0.0108,
- log loss improved in 3/3 seeds.

Interpretation: the small unseen-company signal did not become stable when the definition was broadened to a somewhat larger sparse-history subgroup.

## 3. Underrepresented-sector shift

### Definition

A sector is underrepresented when its pre-2022 usable-episode count is at or below the 25th percentile across non-empty sectors.

This selected:

- Basic Materials,
- Communication Services,
- Energy.

### Sample size

- 2023: 133 examples per seed.
- 2024-2025: 204 examples per seed.

### Result

This was the most statistically useful shift subgroup and produced a clear negative result.

2023:

- mean epistemic-minus-confidence error-detection AUROC: approximately -0.0208,
- epistemic beat confidence on raw error-detection AUROC in 0/3 seeds,
- mean incremental meta-model AUROC delta: approximately -0.0156,
- confidence+epistemic beat confidence-only AUROC in 0/3 seeds.

2024-2025:

- mean epistemic-minus-confidence error-detection AUROC: approximately -0.0206,
- epistemic beat confidence on raw error-detection AUROC in 0/3 seeds,
- mean incremental meta-model AUROC delta: approximately -0.0034,
- confidence+epistemic beat confidence-only AUROC in 1/3 seeds.

Interpretation: broader sector underrepresentation did not reveal incremental value from Bayesian epistemic uncertainty beyond ordinary confidence.

## Overall interpretation

The distribution-shift experiments do not provide robust support for the hypothesis that Bayesian-LoRA epistemic uncertainty adds decision-relevant information beyond deterministic confidence in this earnings-event setup.

The evidence is internally consistent:

1. Epistemic uncertainty is associated with harder examples.
2. Ordinary confidence is generally at least as strong for error detection.
3. A strict unseen-company subgroup gave a small positive hint, but with only 12-35 examples per seed.
4. Broadening the shift to sparse-history companies produced mixed results.
5. Broadening further to underrepresented sectors, with 133-204 examples per seed, produced a clear negative result.

The project therefore stops further subgroup search at this point to avoid post-hoc subgroup fishing.

## Scientific conclusion

The strongest supported conclusion is:

> Bayesian-LoRA epistemic uncertainty is diagnostically meaningful, but in this financial earnings-event task ordinary deterministic confidence captures essentially all of the practically useful information tested here, including under the evaluated distribution shifts.

This conclusion is specific to the present model, task, Bayesian approximation, and shift definitions. It does not imply that Bayesian epistemic uncertainty is never useful. A genuinely new future study would require a separately pre-specified task or dataset with a stronger and more external distribution shift rather than additional slicing of the already-inspected sample.
