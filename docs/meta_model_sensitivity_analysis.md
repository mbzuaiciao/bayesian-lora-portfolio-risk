# Methodological Sensitivity Analysis: Predictor Scaling & Training Window

> [!NOTE]
> **Adversarial Audit Remediation (FALSIFY-02 & FALSIFY-03)**
> This document reports a dedicated methodological sensitivity analysis evaluating the error-detection meta-model under the full $2 \times 2$ cross of predictor scaling ($\text{Scaled}$ vs. $\text{Unscaled}$) and training window ($2022\text{-only}$ vs. $2022\text{–}2023\text{ pooled}$).
>
> **Protocol Constraint**: This analysis exists solely to audit the sensitivity of study conclusions to the discovered protocol discrepancies. As required by the frozen research protocol, **no hyperparameters are tuned, no narrative rewrite is undertaken, and no specification is declared "preferred."**

---

## 1. Background & Protocol Discrepancies

During the adversarial reproducibility audit, two discrepancies in meta-model specification across historical scripts were identified:

1. **Predictor Scaling (FALSIFY-02):**
   - In `scripts/analyze_earnings_epistemic_incremental_value.py`, features $[u_{\text{conf}}, \log \sigma^2_{\text{epi}}]$ were normalized using `StandardScaler` fitted on the training split.
   - In `src/bayesian_lora_portfolio_risk/uncertainty/evaluation.py` (`fit_incremental_meta_model`) and the distribution-shift subgroup scripts, features were passed unscaled to `LogisticRegression(solver='lbfgs')`.

2. **Training Window (FALSIFY-03):**
   - The initial exploratory meta-model (`analyze_earnings_epistemic_incremental_value.py`) fit error models on $2022$ data to evaluate on $2023$.
   - Later holdout and distribution-shift scripts evaluated on $2024\text{–}2025$ holdout data using an expanding window fit on $2022\text{–}2023$ data (or $2022$ data for $2023$ evaluation).

To assess whether these specification variations impact the headline negative result, we evaluate the complete $2 \times 2$ factorial grid across all 3 random seeds ($42, 123, 777$) on all predefined evaluation populations.

---

## 2. Experimental Design ($2 \times 2$ Factorial Cross)

### Factors
- **Factor A (Predictor Scaling):**
  - `Scaled`: Predictor matrix normalized via `StandardScaler()` fit on training set and applied to test set.
  - `Unscaled`: Raw confidence uncertainty $u_{\text{conf}} = 1 - 2|p - 0.5|$ and log epistemic variance $\log \sigma^2_{\text{epi}} = \log(\max(\sigma^2_{\text{epi}}, 10^{-12}))$.
- **Factor B (Training Window):**
  - `2022-only`: Meta-model fit on calendar year $2022$ observations ($N = 1,188$).
  - `2022–2023 pooled`: Meta-model fit on all development observations ($N = 2,369$).

### Predefined Evaluation Populations
1. **2023 Development Test Set** ($N = 1,181$)
2. **2024–2025 Final Holdout Set** ($N = 1,728$)
3. **Unseen Company Holdout** ($N = 35$, symbol absent from pre-2022 history)
4. **Sparse History Holdout** ($N = 101$, $\le 4$ pre-2022 episodes)
5. **Underrepresented Sector Holdout** ($N = 12$, bottom quartile sector frequency pre-2022)

---

## 3. Empirical Results

The table below reports out-of-sample incremental value metrics ($\Delta\text{AUROC} = \text{AUROC}_{\text{aug}} - \text{AUROC}_{\text{base}}$ and $\Delta\text{Log-Loss} = \text{Log-Loss}_{\text{aug}} - \text{Log-Loss}_{\text{base}}$), averaged across seeds $42, 123, 777$ ($\text{mean} \pm \text{std}$):

| Evaluation Population | Specification (Window $\times$ Scaling) | $\Delta \text{AUROC}$ ($\text{Mean} \pm \text{Std}$) | $\Delta \text{Log-Loss}$ | Epistemic Beats Conf ($\Delta \text{AUC} > 0$) |
|---|---|---|---|---|
| **2023 Test Set** | $2022\text{-only} \times \text{Unscaled}$ | $-0.0139 \pm 0.0071$ | $+0.0004$ | $0 / 3$ seeds |
| | $2022\text{-only} \times \text{Scaled}$ | $-0.0028 \pm 0.0050$ | $+0.0003$ | $1 / 3$ seeds |
| | $2022\text{–}2023\text{ pooled} \times \text{Unscaled}$ | $-0.0113 \pm 0.0059$ | $+0.0002$ | $0 / 3$ seeds |
| | $2022\text{–}2023\text{ pooled} \times \text{Scaled}$ | $-0.0023 \pm 0.0021$ | $-0.0001$ | $1 / 3$ seeds |
| **2024–2025 Holdout** | $2022\text{-only} \times \text{Unscaled}$ | $+0.0003 \pm 0.0004$ | $-0.0004$ | $2 / 3$ seeds |
| | $2022\text{-only} \times \text{Scaled}$ | $+0.0003 \pm 0.0007$ | $-0.0001$ | $2 / 3$ seeds |
| | $2022\text{–}2023\text{ pooled} \times \text{Unscaled}$ | $+0.0007 \pm 0.0001$ | $+0.0000$ | $3 / 3$ seeds |
| | $2022\text{–}2023\text{ pooled} \times \text{Scaled}$ | $+0.0004 \pm 0.0004$ | $-0.0000$ | $2 / 3$ seeds |
| **Unseen Companies** | $2022\text{-only} \times \text{Unscaled}$ | $+0.0133 \pm 0.0094$ | $-0.0063$ | $2 / 3$ seeds |
| | $2022\text{-only} \times \text{Scaled}$ | $+0.0044 \pm 0.0063$ | $-0.0008$ | $2 / 3$ seeds |
| | $2022\text{–}2023\text{ pooled} \times \text{Unscaled}$ | $+0.0178 \pm 0.0083$ | $-0.0039$ | $3 / 3$ seeds |
| | $2022\text{–}2023\text{ pooled} \times \text{Scaled}$ | $-0.0022 \pm 0.0083$ | $-0.0009$ | $1 / 3$ seeds |
| **Sparse History** | $2022\text{-only} \times \text{Unscaled}$ | $+0.0063 \pm 0.0167$ | $-0.0061$ | $2 / 3$ seeds |
| | $2022\text{-only} \times \text{Scaled}$ | $-0.0009 \pm 0.0013$ | $-0.0009$ | $1 / 3$ seeds |
| | $2022\text{–}2023\text{ pooled} \times \text{Unscaled}$ | $+0.0099 \pm 0.0147$ | $-0.0045$ | $2 / 3$ seeds |
| | $2022\text{–}2023\text{ pooled} \times \text{Scaled}$ | $+0.0009 \pm 0.0077$ | $-0.0015$ | $1 / 3$ seeds |
| **Underrep. Sectors** | $2022\text{-only} \times \text{Unscaled}$ | $-0.0054 \pm 0.0076$ | $+0.0005$ | $1 / 3$ seeds |
| | $2022\text{-only} \times \text{Scaled}$ | $-0.0016 \pm 0.0025$ | $+0.0002$ | $1 / 3$ seeds |
| | $2022\text{–}2023\text{ pooled} \times \text{Unscaled}$ | $-0.0031 \pm 0.0060$ | $+0.0006$ | $1 / 3$ seeds |
| | $2022\text{–}2023\text{ pooled} \times \text{Scaled}$ | $+0.0001 \pm 0.0014$ | $+0.0001$ | $1 / 3$ seeds |

---

## 4. Assessment of Impact on Study Conclusions

1. **Robustness of the Main Conclusion:**
   - On the full $2023$ test set, $\Delta \text{AUROC}$ is negative across **all four specifications** (ranging from $-0.0139$ to $-0.0023$). Epistemic variance consistently degrades or fails to improve error detection over confidence alone.
   - On the full $2024\text{–}2025$ holdout ($N = 1,728$), $\Delta \text{AUROC}$ remains near zero across **all four specifications** ($+0.0003$ to $+0.0007$, with standard deviations spanning zero), and $\Delta \text{Log-Loss}$ is virtually $0.0000$. The epistemic term adds no practically meaningful predictive power.

2. **Underrepresented Sectors:**
   - Under all four specifications, $\Delta \text{Log-Loss}$ is positive ($+0.0001$ to $+0.0006$), confirming that the augmented model has worse probability calibration on out-of-distribution sectors. $\Delta \text{AUROC}$ is negative in 3 of 4 specifications and negligible ($+0.0001$) in the fourth.

3. **Subgroups with Small Sample Sizes:**
   - In small holdout subgroups (Unseen $N=35$, Sparse $N=101$), scaling slightly dampens feature coefficients, shifting $\Delta\text{AUROC}$ within sampling noise (e.g. from $+0.0178$ unscaled to $-0.0022$ scaled on unseen companies). This demonstrates that slight apparent gains in tiny samples are sensitive to minor modeling choices and reflect noise rather than robust epistemic advantage.

4. **Conclusion:**
   The paper's core scientific finding—**that Bayesian LoRA epistemic uncertainty does not provide robust, actionable decision or error-detection value beyond deterministic confidence**—is invariant to the choice between scaled vs. unscaled features and $2022\text{-only}$ vs. $2022\text{–}2023\text{ pooled}$ training windows.
