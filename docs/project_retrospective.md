# Project Retrospective: Bayesian LoRA Portfolio Risk

**Repository:** `bayesian-lora-portfolio-risk`  
**Tagged Release:** `v1.0-research-artifact` (`de7e5ee97c9365f0dbb06b8aa91fd9012220c9fd`)  
**Study Status:** Completed, Audited, and Frozen (Controlled Negative Result)  

---

## 1. Project Overview

The central research question that motivated this project was:

> **Does epistemic model uncertainty from a Bayesian-adapted financial language model improve downstream financial-risk decisions beyond the information contained in its deterministic NLP prediction and ordinary classifier confidence?**

In empirical machine learning and quantitative finance, it is common to ask whether Bayesian uncertainty correlates with model error. However, that question is insufficient for decision-making. In real-world asset allocation, portfolio managers already have access to:
1. **The point prediction itself** (e.g., predicted downside event probability $p$).
2. **Deterministic classifier confidence** derived directly from that probability ($1 - 2|p - 0.5|$).
3. **Structured market and volatility estimates** (e.g., historical return volatility, asset covariances).

A Bayesian adapter that produces high uncertainty merely because the model is uncertain between two outcomes—or because an example lies near the classification boundary—may provide diagnostic signal without offering any **incremental** information beyond ordinary confidence. To test whether Bayesian parameter uncertainty has genuine practical utility, a study must distinguish four hierarchically nested concepts:

- **Predictive Signal:** Does financial text contain point-in-time information that predicts future asset risk better than naive baselines?
- **Uncertainty / Error-Diagnostic Signal:** Does Bayesian epistemic uncertainty reproducibly identify harder, more error-prone predictions?
- **Incremental Information Beyond Confidence:** Does epistemic uncertainty improve error detection or probability calibration once deterministic classifier confidence is controlled for?
- **Downstream Decision Value:** Does incorporating epistemic uncertainty into capital allocation or risk penalization improve portfolio risk-adjusted performance after controlling for portfolio concentration?

This retrospective provides a comprehensive, durable record of the experimental trajectory that evaluated this chain of hypotheses, the empirical hurdles encountered, the rigorous controls introduced, and the final controlled negative result.

---

## 2. Initial Hypothesis

The project originated from the hypothesis that adapting pre-trained financial language models with variational parameters could solve a fundamental challenge in financial NLP: **model risk under non-stationarity and distribution shift**.

Standard Low-Rank Adaptation (LoRA) freezes the base language model and trains low-rank parameter matrices $W = W_0 + \frac{\alpha}{r} B A$. While computationally efficient, deterministic LoRA yields a single point estimate of adapter parameters, leaving the model vulnerable to silent overconfidence on unfamiliar text, regime changes, and out-of-distribution corporate disclosures.

The hypothesized causal chain was:
$$\text{Financial Text } D_{i,t} \longrightarrow \text{Bayesian Model } \mathbf{w} \sim q_\theta(\mathbf{w}) \longrightarrow \text{Posterior Epistemic Variance } \sigma^2_{\text{epi}} \longrightarrow \text{Selective Filtering / Risk Penalty } e^{-\lambda u} \longrightarrow \text{Downstream Capital Protection}$$

Under this hypothesis, when corporate text displays novel linguistic phrasing, unfamiliar management tone, or anomalous financial context, the variational posterior across adapter samples would diverge, resulting in elevated prediction variance ($\sigma^2_{\text{epi}}$). That elevated uncertainty could then serve as an operational safety valve: positions with high model risk would be systematically down-weighted or selectively avoided, protecting the portfolio from large downside shocks.

Crucially, this chain was treated from the outset as an empirical hypothesis to be tested against strict falsification criteria rather than an architectural premise to be defended.

---

## 3. Phase I — FNSPID Daily-News Formulation

### Task Formulation and Design
The initial experimental phase (Milestone 0) was built upon the Financial News and Stock Price Integration Dataset (FNSPID). FNSPID joined timestamped news articles with daily equity price histories across U.S. equities.

- **Unit of Observation:** An **asset-day** $(i, t)$. All news articles timestamped for stock $i$ on trading day $t$ were aggregated into a bounded text input. Strict point-in-time rules assigned any article published at or after 16:00 New York time to the following calendar day ($t+1$).
- **Prediction Target:** 20-day forward maximum drawdown:
  $$DD_{i,t}^{(20)} = \min_{1 \le k \le 20} \left(\frac{P_{i,t+k}}{P_{i,t}} - 1\right), \quad y_{i,t} = \mathbf{1}\left[DD_{i,t}^{(20)} \le -\delta\right]$$
  with $\delta = 10\%$ fixed on the pre-test sample.
- **Baseline Models Tested:** Structured market baselines (trailing volatility, asset dummy variables), TF-IDF text baselines, single-window FinBERT embeddings, and chunked semantic document representations.

### The Falsification Hurdle: Failure of the Incremental Information Gate
Before introducing Bayesian LoRA, the project enforced a fundamental methodological requirement:

> **Information Gate Rule:** Bayesian uncertainty cannot rescue a text representation that has not first demonstrated robust incremental predictive information beyond simple structured market features.

Extensive diagnostic sweeps were conducted to test whether the daily text signal passed this gate:
1. **Structured Baselines vs. Raw Text:** Models using ticker identity and trailing volatility achieved AUROCs around 0.60–0.65, whereas text-only models struggled to exceed 0.52.
2. **Volatility Normalization:** To prevent models from learning stock-identity shortcuts (e.g., tech stocks have higher baseline volatility than utilities), the target was normalized by trailing 20-day historical volatility ($\sigma_{i,t}^{(20)}$). Under volatility normalization, the predictive performance of bag-of-words and pretrained transformer text baselines collapsed to random guessing (AUROC $\approx 0.50$).
3. **Relevance Filtering & Content Expansion:** Filtering news articles by company-mention relevance and expanding inputs from headline snippets to long article bodies failed to produce an incremental boost.
4. **Horizon Sweeps:** Evaluating drawdown windows at 3, 5, 10, and 20 trading days (`artifacts/relevance_filtered_horizon_sweep.json`) showed no horizon at which daily news text added statistically robust information over trailing volatility.
5. **Chunked Semantic Embeddings:** Long-document FinBERT chunking (`artifacts/chunked_semantic_baseline_2022.json`) confirmed that semantic aggregation did not overcome the low signal-to-noise ratio of daily news.

### Methodological Decision
The daily news formulation was formally rejected. Rather than applying Bayesian LoRA to an uninformative text stream, the study recognized that evaluating uncertainty on an uninformative base model would be scientifically meaningless. The project pivoted to an event-driven formulation where corporate text has an established, direct economic link to subsequent asset pricing.

---

## 4. Phase II — Earnings-Event Reformulation

### Task Definition and S&P 500 Earnings Corpus
The project pivoted to quarterly corporate earnings calls, utilizing long-form disclosure transcripts joined with pre-event market context and post-event returns (`RudrakshNanavaty/earnings-call-data`).

- **Observation Unit:** One company earnings episode $(i, q, t_e, D_{i,q})$ consisting of the earnings call transcript or release for company $i$ in quarter $q$ known at disclosure date $t_e$.
- **Downside Risk Target:** The close-to-close earnings announcement reaction:
  $$r^{(1)}_{i,q} = \frac{P_{i,D+1}^{\text{close}}}{P_{i,D-1}^{\text{close}}} - 1, \quad y_{i,q} = \mathbf{1}\left[r^{(1)}_{i,q} \le \tau\right]$$
  where $\tau$ was strictly estimated on pre-2022 training data at the 20th percentile ($\tau \approx -0.0384$, representing a 1-day downside move $\le -3.84\%$). Threshold $\tau$ was frozen before evaluating any post-2021 data.
- **Audited Price Anchors:** An audit of 18,201 episodes confirmed floating-point exact reconstruction of next-day price moves (`move_1d`) from verified open/high/low/close price anchors (`artifacts/earnings_label_audit/summary.json`).

### Temporal Split Architecture
The temporal structure was locked into three non-overlapping, strictly chronological periods:
1. **Pre-2022 Training Period ($\le 2021\text{-}12\text{-}31$):** Used for threshold estimation ($\tau$), vocabulary extraction, base model pre-training/tuning, and validation hyperparameter selection.
2. **2022–2023 Development Diagnostic Period ($2022\text{-}01\text{-}01 \text{ to } 2023\text{-}12\text{-}31$, $N = 2,369$):** Used for frozen Bayesian LoRA convergence diagnostics, confidence uncertainty comparisons, and exploratory allocation bridge design.
3. **2024–2025 Final Holdout Period ($2024\text{-}01\text{-}01 \text{ to } 2025\text{-}05\text{-}15$, $N = 1,728$):** Kept completely untouched during all model development; evaluated only once as a final out-of-time confirmation.

### Deterministic Gate 0 Success & LoRA Selection
Unlike daily news, long-form earnings transcripts cleanly passed Gate 0:
- TF-IDF and FinBERT text models achieved out-of-sample downside-event detection AUROCs of $0.58\text{--}0.61$, significantly outperforming random baselines.
- A deterministic LoRA hyperparameter sweep across ranks $r \in \{4, 8, 16\}$ and learning rates $\eta \in \{5\times 10^{-5}, 1\times 10^{-4}, 2\times 10^{-4}\}$ on pre-2022 validation splits identified rank $r=16$, learning rate $\eta = 2\times 10^{-4}$ as optimal (`results/earnings_lora_sweep_stage1/leaderboard.json`, `results/earnings_lora_stage2_repro/summary.json`).

With an informative deterministic baseline secured, the foundation was established to test Bayesian parameter uncertainty.

---

## 5. Phase III — Bayesian LoRA

### Conceptual Architecture
Bayesian LoRA replaces deterministic adapter matrices with variational distributions over weights. For linear projection layers $h = W_0 x + \Delta W x$, ordinary LoRA sets $\Delta W = \frac{\alpha}{r} B A$. Bayesian LoRA models the adapter parameters as random variables:
$$A \sim q_\theta(A), \quad B \sim q_\theta(B)$$

In the implementation ([`src/bayesian_lora_portfolio_risk/uncertainty/bayesian_lora.py`](file:///Users/xq/Documents/GitHub/bayesian-lora-portfolio-risk/src/bayesian_lora_portfolio_risk/uncertainty/bayesian_lora.py)):
- **Variational Posterior:** Mean-field Gaussian posterior $q_\theta(w) = \mathcal{N}(\mu, \sigma^2)$, parameterized by learned mean $\mu$ and unconstrained parameter $\rho$, where standard deviation $\sigma = \text{softplus}(\rho) = \log(1 + e^\rho)$.
- **Reparameterization Trick:** Weights are sampled differentiably during training: $w = \mu + \sigma \odot \epsilon$, where $\epsilon \sim \mathcal{N}(0, I)$.
- **Prior and Objective:** A standard isotropic Gaussian prior $p(w) = \mathcal{N}(0, \sigma_0^2 I)$. Training minimizes the negative Evidence Lower Bound (ELBO):
  $$\mathcal{L}(\theta) = \mathbb{E}_{q_\theta(w)} [\mathcal{L}_{\text{BCE}}(y, f_w(x))] + \beta \cdot D_{\text{KL}}(q_\theta(w) \,\|\, p(w))$$
  where KL weight $\beta = 0.003$ was selected on pre-2022 validation ELBO across a sweep $\beta \in \{0.001, 0.003, 0.01\}$.

### Monte Carlo Posterior Inference
At evaluation time, the model draws $S = 32$ Monte Carlo parameter samples $w^{(s)} \sim q_\theta(w)$. For each earnings disclosure, the forward pass produces probabilities $p^{(s)} = \sigma(f_{w^{(s)}}(x))$. These samples yield two distinct statistical quantities:
1. **Posterior Predictive Mean:**
   $$\bar{p} = \frac{1}{S} \sum_{s=1}^S p^{(s)}$$
   representing the consensus downside risk estimate.
2. **Epistemic Variance (Parameter Uncertainty):**
   $$\sigma^2_{\text{epi}} = \frac{1}{S} \sum_{s=1}^S \left(p^{(s)} - \bar{p}\right)^2$$
   measuring the disagreement among plausible adapter parameterizations.

---

## 6. First Important Result — Epistemic Uncertainty Is Real

The first major finding of the earnings-event experiment was positive: **Bayesian epistemic uncertainty is a real, reproducible diagnostic signal.**

Across three random seeds ($42, 123, 777$), predictions with higher epistemic variance were systematically more error-prone than predictions with low epistemic variance:
- **Quintile Error Rate Spread (2022–2023 Development Diagnostic):**
  - Quintile 1 (lowest $\sigma^2_{\text{epi}}$): Mean error rate $\approx 18.2\%$
  - Quintile 5 (highest $\sigma^2_{\text{epi}}$): Mean error rate $\approx 30.3\%$
  - Error spread ($Q_5 - Q_1$): $+12.06\%$ across seeds (`results/earnings_bayesian_lora/uncertainty_control_summary.json`).
- **Error-Detection AUROC:** Epistemic variance alone separated correct from incorrect predictions with an out-of-sample AUROC of $0.580$ on 2022–2023 development data and $0.590$ on the 2024–2025 final holdout.
- **Selective Risk Invariant:** Retaining only the lowest-uncertainty predictions monotonically reduced the overall portfolio classification error rate.

### Crucial Methodological Distinction
This result proved **diagnostic validity**: Bayesian LoRA was not outputting random noise; parameter variance meaningfully reflected prediction difficulty.

However, diagnostic validity does **not** prove incremental decision value. The critical next question was whether epistemic variance provided information that could not already be obtained from the predictive probability itself.

---

## 7. The Confidence Control

### Why Deterministic Confidence Is the Essential Control
In any binary classification task with output probability $p \in [0, 1]$, an example with $p = 0.50$ is maximally ambiguous, while an example with $p = 0.05$ or $p = 0.95$ represents a high-confidence prediction.

If a Bayesian model's adapter weights disagree most strongly on examples that are already close to the decision boundary ($p \approx 0.5$), the epistemic variance $\sigma^2_{\text{epi}}$ will correlate with classification errors simply because boundary cases are inherently more error-prone. To establish that Bayesian uncertainty adds value, it must outperform or augment the simplest deterministic measure of uncertainty: **confidence uncertainty**.

The repository defined confidence uncertainty as:
$$u_{\text{conf}} = 1 - 2\left|\bar{p} - 0.5\right| \in [0, 1]$$
where $u_{\text{conf}} = 0$ for certain predictions ($\bar{p} \in \{0, 1\}$) and $u_{\text{conf}} = 1$ at maximal ambiguity ($\bar{p} = 0.5$).

### The Primary Comparison: Epistemic vs. Confidence
The empirical results revealed a clear and consistent pattern across both development and holdout periods:

| Dataset Period | Epistemic Error-Detection AUROC | Confidence Error-Detection AUROC | Epistemic Minus Confidence $\Delta\text{AUROC}$ | Seeds Epistemic $>$ Conf |
|---|---|---|---|---|
| **2022–2023 Development Diagnostic** | $0.5801$ | **$0.5968$** | **$-0.0167$** | $0 / 3$ seeds |
| **2024–2025 Final Holdout** | $0.5898$ | **$0.5985$** | **$-0.0087$** | $0 / 3$ seeds |

In both periods and across every random seed, **ordinary classifier confidence outperformed Bayesian epistemic variance** at detecting model errors.

### The Incremental Meta-Model
To test whether combining both signals yielded incremental power, the study fit logistic error-prediction meta-models:
$$\text{Baseline: } \text{logit}(P(\text{error})) = \alpha_0 + \alpha_1 u_{\text{conf}}$$
$$\text{Augmented: } \text{logit}(P(\text{error})) = \gamma_0 + \gamma_1 u_{\text{conf}} + \gamma_2 \log(\sigma^2_{\text{epi}})$$

The out-of-sample incremental metrics were:
- **2022–2023 Development:** $\Delta\text{AUROC} = -0.0028 \pm 0.0050$, $\Delta\text{Log-Loss} = +0.0003$. Adding epistemic variance slightly degraded error detection (`results/earnings_bayesian_lora/incremental_epistemic_value.json`).
- **2024–2025 Final Holdout:** $\Delta\text{AUROC} = +0.0003 \pm 0.0004$, $\Delta\text{Log-Loss} = -0.0001$. The epistemic term added essentially zero predictive power (`results/earnings_bayesian_lora_final_holdout/summary.json`).

This transformed the study's interpretation: Bayesian LoRA epistemic uncertainty was real, but ordinary deterministic confidence captured virtually all of the actionable error-detection information.

---

## 8. Economic Decision Experiments

### The Allocation Bridge
Statistical error-detection metrics do not directly answer how uncertainty affects portfolio returns and capital risk. The project therefore built an economic allocation bridge mapping signals into stock weights for an event-driven portfolio.

All allocation strategies were constrained to have **equal gross exposure** ($\sum |w_i| = N$, mean weight $= 1.0$), ensuring that any observed difference in downside risk was driven by position sizing rather than aggregate leverage:
- **Posterior Mean Only:** $w_i \propto (1 - \bar{p}_i)$
- **Confidence-Adjusted:** $w_i \propto (1 - \bar{p}_i) \cdot (1 - u_{\text{conf}, i}) = (1 - \bar{p}_i) \cdot 2|\bar{p}_i - 0.5|$
- **Epistemic-Adjusted:** $w_i \propto (1 - \bar{p}_i) \cdot (1 - F_{\text{dev}}(\sigma^2_{\text{epi}, i}))$, scaling by empirical reliability relative to the prior development distribution.

### The Initial Apparent Advantage
In early unconstrained tests on the 2024–2025 holdout (`results/earnings_event_risk_allocation/summary.json`), epistemic sizing appeared superior:
- Posterior Mean Only Downside Event Rate: $25.71\%$
- Confidence-Adjusted Downside Event Rate: $24.46\%$
- Epistemic-Adjusted Downside Event Rate: **$22.62\%$** (a seemingly impressive $3.09$ percentage point reduction).

### The Concentration Discovery
Closer inspection of portfolio structure revealed a critical confounder: **portfolio concentration**.
The effective number of positions ($N_{\text{eff}}$, the inverse Herfindahl-Hirschman index) differed dramatically across rules:
$$N_{\text{eff}} = \frac{1}{\sum_{i=1}^N \left(\frac{w_i}{\sum w_k}\right)^2}$$
- Posterior Mean Only $N_{\text{eff}} \approx 1,719$
- Confidence-Adjusted $N_{\text{eff}} \approx 1,636$
- Epistemic-Adjusted $N_{\text{eff}} \approx 1,306$

Because the epistemic distribution had a heavy right tail, the epistemic reliability multiplier $(1 - F(\sigma^2_{\text{epi}}))$ drastically down-weighted many stocks, concentrating capital into a smaller subset of high-conviction positions. In a market where stock selection has positive average drift, a more concentrated portfolio naturally achieves a different risk profile.

### Concentration Matching via $\gamma$-Bisection
To isolate the information content of the uncertainty signal from pure concentration bias, the study introduced **concentration matching** (`scripts/analyze_earnings_event_risk_allocation_concentration_control.py`).
Each rule's raw weights were adjusted by a power transformation:
$$w_i(\gamma) = \text{normalize\_gross}(w_i^\gamma)$$
Using bisection, $\gamma$ was solved to exactly match the target rule's $N_{\text{eff}}$ without using future price outcomes.

The results dissolved the apparent epistemic advantage:
- **In 2023 (Target $N_{\text{eff}} = 856.3$):** When confidence allocation was concentrated to match epistemic concentration, its downside event rate dropped from $22.66\%$ to **$20.01\%$**, outperforming epistemic sizing ($20.81\%$).
- **In 2024–2025 Holdout (Target $N_{\text{eff}} = 1,306.5$):** Matched confidence achieved a downside event rate of **$21.75\%$**, beating epistemic sizing ($22.62\%$).
- In **0 out of 3 seeds** did epistemic sizing beat matched confidence.

The apparent economic superiority of Bayesian uncertainty was entirely an artifact of portfolio concentration.

---

## 9. Incremental Model-Risk Penalty

To give Bayesian uncertainty the most favorable possible decision formulation, the study tested an explicit **model-risk penalty** (`scripts/analyze_earnings_model_risk_penalty.py`).

### Design
Starting from the superior confidence-adjusted allocation, an exponential epistemic penalty was applied:
$$w_i(\lambda) = w_i^{\text{conf}} \cdot \exp\left(-\lambda \cdot F_{\text{cal}}(\sigma^2_{\text{epi}, i})\right)$$
where $\lambda$ governs penalty strength. To ensure fairness, each penalized candidate was concentration-matched back to the baseline confidence $N_{\text{eff}}$ before evaluation.

A grid search $\lambda \in \{0.0, 0.25, 0.5, 1.0, 2.0\}$ was calibrated strictly on prior-period data:
- In 2022 calibration data, the weighted downside event rate was lowest at $\lambda = 0.0$ ($0.2293$). Increasing $\lambda$ monotonically worsened performance:
  - $\lambda = 0.25 \implies 0.2295$
  - $\lambda = 0.50 \implies 0.2298$
  - $\lambda = 1.00 \implies 0.2304$
  - $\lambda = 2.00 \implies 0.2312$

### Operational Result
The calibration procedure selected **$\lambda^* = 0.0$ across 100% of seeds and 100% of evaluation periods** (`results/earnings_model_risk_penalty/summary.json`). Once deterministic confidence was accounted for, adding an explicit penalty for Bayesian model risk was rejected by the data.

---

## 10. Distribution-Shift Tests

The final scientific hypothesis was that epistemic uncertainty might become useful under conditions of **model unfamiliarity** where pre-trained representations degrade. Three pre-specified subgroups were audited before inspecting subgroup outcomes:

```
[Targeted Distribution-Shift Tests]
   │
   ├── Strict Unseen Companies (Symbol absent from pre-2022 data)
   │     └── 2023: N=12 per seed (unstable, mixed)
   │     └── 2024–2025: N=35 per seed (+0.0178 ΔAUROC hint; underpowered)
   │
   ├── Sparse-History Companies (<= 4 pre-2022 earnings episodes)
   │     └── 2023: N=20 per seed (negative, -0.0400 ΔAUROC)
   │     └── 2024–2025: N=47 per seed (+0.0108 ΔAUROC hint; unstable)
   │
   └── Underrepresented Sectors (Pre-2022 sector count <= 25th percentile)
         └── Basic Materials, Communication Services, Energy
         └── 2023: N=133 per seed (clear negative, -0.0156 ΔAUROC; 0/3 seeds beat conf)
         └── 2024–2025: N=204 per seed (clear negative, -0.0034 ΔAUROC; 1/3 seeds beat conf)
```

1. **Strict Unseen Companies ($N=35$ in Holdout):** Evaluated companies never seen in training. A slight positive hint appeared in 2024–2025 ($\Delta\text{AUROC} \approx +0.0178$), but sample size was tiny ($N=35$, only 5 error events) and 2023 was mixed.
2. **Sparse-History Companies ($N=47$ in Holdout):** Broadening the definition to companies with $\le 4$ prior episodes weakened the signal; 2023 was sharply negative ($\Delta\text{AUROC} = -0.0400$).
3. **Underrepresented Sectors ($N=133$ in 2023, $N=204$ in Holdout):** The largest, most statistically powered shift subgroup (Energy, Basic Materials, Communication Services) produced a **clear negative result**. Epistemic error detection underperformed confidence in both periods ($\Delta\text{AUROC} = -0.0208$ in 2023; $-0.0206$ in holdout; $0/3$ seeds beating confidence).

Broader domain shift failed to rescue the epistemic advantage.

---

## 11. The Stopping Rule

At this juncture in an empirical machine-learning project, researchers face a well-known temptation: continue slicing the dataset, testing alternative subgroup thresholds (e.g., small-cap stocks, high-volatility months, specific call transcript lengths), or trying alternative nonlinear uncertainty transformations until an empirically positive result emerges.

The project explicitly foresaw and rejected this practice. As formulated in `docs/research_spec.md`:

> **Stopping Rule for the Current Study:**  
> *"No additional subgroup thresholds or allocation transformations should be tuned on the already-inspected 2022–2025 sample. The current study is considered complete as a controlled negative-result investigation. A future continuation should begin as a genuinely new experiment with a separately pre-specified task or dataset, ideally involving an external source, a stronger domain shift, or a different uncertainty approximation."*

Stopping after the underrepresented sector analysis was essential to preserving the evidentiary value of the study. A negative result subject to pre-specified stopping rules is an informative scientific finding; a "positive" result obtained by exhaustive post-hoc subgroup fishing is simply an overfit illusion.

---

## 12. Final Scientific Conclusion

### Supported Findings
1. **Financial text contains predictive signal:** Earnings call transcripts provide statistically significant predictive information for next-day downside earnings event risk (Gate 0 passed).
2. **Bayesian parameter uncertainty is diagnostically valid:** Variational LoRA produces epistemic variance that is reproducibly associated with prediction errors and difficulty across multiple random seeds and time periods.
3. **Ordinary classifier confidence is superior:** A simple transformation of the deterministic prediction ($1 - 2|p - 0.5|$) captures equal or greater error-detection power than epistemic variance.
4. **Incremental predictive value is negligible:** Incremental meta-models combining confidence and epistemic variance show essentially zero out-of-sample benefit ($\Delta\text{AUROC} < 0.001$).
5. **Economic advantages disappear under concentration controls:** Apparent capital-allocation gains from epistemic weighting are fully explained by induced portfolio concentration.
6. **Model-risk penalization selects zero weight:** Optimization of an explicit model-risk penalty sets $\lambda^* = 0.0$ across all seeds and periods.
7. **Tested distribution shifts do not restore incremental value:** Slicing by company unfamiliarity or underrepresented sectors does not show robust epistemic utility beyond confidence.

### What the Study Does Not Claim
- The study does **not** claim that Bayesian deep learning is universally useless.
- The study does **not** claim that Bayesian LoRA cannot add value in other NLP domains (e.g., medical diagnostics, open-domain question answering).
- The study does **not** rule out that different uncertainty approximations (e.g., Deep Ensembles, Laplace approximations), different tasks (e.g., continuous volatility forecasting), or severe out-of-distribution shifts (e.g., cross-asset or international transfer) might succeed.

The finding is a **clean, controlled negative result**: in the context of S&P 500 earnings disclosure risk, ordinary deterministic confidence captures essentially all practically useful error-diagnostic information found in the study.

---

## 13. Research-Artifact Hardening

Once the scientific conclusions were locked, a comprehensive engineering effort hardened the repository into an immutable computational appendix:
1. **Claims-to-Code Mapping ([`docs/claims_to_code_mapping.md`](file:///Users/xq/Documents/GitHub/bayesian-lora-portfolio-risk/docs/claims_to_code_mapping.md)):** Mapped every scientific claim in `README.md` and `docs/research_spec.md` to its exact script, function, and result artifact.
2. **Pure Evaluation Extraction ([`src/.../uncertainty/evaluation.py`](file:///Users/xq/Documents/GitHub/bayesian-lora-portfolio-risk/src/bayesian_lora_portfolio_risk/uncertainty/evaluation.py)):** Decoupled monolithic scripts into tested library functions for confidence uncertainty, empirical CDFs, effective position counts, $\gamma$-bisection, allocation metrics, and subgroup partitioning.
3. **Test Suite Expansion:** Expanded test coverage from 23 basic pilot tests to 50 comprehensive tests across 10 modules:
   - Variational parameter sampling, KL analytical divergence, and Bayesian LoRA attention injection (`tests/test_bayesian_lora.py`).
   - Temporal split disjointness and chronological causality invariants (`tests/test_leakage_and_invariants.py`).
   - Numerical boundary stability under precision limits (`tests/test_numerical_stability.py`).
   - Committed JSON schema snapshots (`tests/test_results_and_stability.py`).
4. **Reproducibility Verification Script ([`scripts/verify_reproducibility.py`](file:///Users/xq/Documents/GitHub/bayesian-lora-portfolio-risk/scripts/verify_reproducibility.py)):** Built a standalone verification tool to validate the complete mathematical and statistical calculation chain in seconds without requiring GPU retraining.
5. **Automated CI & Tooling ([`.github/workflows/ci.yml`](file:///Users/xq/Documents/GitHub/bayesian-lora-portfolio-risk/.github/workflows/ci.yml)):** Added GitHub Actions running Ruff, Pyright, Pytest, and reproducibility verification.

---

## 14. Adversarial Audit

To test the integrity of the claims, an independent adversarial audit was conducted on the hardened repository. The audit specifically looked for temporal leakage, outcome-informed parameter tuning, code drift, broken dependencies, and unjustified claims.

The audit produced six distinct findings (**FALSIFY-01** through **FALSIFY-06**):
- **FALSIFY-01 (Clean-Clone Failure):** `scripts/verify_reproducibility.py` hard-failed on a clean `git clone` because it unconditionally expected prediction CSVs in `artifacts/`, which were `.gitignore`d. Documentation erroneously claimed these large CSVs were committed to git.
- **FALSIFY-02 (Predictor Scaling Discrepancy):** `analyze_earnings_epistemic_incremental_value.py` applied `StandardScaler` to meta-model predictors $[u_{\text{conf}}, \log \sigma^2_{\text{epi}}]$, whereas `evaluation.py` and subgroup scripts used unscaled features.
- **FALSIFY-03 (Training Window Discrepancy):** Meta-models across historical scripts varied between 2022-only training and expanding 2022–2023 pooled training.
- **FALSIFY-04 (Calibration Protocol Discrepancy):** The exploratory raw allocation script used a pooled 2022–2023 reference CDF, while later concentration-control scripts used a sequential expanding-window calibration ($F_{2022} \to 2023$ and $F_{\text{pre-2024}} \to 2024\text{–}2025$).
- **FALSIFY-05 (Test Coupling):** `tests/test_results_and_stability.py` combined pure mathematical unit tests with frozen artifact regression snapshot checks without distinguishing functional proof from file integrity.
- **FALSIFY-06 (Software Edge Case):** When raw scores were identically zero (e.g., $p \equiv 0.5$ everywhere where confidence reliability is zero), gross normalization crashed with `ValueError: allocation rule produced zero gross exposure`.

Crucially, the adversarial audit **did not identify temporal leakage or outcome-informed tuning** in the headline scientific results. Reconstructed metrics matched committed numbers.

---

## 15. Audit Remediation

Every audit finding was remediated under strict constraints: **no historical scripts or committed result files were altered.**

### FALSIFY-01: Dual Verification Tiers
`scripts/verify_reproducibility.py` was refactored into two distinct, clearly separated modes:
1. **Clean-Clone Mode (Default, CI-Safe):** Verifies all 28 committed result JSON files, schemas, and statistical invariants. Requires zero untracked artifacts. Runs in CI on a fresh `git clone`.
2. **Deep Local-Artifact Mode (`--deep`):** When local prediction CSVs are present, recomputes AUROCs and portfolio allocations directly from raw Monte Carlo prediction files.
Documentation in `docs/reproducibility.md` and `docs/audit_log.md` was corrected to state that large prediction CSVs are local artifacts excluded via `.gitignore`.

### FALSIFY-02 & FALSIFY-03: $2 \times 2$ Factorial Sensitivity Analysis
Rather than choosing a single preferred specification, a dedicated sensitivity analysis script was added: [`scripts/analyze_meta_model_sensitivity.py`](file:///Users/xq/Documents/GitHub/bayesian-lora-portfolio-risk/scripts/analyze_meta_model_sensitivity.py). It evaluated the complete $2 \times 2$ grid:
$$(\text{Scaled vs. Unscaled}) \times (2022\text{-only vs. } 2022\text{–}2023\text{ pooled training})$$
across all 3 seeds and all predefined evaluation populations (`docs/meta_model_sensitivity_analysis.md`).

**Committed Sensitivity Findings:**
- On the full 2024–2025 holdout ($N=1,728$), $\Delta\text{AUROC}$ ranged from $+0.0003$ to $+0.0007$, with $\Delta\text{Log-Loss}$ near $0.0000$.
- On underrepresented sectors, $\Delta\text{AUROC}$ remained negative in 3 of 4 specifications ($-0.0054$ to $+0.0001$), with positive log loss across all four.
- The headline negative conclusion was completely invariant to predictor scaling and training window choices.

### FALSIFY-04: Documenting Calibration Evolution
Section 4.1 was added to [`docs/reproducibility.md`](file:///Users/xq/Documents/GitHub/bayesian-lora-portfolio-risk/docs/reproducibility.md#41-calibration-protocol-evolution-falsify-04-remediation) explaining the methodological evolution from exploratory pooled calibration to sequential expanding-window calibration.

### FALSIFY-05: Decoupled Test Architecture
Mathematical stability tests were separated into [`tests/test_numerical_stability.py`](file:///Users/xq/Documents/GitHub/bayesian-lora-portfolio-risk/tests/test_numerical_stability.py) (verifying mathematical properties, boundary limits, and scale invariance), while [`tests/test_results_and_stability.py`](file:///Users/xq/Documents/GitHub/bayesian-lora-portfolio-risk/tests/test_results_and_stability.py) was explicitly documented as artifact snapshot invariants.

### FALSIFY-06: Zero-Score Allocation Fallback
Added `fallback_equal=True` to `normalize_same_gross`, `compute_allocation_weights`, and `allocation_metrics` in `src/.../uncertainty/evaluation.py`. When an allocation rule produces all-zero raw scores (e.g., $p \equiv 0.5$ everywhere), the functions return uniform equal weights ($1.0$) rather than crashing. Added unit tests in `tests/test_evaluation_metrics.py`.

---

## 16. Final Verification and Freeze

The finalized computational appendix was verified across all tiers and frozen:
- **Final Release Tag:** `v1.0-research-artifact`
- **Tagged Commit SHA:** `de7e5ee97c9365f0dbb06b8aa91fd9012220c9fd`
- **Ruff Linter:** Passed (0 errors).
- **Pyright Type Checker:** Passed (0 errors, 47 benign pandas-stubs union warnings fully classified in `docs/audit_log.md`).
- **Pytest Suite:** 50 passed in 2.94s.
- **Clean-Clone Verification:** Passed 100% in a pristine `/tmp` simulation containing no `artifacts/` folder, no GPU, no weights download, and no network access.
- **Deep Local Verification:** Passed 100%, matching committed research outputs bit-for-bit.

The release tag represents an immutable snapshot of the audited study.

---

## 17. Lessons Learned

The journey of `bayesian-lora-portfolio-risk` offers several methodological lessons for applied machine learning and quantitative research:

1. **Information Gates Before Complex Modeling:** Do not apply sophisticated Bayesian uncertainty methods to an uninformative text stream. Verifying that the deterministic representation contains incremental predictive power (Phase II vs. Phase I) is an essential prerequisite.
2. **Diagnostic Correlation $\ne$ Incremental Utility:** Demonstrating that uncertainty correlates with error ($Q_5 - Q_1 = +12\%$) is easy, but it does not establish that the uncertainty is decision-relevant. Always test whether the signal adds value beyond simple confidence baselines ($1 - 2|p - 0.5|$).
3. **Beware Hidden Concentration Bias in Portfolio Sizing:** In financial applications, weighting assets by an uncertainty score almost always alters portfolio concentration. Apparent risk reductions can simply be the artifact of holding fewer assets. Concentration matching ($\gamma$-bisection) is required to evaluate the true information content of an allocation rule.
4. **Negative Results Demand Pre-Specified Stopping Rules:** Slicing data into narrower subgroups in search of a "positive" finding introduces massive researcher degrees of freedom. Adhering to a stopping rule preserves scientific credibility.
5. **Separate Committed Evidence from Local Artifacts:** An auditable computational appendix must distinguish lightweight, version-controlled summary evidence (`results/`) from large, machine-specific scratch artifacts (`artifacts/`). Clean-clone CI must succeed from git alone.
6. **Adversarial Auditing Strengthens Research:** Auditing code from an adversarial perspective identifies protocol discrepancies, test coupling, and software edge cases that normal reviews miss. Remediating them transparently makes the evidence chain unassailable.

---

## 18. What Comes Next

Because the 2022–2025 earnings-event study is complete and closed, no further parameter tuning, subgroup searching, or weighting adjustments should be conducted on this dataset.

Any future research continuing from this repository should be structured as a **genuinely new, separately pre-specified study**:
- **Stronger External Distribution Shifts:** Evaluating cross-market transfer (e.g., U.S. models applied to European or Asian disclosures) or macroeconomic crisis regimes (e.g., sudden interest-rate shock episodes) where base model vocabulary shifts more severely.
- **Alternative Uncertainty Architectures:** Comparing variational LoRA against Deep Ensembles, Laplace approximations, or conformal prediction sets.
- **Continuous Targets:** Extending the formulation from binary event classification to continuous volatility or tail-risk density forecasting.

Until such a study is launched, this repository stands as an audited, reproducible, and immutable record of a completed computational investigation.
