# Bayesian LoRA Portfolio Risk — Expanded Personal Walkthrough

This note is for **personal preparation**, not for presentation. The goal is to make the code easy to explain verbally without memorizing implementation details.

---

## 0. The whole project in one sentence

The project asks:

> If I make the LoRA adapters Bayesian, can the resulting epistemic uncertainty tell me something useful about model risk that ordinary prediction confidence does not already tell me, especially for portfolio decisions?

The answer so far is:

> The Bayesian uncertainty is real and reproducibly associated with harder examples, but in this earnings-event setup it has not shown clear incremental value beyond ordinary confidence once stronger controls are added.

That is the main story. Everything in the repo is some version of testing that statement more carefully.

---

# 1. Start with the basic prediction task

We have an earnings-call transcript and want to estimate whether the stock will have a sufficiently negative one-day move after the earnings event.

Very roughly:

`earnings transcript -> model -> probability of downside event`

For example, the model might output:

- `p = 0.10`: low predicted downside probability
- `p = 0.45`: ambiguous prediction
- `p = 0.80`: high predicted downside probability

The backbone is FinBERT, which is a BERT-style transformer pretrained for financial language.

We do **not** fully fine-tune all of FinBERT. Instead, we adapt it using LoRA.

---

# 2. What ordinary LoRA is doing

A transformer contains many large linear projections. Fine-tuning all of their weights is expensive.

LoRA says:

> Keep the original large weight matrix frozen, and learn a small low-rank correction instead.

If the original layer is conceptually:

`y = W x`

LoRA changes it to something like:

`y = W x + small learned update`

The small update is represented by two thin matrices, usually called `A` and `B`.

Instead of learning a huge new matrix directly, we learn:

`B @ A`

where the inner dimension is small, such as rank 16.

### Mental picture

Imagine FinBERT is a large machine that already works reasonably well.

Rather than rebuilding the machine, LoRA adds a small adjustable attachment to certain components.

The base machine stays fixed; only the attachments learn.

---

# 3. What makes our LoRA Bayesian

This is the core technical idea.

Ordinary LoRA learns **one fixed value** for every entry of `A` and `B`.

Bayesian LoRA instead learns a **distribution** for every entry.

So instead of saying:

> This LoRA parameter is exactly 0.037.

we say:

> This parameter is approximately Gaussian with some learned mean and learned standard deviation.

The implementation lives in:

`src/bayesian_lora_portfolio_risk/uncertainty/bayesian_lora.py`

The class `VariationalGaussianParameter` represents one group of Bayesian parameters.

It learns two things:

- `mu`: the mean
- `rho`: an unconstrained number that is transformed into a positive standard deviation

The code converts `rho` to a standard deviation with `softplus`:

```python
@property
def std(self):
    return F.softplus(self.rho) + self.eps
```

### Why not learn the standard deviation directly?

A standard deviation must be positive.

If we directly optimized `sigma`, gradient descent could push it negative.

So we optimize `rho`, which can be any real number, and transform it into a positive value with `softplus`.

That is a common variational-inference trick.

---

# 4. How we sample Bayesian LoRA weights

The key line is conceptually:

```python
theta = mu + sigma * epsilon
```

where:

`epsilon ~ Normal(0, 1)`

In the code:

```python
def sample(self):
    return self.mu + self.std * torch.randn_like(self.mu)
```

This is the **reparameterization trick**.

### Plain-English explanation

Instead of directly saying “sample a parameter from a Gaussian,” we construct the sample from:

1. the learned mean,
2. the learned standard deviation,
3. fresh standard-normal noise.

This matters because gradients can still flow through `mu` and `std` during training.

### What I should say verbally

> For every LoRA parameter I learn a Gaussian posterior. During a stochastic forward pass, I draw a sample using the reparameterization trick, so the model sees one plausible LoRA configuration. Repeating this gives different plausible models from the learned posterior.

---

# 5. What `BayesianLoRALinear` does

The class `BayesianLoRALinear` wraps an existing linear layer.

First, it freezes the original layer:

```python
for parameter in self.base_layer.parameters():
    parameter.requires_grad = False
```

Then it creates Bayesian versions of LoRA matrices `A` and `B`:

```python
self.param_a = VariationalGaussianParameter(...)
self.param_b = VariationalGaussianParameter(...)
```

During the forward pass:

1. sample `A` and `B` if posterior sampling is enabled;
2. compute the low-rank update;
3. add that update to the frozen base-layer output.

Conceptually:

`output = frozen FinBERT projection + Bayesian LoRA correction`

### Important initialization detail

`A` is initialized with a normal neural-network initialization, while the mean of `B` starts at zero.

That means the LoRA update initially contributes approximately nothing, so the model begins close to the pretrained FinBERT behavior.

### Verbal explanation

> The wrapper leaves the original FinBERT projection untouched and adds a low-rank Bayesian correction on top. At initialization the correction is near zero, so training starts from the pretrained model rather than disrupting it immediately.

---

# 6. Why query and value projections?

The function:

`inject_bert_attention_bayesian_lora(...)`

searches for BERT attention layers whose names end in:

- `attention.self.query`
- `attention.self.value`

and replaces those linear layers with Bayesian LoRA wrappers.

### What query means

The query projection helps determine:

> What information is this token looking for?

### What value means

The value projection controls:

> What information is actually passed forward when another token attends to this token?

### Why use LoRA there?

Query/value projections are common LoRA targets because they allow meaningful adaptation of attention behavior without changing every model parameter.

### What I should avoid saying

I should not imply that query/value is theoretically uniquely optimal here. It is a practical, standard low-parameter adaptation choice.

---

# 7. What the prior and posterior mean

We have an approximate posterior:

`q(theta)`

which is the learned distribution over Bayesian LoRA parameters.

We also choose a prior:

`p(theta)`

which is a Gaussian centered around zero with a fixed prior standard deviation.

The posterior is what the data wants the LoRA weights to look like.

The prior expresses the preference that the Bayesian correction should not move arbitrarily far unless the data supports it.

### Simple mental model

Prediction loss says:

> Fit the data.

KL regularization says:

> But do not become unnecessarily complicated or move too far away from the prior.

---

# 8. What the KL-divergence term is doing

Each Bayesian LoRA layer computes:

`KL(q || p)`

The code has a closed-form Gaussian KL calculation.

Then:

`model_kl_divergence(model)`

sums the KL over all Bayesian LoRA layers.

The training objective is conceptually:

`prediction loss + beta * KL penalty`

### What beta means

- large beta -> stronger pull toward the prior
- small beta -> more freedom to fit the data

Our early beta was too large.

That meant the KL term dominated too much of the objective.

We later ran a smaller beta sweep and selected approximately:

`beta = 0.003`

using validation behavior rather than choosing from test outcomes.

### Plain-English explanation

> Beta controls the tradeoff between fitting the task and staying close to the Bayesian prior. The original regularization was too strong, so I reduced it using validation-only experiments and then froze the choice before later robustness tests.

---

# 9. Posterior mean mode versus posterior sampling mode

The Bayesian layer has a flag:

`sample_weight`

When sampling is enabled:

```python
a = self.param_a.sample()
b = self.param_b.sample()
```

When sampling is disabled:

```python
a = self.param_a.mu
b = self.param_b.mu
```

So there are two useful modes.

### Posterior-mean mode

Use the learned means only.

This behaves like one deterministic model.

### Posterior-sampling mode

Repeatedly sample plausible LoRA weights.

Each sample gives a slightly different prediction.

That variation is what lets us estimate epistemic uncertainty.

---

# 10. How epistemic uncertainty is calculated

Suppose we run the same transcript through 32 posterior samples.

We might get downside probabilities like:

`0.31, 0.34, 0.27, 0.36, ...`

We compute:

### Posterior predictive mean

The average probability across samples.

This is the main prediction.

### Epistemic variance

The variance of those probabilities across samples.

If all sampled LoRA models agree, variance is low.

If plausible LoRA models disagree, variance is high.

### Mental interpretation

Low epistemic variance:

> Most plausible parameter settings give similar answers.

High epistemic variance:

> The answer depends more strongly on which plausible parameter setting I happened to sample.

That is why we interpret it as model/parameter uncertainty.

---

# 11. Epistemic uncertainty versus ordinary confidence

This distinction is essential.

Suppose the posterior mean downside probability is:

`p = 0.51`

The classifier itself is nearly undecided.

That is low ordinary confidence.

A simple confidence-uncertainty measure can therefore be based on distance from 0.5.

For example:

`2 * abs(p - 0.5)`

is high when the model is confident and low near 0.5.

Epistemic uncertainty asks something different:

> Do different plausible parameter samples disagree?

In theory, these are conceptually different signals.

But empirically they can contain overlapping information.

And that overlap turned out to be the central issue in this project.

---

# 12. The first result that looked promising

The initial result was encouraging because errors increased with epistemic uncertainty.

The lowest-uncertainty group had much lower error than the highest-uncertainty group.

Also, if we retained only the least-uncertain predictions, the overall error rate fell.

That established:

> The epistemic variance was not arbitrary noise. It was associated with model difficulty.

Praneeth specifically found this part interesting.

So the next scientific question became:

> Is that information uniquely Bayesian, or is ordinary confidence already telling us the same thing?

---

# 13. Why the confidence control was necessary

Without the confidence control, it would be easy to say:

> High Bayesian uncertainty predicts more mistakes, therefore Bayesian uncertainty is valuable.

But suppose high Bayesian uncertainty mostly occurs when predictions are already near 0.5.

Then ordinary confidence could explain the entire result.

So we directly compared:

- epistemic uncertainty
- ordinary confidence uncertainty

The result was that ordinary confidence was generally as good or slightly better for detecting errors.

Then we asked an even stronger question:

> Once confidence is already known, does epistemic variance add anything extra?

We fit simple error-prediction models:

`confidence only`

versus

`confidence + epistemic uncertainty`

The incremental gain was tiny or absent.

### Current interpretation

The epistemic signal is meaningful, but much of its useful information overlaps with ordinary confidence.

---

# 14. Why this is not the same as saying Bayesian LoRA failed

There are several different claims we could make.

### Claim A

Epistemic uncertainty is random.

This appears false.

### Claim B

Epistemic uncertainty predicts harder examples.

This appears true.

### Claim C

Epistemic uncertainty predicts errors better than ordinary confidence.

This has not held up.

### Claim D

Epistemic uncertainty adds economically useful information beyond confidence.

This has also not held up under the stronger controls tested so far.

So the correct conclusion is nuanced.

---

# 15. How we tried to turn uncertainty into portfolio decisions

The next question was:

> Even if epistemic uncertainty is not the best pure error detector, might it still be useful for position sizing?

We constructed simple event-risk allocation rules.

The base idea was:

`more predicted downside risk -> smaller position`

Then we compared three approaches.

### Posterior-mean-only sizing

Conceptually:

`weight ~ 1 - p`

If downside probability is high, invest less.

### Confidence-adjusted sizing

Further reduce exposure when the prediction is near 0.5.

### Epistemic-adjusted sizing

Further reduce exposure when posterior samples disagree strongly.

All strategies were normalized to the same gross exposure so that one could not appear safer merely by holding less capital overall.

---

# 16. Why the first allocation result looked exciting

Initially, epistemic-adjusted sizing looked substantially better.

It allocated less weight to bad events and improved downside metrics.

That suggested:

> Maybe Bayesian uncertainty has economic value even if it is not a superior classification-error detector.

But then we found an important confound.

The epistemic strategy was also much more concentrated.

In other words, it effectively said:

> Avoid a large number of events and concentrate capital in fewer cases.

That alone can change portfolio outcomes.

So we needed to determine whether the improvement was due to Bayesian ranking information or simply concentration.

---

# 17. What effective number of positions means

A portfolio may technically contain 1,000 positions but place almost all of its weight on 50 of them.

So raw position count is not enough.

The **effective number of positions** measures how diversified the weights really are.

Roughly:

- equal weights across many positions -> high effective N
- concentrated weights -> lower effective N

The epistemic-adjusted rule had substantially lower effective N than the confidence-adjusted rule in the initial tests.

That motivated the concentration-control experiment.

---

# 18. What the concentration-control experiment did

We transformed the weights so that the confidence strategy and epistemic strategy had the **same effective number of positions**.

Crucially, the matching transformation did not use outcomes to decide which individual events should receive more weight; it just adjusted concentration while preserving each rule's ranking.

Then we compared outcomes again.

The earlier epistemic advantage largely disappeared or reversed.

### Simple interpretation

Before concentration control:

> Epistemic looks better.

After equalizing concentration:

> Confidence is as good or better.

Therefore much of the earlier apparent Bayesian advantage was probably caused by concentration rather than uniquely better uncertainty information.

---

# 19. The model-risk penalty experiment

We then gave epistemic uncertainty another chance to contribute in a more explicit way.

The baseline was confidence-adjusted allocation.

Then we added an epistemic model-risk penalty:

`baseline weight * exp(-lambda * epistemic percentile)`

Interpretation:

- `lambda = 0`: ignore epistemic uncertainty
- larger lambda: increasingly penalize high epistemic uncertainty

We selected lambda on earlier calibration data while matching concentration.

Across all tested seeds and periods, the procedure selected:

`lambda = 0`

That is one of the clearest negative results in the project.

The data was effectively saying:

> Once confidence is already in the sizing rule and concentration is controlled, do not add an extra epistemic penalty.

---

# 20. Why we tested distribution shift

Epistemic uncertainty should theoretically be most useful when the model encounters unfamiliar situations.

So perhaps our ordinary test data was simply too similar to training data.

We therefore tested several forms of shift.

### Unseen companies

Companies with no usable pre-2022 examples.

This is conceptually clean, but the sample was tiny.

### Sparse-history companies

Companies with at most four pre-2022 earnings episodes.

This creates a somewhat larger group while still representing limited prior exposure.

### Underrepresented sectors

Sectors in the bottom quarter of pre-2022 episode counts.

This produced a substantially larger subgroup and therefore a more stable test.

---

# 21. What happened under distribution shift

The strict unseen-company and sparse-history tests contained a few positive hints, especially in some later-period subsets.

But those subgroups were very small, so the estimates were unstable.

The broader underrepresented-sector test was more informative.

The underrepresented sectors were defined only from pre-2022 sample frequency, not from downstream outcomes.

In that larger test, ordinary confidence still performed at least as well as epistemic uncertainty overall.

So distribution shift did not rescue the central hypothesis in a robust way.

### Why we stopped

At that point, continuing to invent more subgroup definitions could become post-hoc fishing.

So we adopted a stopping rule:

> Do not keep searching the already-inspected 2022-2025 sample for a subgroup where Bayesian uncertainty happens to win.

That is an important part of the scientific story.

---

# 22. What the project currently establishes

The cleanest current conclusion is:

> Bayesian-LoRA epistemic uncertainty is diagnostically meaningful, but in this earnings-event task ordinary deterministic confidence captures essentially all of the practically useful information we have tested.

More specifically:

1. Bayesian epistemic variance is associated with errors.
2. The ordering replicates across seeds.
3. It remains meaningful on later temporal data.
4. Ordinary confidence usually matches or beats it for error detection.
5. Conditional on confidence, little robust epistemic signal remains.
6. Initial allocation gains largely disappear after concentration matching.
7. An incremental model-risk penalty selects zero epistemic penalty.
8. Broader distribution-shift tests do not provide robust contrary evidence.

---

# 23. The most important code files and what each one means

## `src/.../uncertainty/bayesian_lora.py`

Purpose:

> Define the actual Bayesian LoRA mechanism.

Key things to understand:

- Gaussian variational parameters
- `mu` and `rho`
- reparameterized sampling
- KL to Gaussian prior
- frozen base layer + sampled low-rank update
- injection into query/value projections

If Praneeth wants implementation detail, this is probably the first file to open.

---

## `scripts/run_earnings_bayesian_lora.py`

Purpose:

> Train and evaluate the Bayesian-LoRA earnings model.

Conceptual flow:

`load data -> chronological split -> tokenize/chunk -> train -> posterior sampling -> uncertainty metrics -> save predictions`

Key outputs:

- posterior mean downside probability
- epistemic variance
- AUROC/Brier/NLL
- uncertainty/error diagnostics

---

## `scripts/run_earnings_bayesian_lora_uncertainty_control.py`

Purpose:

> Compare Bayesian epistemic uncertainty directly with ordinary confidence.

This is the first major falsification/control step.

---

## `scripts/analyze_earnings_epistemic_incremental_value.py`

Purpose:

> Ask whether epistemic uncertainty adds information after confidence is already known.

This is conceptually stronger than simply comparing the two signals separately.

---

## `scripts/analyze_earnings_event_risk_allocation.py`

Purpose:

> Translate prediction/uncertainty signals into simple position-sizing rules.

This produced the initially encouraging economic result.

---

## `scripts/analyze_earnings_event_risk_allocation_concentration_control.py`

Purpose:

> Check whether the apparent epistemic allocation advantage was really just caused by different concentration.

This changed the interpretation substantially.

---

## `scripts/analyze_earnings_model_risk_penalty.py`

Purpose:

> Add epistemic uncertainty as a separate penalty on top of confidence and let earlier calibration data decide whether the penalty should be used.

The selected penalty was zero.

---

## Distribution-shift scripts

Purpose:

> Test whether epistemic uncertainty becomes more useful on unfamiliar parts of the data.

The main variants were:

- unseen company
- sparse company history
- underrepresented sector

---

# 24. A simple verbal walkthrough I can actually use

If Praneeth asks me to explain the project from the code, I can say:

> I start with FinBERT and freeze the base model projections. In the query and value attention projections, I replace ordinary LoRA with Bayesian LoRA. Instead of learning fixed A and B matrices, each LoRA parameter has a Gaussian variational posterior with a learned mean and standard deviation. During training I sample these weights using the reparameterization trick and optimize prediction loss plus a KL term to a Gaussian prior.

Then:

> At inference I repeatedly sample the LoRA weights, run the same transcript through the model, and collect the downside probabilities. Their average is the posterior predictive mean, and their variance is my epistemic uncertainty measure.

Then:

> The first result was that this uncertainty was meaningfully ordered: high-uncertainty examples had more errors. That replicated across seeds. But once I compared it against ordinary confidence, confidence was generally as strong or stronger, and epistemic uncertainty added little incremental information after confidence was controlled for.

Then:

> I tried using epistemic uncertainty for position sizing. Initially it looked better, but it also generated a more concentrated portfolio. Once I matched concentration, the advantage mostly disappeared. I then tested an explicit model-risk penalty on top of confidence, and calibration consistently chose a penalty strength of zero.

Finally:

> I tested whether Bayesian uncertainty became more useful under company- and sector-level distribution shift. There were a few hints in very small subgroups, but the larger shift test was negative, so I stopped further subgroup searches to avoid post-hoc fishing.

That is enough for the high-level walkthrough.

---

# 25. Questions Praneeth may ask, with simple answers

## “Why Bayesian LoRA instead of full Bayesian fine-tuning?”

> Full Bayesian treatment of all FinBERT parameters would be much more expensive. Bayesian LoRA gives a tractable low-dimensional posterior over the adaptation parameters while leaving the pretrained backbone fixed.

## “Why mean-field Gaussian?”

> It is the simplest tractable variational posterior. Each parameter has its own independent Gaussian. It is computationally convenient, but it may also be too restrictive, which is one possible limitation of the current setup.

## “What exactly is epistemic uncertainty here?”

> Variance of the predicted downside probability across posterior samples of the Bayesian LoRA parameters.

## “Is that truly epistemic uncertainty?”

> It is a variational-posterior proxy for parameter uncertainty. I would avoid claiming it is a perfect decomposition of epistemic uncertainty, especially because the backbone and classifier are deterministic.

## “Why are only the adapters Bayesian?”

> Computational tractability. The experiment asks whether uncertainty in the learned task-specific adaptation is already enough to create useful model-risk information.

## “Why query and value?”

> They are standard high-leverage LoRA targets in transformer attention and give a compact adaptation mechanism. I do not claim they are uniquely optimal.

## “What does beta do?”

> It controls the strength of the KL regularization. Larger beta keeps the learned posterior closer to the prior; smaller beta lets the adapter fit the task more freely.

## “Why did you choose beta 0.003?”

> The original beta was too strong, so I ran a smaller validation-only sweep. Beta 0.003 and 0.01 were close, and the multi-seed validation result slightly favored 0.003. I then froze it for the later tests.

## “Why 32 posterior samples?”

> More samples give a more stable Monte Carlo estimate of predictive variance. We increased the count for the final uncertainty diagnostics once the expensive training setup was fixed.

## “Why does ordinary confidence matter so much?”

> Because predictions close to 0.5 are already naturally difficult. If epistemic variance rises on those same examples, it may look useful while mostly duplicating information available from the predictive probability itself.

## “Why did epistemic allocation initially look better?”

> It was more aggressive about shrinking uncertain positions, but that also made the portfolio substantially more concentrated. Once effective concentration was matched, confidence performed as well or better.

## “Does this mean Bayesian uncertainty is useless?”

> No. It is reproducibly associated with harder cases. The narrower conclusion is that it has not shown robust incremental decision value beyond ordinary confidence in this task and posterior parameterization.

## “What could make it work better?”

Potential possibilities to discuss rather than assert:

- richer posterior than mean-field Gaussian
- uncertainty in more of the network
- different Bayesian adapter placement
- a task where parameter uncertainty is structurally stronger
- more severe and externally defined distribution shift
- a portfolio formulation where model uncertainty enters covariance/risk rather than only scalar sizing

---

# 26. Things I do not need to memorize

I do **not** need to memorize:

- exact tensor dimensions in every function
- exact KL formula term by term
- every experiment filename
- every numerical result
- every helper function

I should instead remember:

1. what the code block is trying to accomplish;
2. what information goes in;
3. what comes out;
4. why that design choice was made;
5. what empirical result changed our interpretation.

If a detailed question comes up, it is completely reasonable to open the exact function and walk through it carefully.

---

# 27. Five ideas I should be able to explain without notes

Before talking with Praneeth, I should be able to explain these five things in my own words:

### 1. Bayesian LoRA

Instead of fixed low-rank adapter weights, learn distributions over the adapter weights.

### 2. Posterior sampling

Sample several plausible adapters and see how much their predictions disagree.

### 3. Epistemic variance

Prediction variance across those sampled adapters is used as the model-uncertainty signal.

### 4. Confidence control

Check whether that Bayesian signal tells us anything beyond the ordinary predicted probability's own confidence.

### 5. Portfolio concentration control

If one sizing rule looks safer but also concentrates capital much more aggressively, equalize concentration before attributing the improvement to better uncertainty information.

If I can explain these five clearly, I understand the central logic of the project.

---

# 28. Current scientific takeaway

The strongest version of the conclusion I should defend is:

> The Bayesian-LoRA posterior produces an uncertainty signal that is reproducibly related to prediction difficulty, but in the current FinBERT earnings-event setup that signal largely overlaps with ordinary confidence. Once confidence and portfolio concentration are controlled, I do not yet see robust incremental economic value from epistemic uncertainty.

This is a **negative but informative result**, not a failed implementation.

It tells us that the key open question is no longer merely whether we can produce Bayesian uncertainty.

We can.

The harder question is:

> Under what model, task, or portfolio formulation would Bayesian parameter uncertainty contain decision-relevant information that ordinary predictive confidence cannot already capture?

That is the natural point for discussion with Praneeth.
