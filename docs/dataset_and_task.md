# Milestone 0 Dataset and Prediction Task

## Dataset choice

Use FNSPID (Financial News and Stock Price Integration Dataset) for the first controlled experiment.

FNSPID joins timestamped financial news with historical stock-price data across thousands of U.S. equities over a long historical window. That makes it suitable for constructing strictly forward-looking labels without first building a separate SEC/market-data ingestion stack.

A later phase can move to SEC filings, where the documents are more fundamental and economically interpretable but the data engineering burden is materially larger.

## Unit of observation

The initial unit is an **asset-day** rather than an individual article.

For stock i on trading day t:

- collect news associated with i that is available on day t,
- concatenate or summarize the eligible text into one bounded model input,
- associate that input with a future market-risk label beginning after t.

This avoids treating several articles about the same event as independent observations.

## Primary target

Milestone 0 predicts a future downside event over the next 20 trading days.

Let P(i,t) be the closing price on the observation date and let P(i,t+k) be the closing price k trading days later. Define the forward maximum drawdown from the observation price as

\[
DD_{i,t}^{(20)} = \min_{1 \le k \le 20} \left(\frac{P_{i,t+k}}{P_{i,t}} - 1\right).
\]

The initial binary target is

\[
y_{i,t} = \mathbf{1}\left[DD_{i,t}^{(20)} \le -\delta\right].
\]

The threshold delta is a configuration parameter, not a hidden tuning choice. Start with 10%, inspect class balance on the training period only, and pre-register any adjustment before evaluating the test periods.

## Why this target

The target is deliberately a **risk event**, not next-period return direction.

That matches the downstream research question: can text-derived epistemic uncertainty improve portfolio risk decisions? A future drawdown event is more naturally connected to risk adjustment and position sizing than a generic up/down label.

## Point-in-time rule

No information after the asset-day cutoff may enter the model input.

The label window begins after t. Same-day price movement is not part of the prediction target. If an article timestamp cannot establish whether it was available before the market close, the conservative default is to assign it to the next trading-day information set.

This rule is more important than maximizing sample count.

## Initial universe

Start with 50-100 liquid U.S. equities with:

- long continuous price histories,
- sufficient news coverage,
- representation across multiple sectors.

The universe must be selected using information available independently of the future test outcome. Avoid choosing companies because they experienced notable later drawdowns.

## Text representation

For the first experiment, use a bounded amount of text per asset-day.

Preferred order:

1. article title,
2. available article summary,
3. truncated article body only if needed.

Using titles/summaries first keeps Milestone 0 computationally tractable and reduces the chance that document length becomes the dominant engineering problem.

## Temporal splits

Do not randomly split examples.

Use chronological splits so the shifted test set is genuinely later than the training data. Exact dates will be chosen after inspecting coverage, but the structure is fixed:

- **Train:** earlier years,
- **Validation / ID test:** later observations still near the training regime,
- **Temporal-shift test:** a strictly later period containing materially different market conditions.

All hyperparameter and threshold decisions must use only train and validation data.

## Primary comparison

Hold the data, tokenizer, base model, train/validation/test split, and prediction head fixed.

Compare:

1. deterministic LoRA,
2. Bayesian LoRA posterior mean,
3. Bayesian LoRA posterior predictive distribution.

The key Milestone 0 question is not whether Bayesian LoRA improves accuracy. It is whether its uncertainty identifies cases in which the text-derived prediction is unreliable.

## Required uncertainty tests

For the temporal-shift set, test whether:

1. mean epistemic uncertainty increases relative to ID data,
2. incorrect predictions have higher uncertainty than correct predictions,
3. error rate generally increases across uncertainty quantiles,
4. rejecting the most uncertain examples improves selective risk,
5. Bayesian LoRA improves calibration relative to deterministic LoRA while retaining comparable discrimination.

## Secondary targets

Only after the drawdown experiment is complete should we consider:

- future realized-volatility increase,
- abnormal drawdown relative to a market or sector benchmark,
- credit deterioration,
- earnings deterioration,
- rating downgrade.

Do not run several targets simultaneously in Milestone 0.

## Data-engineering extension

SEC filings are the preferred later extension because filing timestamps and document semantics make the economic story cleaner. They are intentionally deferred until the core uncertainty hypothesis survives the cheaper FNSPID experiment.
