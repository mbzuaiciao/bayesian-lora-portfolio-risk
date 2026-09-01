# Structured Baseline Gate

Before training a language model, test whether the current drawdown target is already largely explained by simple structured variables.

## Why this gate exists

The full eight-ticker pilot audit produced a usable overall event frequency, but the event rate varied sharply by ticker. An absolute 10% drawdown has different meaning for a high-volatility stock than for a low-volatility stock.

A language model could therefore appear useful while merely exploiting stable asset identity or correlates of baseline volatility.

## Required pre-LLM baselines

Evaluate chronologically:

1. Constant training-period event rate.
2. Ticker identity only.
3. Trailing 20-trading-day realized volatility only.
4. Ticker identity plus trailing realized volatility.

The trailing-volatility feature must use only returns known before the labeled observation date.

## Provisional diagnostic split

For the first structured-baseline audit only:

- Train: observations through 2021-12-31.
- Diagnostic test: 2022-01-01 through 2022-12-31.

This is not yet the final Milestone 0 train/ID/shift split. It is a quick diagnostic to determine whether the target definition is dominated by structured risk differences.

## Decision rule

If ticker and/or recent volatility provides substantial discrimination relative to the constant baseline, then do not proceed directly to LoRA on the absolute 10% target.

Instead, consider a risk-normalized target, such as:

- drawdown scaled by trailing volatility,
- abnormal downside relative to a stock-specific historical distribution,
- downside relative to a market or sector benchmark.

If simple structured baselines have little discrimination, the absolute drawdown target remains a plausible first text task.

The objective is not to make the target hard. The objective is to make the NLP contribution identifiable.
