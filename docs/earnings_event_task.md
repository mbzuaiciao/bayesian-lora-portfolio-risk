# Earnings-Event Dataset and Task Plan

## Why pivot from daily news

The FNSPID pilot failed the deterministic incremental-information gate despite relevance filtering, longer article text, multiple prediction horizons, and chunked semantic representations. The next task therefore uses a clearly defined disclosure event with a stronger informational relationship to subsequent asset behavior.

## Preferred public data source for the pilot

Use an S&P 500 earnings-episode dataset that provides one row per company-quarter with long-form earnings text, pre-event market context, and post-event labels. The current pilot source is `RudrakshNanavaty/earnings-call-data`.

The initial 20,000-row audit found:

- 399 unique symbols,
- event coverage from 2005-10-13 through 2025-05-15,
- 99.57% nonempty earnings transcripts,
- median nonempty transcript length of 53,757 characters,
- event-localized D-1, D+1, and D+30 OHLC anchors.

A separate label-consistency audit reconstructed `move_1d`, `move_30d`, and `gap_open_d1` directly from those OHLC anchors. All three matched the supplied values to floating-point precision, with correlation and sign agreement equal to 1.0 on 18,201 valid episodes.

This remains a third-party assembled research dataset. Before relying on it for a final result, provenance, event-time semantics, survivorship/universe construction, and source licensing should still be documented.

## Observation unit

One company earnings episode:

\[
(i,q,t_e,D_{i,q})
\]

where:

- \(i\) is the company,
- \(q\) is the reporting quarter,
- \(t_e\) is the disclosure/call event timestamp or conservative event date,
- \(D_{i,q}\) is the transcript and/or earnings release known at that event.

Do not aggregate unrelated daily news into the episode.

## Locked Gate 0 target

The first deterministic gate asks whether the earnings transcript helps identify **large negative next-session reactions** beyond pre-event market context.

Use the independently verified close-to-close earnings reaction:

\[
r^{(1)}_{i,q} = \frac{P_{i,D+1}^{close}}{P_{i,D-1}^{close}} - 1.
\]

Define the downside-event label using a threshold estimated from the training period only:

\[
y_{i,q}=\mathbf{1}[r^{(1)}_{i,q} \le \tau],
\]

where \(\tau\) is the training-set return quantile corresponding to a target positive rate of 20% by default. Freeze \(\tau\) before evaluating later periods.

Why start here:

1. the label is directly reconstructible from audited OHLC anchors;
2. the event window is tightly linked to the disclosure;
3. a downside-tail task is closer to portfolio risk than raw up/down direction;
4. it avoids inventing a path-dependent drawdown label when the current dataset only provides sparse future price anchors.

A later milestone may construct 5- or 10-day path-dependent drawdown or abnormal-return targets from an independent daily-price source. That is not required for Gate 0.

## Point-in-time rules

1. Text must be available by the event timestamp/date used for the episode.
2. If precise call timing is unavailable, use a conservative next-trading-session anchor.
3. Structured features must be explicitly pre-event. For Gate 0 use only fields documented as pre-event market context.
4. Do not use `gap_open_d1`, D+1/D+30 prices, supplied sentiment labels, or any other post-event-derived field as a predictor.
5. Episodes without a valid `move_1d` are excluded rather than labeled negative.
6. The downside threshold is estimated on the training period only.

## Chronological Gate 0 split

Use a fixed chronological split before examining baseline results:

- Train: event date through 2021-12-31.
- Test: 2022-01-01 through 2023-12-31.

A later robustness check should evaluate 2024-2025 separately rather than using it to tune the first specification.

## Gate 0 baselines

### Structured baseline

Start deliberately small with dataset fields explicitly representing pre-event market context:

- `price_momentum_30d`,
- `price_momentum_90d`,
- `pct_from_52w_high_pt`,
- log-transformed `avg_volume_20d`.

Do not use event-derived text annotations such as `guidance_mentioned`, `beat_mentioned`, or `sentiment_label` in the structured baseline.

### Text baseline

The first text diagnostic is TF-IDF logistic regression over earnings transcripts. This is intentionally cheaper than a frozen transformer or LoRA model. To avoid reducing each long transcript to only its beginning, retain both the beginning and end of very long transcripts before vectorization; this usually preserves prepared remarks plus later Q&A material.

Compare:

1. constant,
2. structured only,
3. TF-IDF text only,
4. TF-IDF text + structured.

The key quantity is the incremental discrimination and probabilistic quality from adding transcript information to structured pre-event context on the chronological test period.

## Gate 0 success criterion

Proceed to a frozen semantic baseline and then standard LoRA only if text provides a credible incremental signal beyond the structured baseline. Do not require a dramatic AUROC increase, but require improvement to be visible in AUROC and/or proper scoring rules and to survive a later-period robustness slice.

If TF-IDF adds no signal, inspect target/event-time construction and transcript representation before using a more expensive model. If a frozen semantic model also fails, stop before LoRA.

## Next implementation steps

1. Run the locked 1-day downside-tail Gate 0 baseline.
2. Inspect class balance, learned training threshold, and structured-vs-text incremental metrics.
3. If text adds signal, run a frozen financial-language-model baseline with explicit long-context handling.
4. Repeat the winning deterministic specification on a later 2024-2025 robustness slice.
5. Only then decide whether standard LoRA is scientifically justified.
