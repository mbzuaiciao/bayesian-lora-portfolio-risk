# Earnings-Event Dataset and Task Plan

## Why pivot from daily news

The FNSPID pilot failed the deterministic incremental-information gate despite relevance filtering, longer article text, multiple prediction horizons, and chunked semantic representations. The next task therefore uses a clearly defined disclosure event with a stronger informational relationship to subsequent asset behavior.

## Preferred public data source for the pilot

Use an S&P 500 earnings-episode dataset that provides one row per company-quarter with long-form earnings text, pre-event market context, and post-event labels. A current candidate is `RudrakshNanavaty/earnings-call-data`, whose published description reports:

- roughly 33k company-quarter episodes,
- coverage from 2005 through 2025,
- full earnings transcripts,
- SEC earnings press materials when available,
- pre-earnings OHLCV / market context,
- post-earnings return labels,
- optional SEC XBRL fundamentals.

This is a third-party assembled research dataset. Before relying on it for a final result, audit provenance, timestamp semantics, survivorship/universe construction, and any derived-label implementation.

A second useful source for transcript-only validation is the public S&P 500 earnings-transcript corpus family (`glopardo` / related maintained mirrors), which contains tens of thousands of earnings-call transcripts across many years.

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

## First target candidate

Prefer a post-event downside-risk target rather than raw return direction.

For a fixed horizon \(h\), define post-event maximum drawdown relative to the first tradable post-disclosure anchor:

\[
DD_{i,q}^{(h)} = \min_{1 \le k \le h}\left(\frac{P_{i,t_e+k}}{P_{i,t_e^+}} - 1\right).
\]

Normalize by pre-event volatility known before the disclosure:

\[
z_{i,q}^{(h)} = \frac{-DD_{i,q}^{(h)}}{\sigma_{i,q}^{\text{pre}}\sqrt{h}}.
\]

Choose an event threshold from the training period only to produce a viable but nontrivial positive class rate. Freeze it before validation/test evaluation.

A five- or ten-trading-day horizon is a reasonable first diagnostic; horizon choice should be treated as an auditable design decision rather than tuned on the final test set.

## Point-in-time rules

1. Text must be available by the event timestamp/date used for the episode.
2. If precise call timing is unavailable, use a conservative next-trading-session anchor.
3. Pre-event volatility and return features must end before the event.
4. Post-event target windows must begin strictly after the chosen tradable anchor.
5. Episodes lacking a complete future horizon are censored, not labeled negative.
6. Any source-provided labels must be independently checked against raw price anchors for a sample before use.

## Gate 0 baselines

Before any LoRA training, compare:

### Structured baseline

Start deliberately small:

- trailing realized volatility,
- recent return / momentum,
- optional pre-event volume surprise if clearly point-in-time.

Do not add large feature sets simply to maximize baseline performance.

### Text baseline

Use a frozen financial-language-model representation. Because earnings transcripts are long, the baseline must explicitly handle long context, either through chunking/pooling or a model designed for longer context.

Compare:

1. constant,
2. structured only,
3. text only,
4. text + structured.

The key quantity is incremental discrimination and probabilistic quality from adding text to structured information on a chronological test period.

## Gate 0 success criterion

Proceed to standard LoRA only if text provides a credible, repeatable incremental signal beyond the structured baseline. Do not require a dramatic AUROC increase, but require the direction of improvement to be robust across more than one time slice or reasonable specification.

If Gate 0 fails again, reconsider the target or document type before introducing Bayesian modeling.

## Next implementation steps

1. Inspect the candidate earnings-episode dataset schema and provenance.
2. Materialize a small pilot slice locally.
3. Audit event-date/timestamp semantics and raw text quality.
4. Recompute at least one post-event label independently from prices.
5. Define a chronological train/test split before examining model results.
6. Run structured and frozen-text baselines.
7. Only then decide whether standard LoRA is scientifically justified.
