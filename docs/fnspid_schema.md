# FNSPID Ingestion Schema

Milestone 0 keeps the raw dataset format separate from the modeling data contract.

## Source dataset

FNSPID provides financial-news records and historical stock-price data. The public news release uses fields including:

- `Date`
- `Article_title`
- `Stock_symbol`
- `Url`
- `Publisher`
- `Author`
- `Article`
- several precomputed summary fields in some releases

The project does **not** use the supplied sentiment scores as model inputs or labels. The purpose is to learn a risk prediction directly from text and compare deterministic versus Bayesian LoRA uncertainty.

## Normalized news schema

Before entering the core builder, raw source columns are converted to:

| Field | Meaning |
| --- | --- |
| `ticker` | uppercase stock symbol |
| `timestamp` | timezone-aware article timestamp |
| `text` | bounded text selected from title/summary/body |

The core data builder depends only on this normalized schema, so alternative source formats can be substituted without changing label logic.

### Initial text policy

Use the article title first. A later experiment may concatenate a bounded body excerpt, but the first dataset audit should begin with titles because they are substantially cheaper to train on and make coverage/duplication easier to inspect.

When a row has both title and body, the raw ingestion code should retain both fields until the audit is complete even if only title is used for the first model.

## Point-in-time assignment

All timestamps are normalized to UTC on ingestion and converted to `America/New_York` for the market-information cutoff.

Initial policy:

- before 4:00 p.m. ET: eligible for that calendar day's information set,
- at/after 4:00 p.m. ET: assign to the following calendar day,
- weekend/holiday dates: align forward to the next observed trading day for that ticker.

This policy is intentionally conservative. If the original source cannot establish a reliable intraday timestamp, ingestion should treat the timestamp as ambiguous and assign the item to the next trading-day information set rather than assuming it was known before the close.

## Asset-day aggregation

Multiple articles for the same ticker and eligible information date are not independent training examples.

They are aggregated into one row:

| Field | Meaning |
| --- | --- |
| `ticker` | asset |
| `information_date` | earliest calendar date at which the text may be used |
| `text` | concatenated bounded article text |
| `article_count` | number of source articles represented |

This prevents heavily covered events from receiving artificial sample multiplicity.

## Normalized price schema

The label builder accepts:

| Field | Meaning |
| --- | --- |
| `ticker` | uppercase stock symbol |
| `date` | trading date |
| `close` | positive closing price |

Each `(ticker, date)` must be unique.

For Milestone 0 we use the source's consistent closing-price series. Before the full experiment, verify whether the downloaded FNSPID history exposes raw close, adjusted close, or both. The chosen series must be fixed before label statistics are inspected and documented in the experiment log.

## Derived fields

For each price observation:

- `forward_drawdown`: minimum return relative to the observation close over the next `horizon` trading observations,
- `label`: one when the forward drawdown is at or below `-threshold`, zero otherwise,
- censored rows without a complete horizon receive no usable label.

Initial configuration:

- horizon: 20 trading days,
- threshold: 10% drawdown.

## Final model-ready row

After forward alignment of news to the trading calendar:

| Field | Meaning |
| --- | --- |
| `ticker` | asset |
| `information_date` | date implied by article availability |
| `date` | trading date used as prediction anchor |
| `text` | asset-day financial text |
| `article_count` | aggregated source count |
| `close` | anchor closing price |
| `forward_drawdown` | realized future downside measure |
| `label` | binary material-drawdown event |

The modeling code should never receive future prices. `forward_drawdown` and `label` are evaluation/training targets only.

## Dataset audit before modeling

The first real FNSPID pull must report, before any LoRA training:

1. total raw news rows read,
2. valid normalized news rows,
3. asset-day observations,
4. ticker coverage,
5. yearly coverage,
6. article count per asset-day,
7. text-length distribution,
8. labeled observations after the complete-horizon requirement,
9. positive-label rate by year and ticker,
10. candidate chronological split sizes.

No model hyperparameters should be selected until this audit is saved as an artifact.
