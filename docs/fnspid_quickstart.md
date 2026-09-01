# FNSPID Milestone 0 Quickstart

The official FNSPID project hosts the large dataset files on Hugging Face. For the first audit, do not load the whole corpus into memory.

## Expected local layout

```text
data/FNSPID/
├── Stock_news/
│   └── nasdaq_exteral_data.csv
└── Stock_price/
    └── full_history/
        ├── AAPL.csv
        ├── MSFT.csv
        └── ...
```

The `data/FNSPID/` directory is gitignored.

## Official downloads

From the repository root:

```bash
mkdir -p data/FNSPID/Stock_news data/FNSPID/Stock_price

wget -O data/FNSPID/Stock_price/full_history.zip \
  https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_price/full_history.zip

wget -O data/FNSPID/Stock_news/nasdaq_exteral_data.csv \
  https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv

unzip data/FNSPID/Stock_price/full_history.zip -d data/FNSPID/Stock_price/
```

The news CSV is very large. The repository loader reads it in chunks and retains only requested tickers and dates.

## First recommended audit slice

Start with a deliberately small cross-sector set:

- AAPL — technology
- MSFT — technology
- JPM — financials
- XOM — energy
- JNJ — health care
- WMT — consumer staples
- CAT — industrials
- NEE — utilities

Use 2018 through mid-2023 so the slice contains both relatively ordinary conditions and the COVID/regime-change period. The exact train/ID/shift split is **not** locked until the audit output is inspected.

```bash
uv sync

uv run python scripts/audit_fnspid_slice.py \
  --news data/FNSPID/Stock_news/nasdaq_exteral_data.csv \
  --prices data/FNSPID/Stock_price/full_history \
  --tickers AAPL MSFT JPM XOM JNJ WMT CAT NEE \
  --start 2018-01-01 \
  --end 2023-06-30 \
  --horizon 20 \
  --threshold 0.10 \
  --output artifacts/fnspid_slice
```

For a smoke test, add `--max-news-rows 5000`.

## Outputs

The command writes:

```text
artifacts/fnspid_slice/
├── model_ready.csv
├── summary.json
├── yearly_label_summary.csv
└── ticker_label_summary.csv
```

Before model training, inspect:

1. total model-ready rows,
2. positive drawdown-event rate,
3. event rate by calendar year,
4. event rate and coverage by ticker,
5. whether the 10% threshold creates a usable training class balance.

Do not adjust the threshold based on the eventual temporal-shift test set.
