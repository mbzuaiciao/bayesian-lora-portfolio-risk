"""Sweep downside horizons and test whether text adds signal beyond volatility.

This is a cheap pre-LoRA diagnostic. It rebuilds volatility-normalized labels for
multiple forward horizons from an existing model-ready relevance-filtered dataset,
then evaluates TF-IDF text, trailing volatility, and their combination on the same
chronological split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bayesian_lora_portfolio_risk.data.fnspid import load_fnspid_price_history
from bayesian_lora_portfolio_risk.data.labels import forward_max_drawdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--horizons", nargs="+", type=int, default=[3, 5, 10, 20])
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--test-end", required=True)
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--target-positive-rate", type=float, default=0.125)
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--output", type=Path, default=Path("artifacts/horizon_sweep.json"))
    return parser.parse_args()


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    result = {
        "brier": float(brier_score_loss(y, p)),
        "nll": float(log_loss(y, np.column_stack([1.0 - p, p]), labels=[0, 1])),
        "auroc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan"),
    }
    return result


def add_price_features(frame: pd.DataFrame, prices_root: Path, horizons: list[int], vol_window: int) -> pd.DataFrame:
    tickers = sorted(frame["ticker"].astype(str).str.upper().unique())
    prices = load_fnspid_price_history(prices_root, tickers)
    prices = prices.sort_values(["ticker", "date"], kind="stable").copy()
    prices["return"] = prices.groupby("ticker", sort=False)["close"].pct_change()
    prices["trailing_vol"] = (
        prices.groupby("ticker", sort=False)["return"]
        .rolling(window=vol_window, min_periods=vol_window)
        .std()
        .reset_index(level=0, drop=True)
    )
    prices["trailing_vol"] = prices.groupby("ticker", sort=False)["trailing_vol"].shift(1)

    parts: list[pd.DataFrame] = []
    for _, group in prices.groupby("ticker", sort=False):
        group = group.copy()
        values = group["close"].to_numpy(dtype=float)
        for horizon in horizons:
            group[f"forward_drawdown_{horizon}"] = forward_max_drawdown(values, horizon=horizon)
        parts.append(group)
    price_features = pd.concat(parts, ignore_index=True)

    keep = ["ticker", "date", "trailing_vol"] + [f"forward_drawdown_{h}" for h in horizons]
    merged = frame.merge(price_features[keep], on=["ticker", "date"], how="left", validate="many_to_one")
    return merged


def fit_eval(train: pd.DataFrame, test: pd.DataFrame, label: str, max_features: int) -> dict[str, object]:
    y_train = train[label].to_numpy()
    y_test = test[label].to_numpy()

    constant_p = np.full(len(test), float(np.mean(y_train)))

    vol_model = Pipeline([
        ("prep", ColumnTransformer([("vol", StandardScaler(), ["trailing_vol"])])),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    vol_model.fit(train[["trailing_vol"]], y_train)
    vol_p = vol_model.predict_proba(test[["trailing_vol"]])[:, 1]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=max_features,
        strip_accents="unicode",
        sublinear_tf=True,
    )
    x_train_text = vectorizer.fit_transform(train["text"].fillna("").astype(str))
    x_test_text = vectorizer.transform(test["text"].fillna("").astype(str))
    text_model = LogisticRegression(max_iter=1000)
    text_model.fit(x_train_text, y_train)
    text_p = text_model.predict_proba(x_test_text)[:, 1]

    from scipy import sparse

    vol_train = StandardScaler().fit_transform(train[["trailing_vol"]])
    scaler = StandardScaler().fit(train[["trailing_vol"]])
    vol_train = scaler.transform(train[["trailing_vol"]])
    vol_test = scaler.transform(test[["trailing_vol"]])
    combo_train = sparse.hstack([x_train_text, sparse.csr_matrix(vol_train)], format="csr")
    combo_test = sparse.hstack([x_test_text, sparse.csr_matrix(vol_test)], format="csr")
    combo_model = LogisticRegression(max_iter=1000)
    combo_model.fit(combo_train, y_train)
    combo_p = combo_model.predict_proba(combo_test)[:, 1]

    return {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_positive_rate": float(np.mean(y_train)),
        "test_positive_rate": float(np.mean(y_test)),
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "constant": metrics(y_test, constant_p),
        "volatility_only": metrics(y_test, vol_p),
        "text_only": metrics(y_test, text_p),
        "text_plus_volatility": metrics(y_test, combo_p),
    }


def main() -> None:
    args = parse_args()
    if not args.dataset.is_file():
        raise SystemExit(f"Dataset not found: {args.dataset}")
    if not args.prices.is_dir():
        raise SystemExit(f"Price directory not found: {args.prices}")
    if any(h <= 0 for h in args.horizons):
        raise ValueError("all horizons must be positive")
    if not (0.0 < args.target_positive_rate < 1.0):
        raise ValueError("target-positive-rate must lie in (0, 1)")

    frame = pd.read_csv(args.dataset)
    required = {"ticker", "date", "text"}
    missing = required.difference(frame.columns)
    if missing:
        raise SystemExit(f"Dataset missing columns: {sorted(missing)}")
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = add_price_features(frame, args.prices, args.horizons, args.vol_window)

    train_end = pd.Timestamp(args.train_end)
    test_start = pd.Timestamp(args.test_start)
    test_end = pd.Timestamp(args.test_end)
    if not (train_end < test_start <= test_end):
        raise ValueError("require train-end < test-start <= test-end")

    output: dict[str, object] = {
        "dataset": str(args.dataset),
        "horizons": args.horizons,
        "target_positive_rate": args.target_positive_rate,
        "results": {},
    }

    table_rows: list[dict[str, object]] = []
    for horizon in args.horizons:
        dd_col = f"forward_drawdown_{horizon}"
        score_col = f"normalized_score_{horizon}"
        label_col = f"normalized_label_{horizon}"

        work = frame.dropna(subset=["trailing_vol", dd_col]).copy()
        work = work[work["trailing_vol"] > 0].copy()
        work[score_col] = -work[dd_col] / (work["trailing_vol"] * np.sqrt(float(horizon)))

        train_mask = work["date"] <= train_end
        threshold = float(work.loc[train_mask, score_col].quantile(1.0 - args.target_positive_rate))
        work[label_col] = (work[score_col] >= threshold).astype(int)

        train = work[work["date"] <= train_end].copy()
        test = work[(work["date"] >= test_start) & (work["date"] <= test_end)].copy()
        if train.empty or test.empty:
            continue

        result = fit_eval(train, test, label_col, args.max_features)
        result["threshold"] = threshold
        output["results"][str(horizon)] = result
        table_rows.append({
            "horizon": horizon,
            "train_pos": result["train_positive_rate"],
            "test_pos": result["test_positive_rate"],
            "vol_auroc": result["volatility_only"]["auroc"],
            "text_auroc": result["text_only"]["auroc"],
            "text_plus_vol_auroc": result["text_plus_volatility"]["auroc"],
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=True) + "\n")
    table = pd.DataFrame(table_rows)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nArtifact written to {args.output}")


if __name__ == "__main__":
    main()
