"""
analyze_features.py
-------------------
Run this once on the server to audit feature collinearity.

    python analyze_features.py

Outputs:
  - Prints all feature pairs with |r| >= 0.85
  - Saves feature_correlation.csv  (full matrix)
  - Saves high_correlation_pairs.csv  (pairs above threshold)
  - Prints LightGBM feature importances from the saved model
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
from config import CONFIG
from data.alpaca_fetcher import AlpacaFetcher
from features.engineer import FeatureEngineer
from models.trainer import ModelTrainer

SYMBOL    = "BTC/USD"
TIMEFRAME = CONFIG.data.timeframe
FORWARD_N = CONFIG.data.forward_n
THRESHOLD = CONFIG.data.threshold
CORR_CUTOFF = 0.85   # flag pairs above this


def main():
    print("Loading data...")
    fetcher = AlpacaFetcher()
    name    = f"{SYMBOL.replace('/', '_')}_{TIMEFRAME}"
    try:
        raw = fetcher.load(name)
    except FileNotFoundError:
        print("No cached data — fetching fresh (this may take a moment)...")
        raw = fetcher.fetch_historical(SYMBOL, TIMEFRAME, days_back=CONFIG.data.days_back)
        fetcher.save(raw, name)

    print("Engineering features...")
    eng = FeatureEngineer(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        forward_n=FORWARD_N,
        direction_threshold=THRESHOLD,
    )
    featured = eng.build(raw)
    X_train, X_val, y_train, _ = eng.split(featured, val_frac=CONFIG.data.val_frac)
    X_train_s, _ = eng.scale(X_train, X_val)

    feature_cols = eng.feature_cols
    print(f"\nTotal features: {len(feature_cols)}")
    print(f"Training rows:  {len(X_train_s)}\n")

    # --- Pearson correlation matrix on unscaled training data ---
    corr = X_train[feature_cols].corr(method="pearson")
    corr.to_csv("feature_correlation.csv")
    print("Saved full correlation matrix → feature_correlation.csv")

    # --- Flag high-correlation pairs ---
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) >= CORR_CUTOFF:
                pairs.append({
                    "feature_a": cols[i],
                    "feature_b": cols[j],
                    "pearson_r": round(r, 4),
                    "abs_r":     round(abs(r), 4),
                })

    if pairs:
        df_pairs = pd.DataFrame(pairs).sort_values("abs_r", ascending=False)
        df_pairs.to_csv("high_correlation_pairs.csv", index=False)
        print(f"\n{'='*60}")
        print(f"HIGH COLLINEARITY PAIRS  (|r| >= {CORR_CUTOFF})  — {len(pairs)} found")
        print(f"{'='*60}")
        print(df_pairs.to_string(index=False))
        print(f"\nSaved → high_correlation_pairs.csv")
    else:
        print(f"\nNo pairs found with |r| >= {CORR_CUTOFF}.")

    # --- Feature importances from saved model ---
    print(f"\n{'='*60}")
    print("LIGHTGBM FEATURE IMPORTANCES (gain)")
    print(f"{'='*60}")
    try:
        trainer = ModelTrainer()
        trainer.load()
        fi = trainer._feature_importance()
        # Show all, marking ones with zero importance
        fi["zero"] = fi["importance"] == 0
        print(fi.to_string(index=False))
        zero_count = fi["zero"].sum()
        if zero_count:
            print(f"\n*** {zero_count} features have ZERO importance — safe to remove ***")
            print(fi[fi["zero"]]["feature"].tolist())
    except Exception as e:
        print(f"Could not load model for importance analysis: {e}")

    # --- Class balance in training set ---
    print(f"\n{'='*60}")
    print("TARGET CLASS DISTRIBUTION (training set)")
    print(f"{'='*60}")
    counts = y_train.value_counts().sort_index()
    total  = len(y_train)
    label_map = {-1: "SELL", 0: "HOLD", 1: "BUY"}
    for label, count in counts.items():
        print(f"  {label_map.get(label, label):4s} ({label:+d}): {count:5d}  ({count/total:.1%})")


if __name__ == "__main__":
    main()
