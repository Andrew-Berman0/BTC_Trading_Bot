"""
tune_hyperparams.py
-------------------
Bayesian hyperparameter search for the LightGBM model using Optuna.

Uses walk-forward cross-validation to avoid future leakage:
  - Data is split into N time-ordered folds
  - Each fold trains on all prior data, validates on the next window
  - MCC is averaged across folds (primary objective)

Run on the server:
    source venv/bin/activate
    pip install optuna
    python3 tune_hyperparams.py

Results are saved to hyperparams_best.json.
Paste the winning params into models/trainer.py.
"""

import json
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import matthews_corrcoef

from config import CONFIG
from data.alpaca_fetcher import AlpacaFetcher
from features.engineer import FeatureEngineer
from models.trainer import LABEL_MAP, LABEL_MAP_INV

logging.basicConfig(level=logging.WARNING)   # suppress noise during search
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger("tune")
logger.setLevel(logging.INFO)

SYMBOL    = "BTC/USD"
TIMEFRAME = CONFIG.data.timeframe
FORWARD_N = CONFIG.data.forward_n
THRESHOLD = CONFIG.data.threshold
N_TRIALS  = 100     # increase for a more thorough search
N_FOLDS   = 5       # walk-forward folds
MIN_TRAIN_FRAC = 0.5  # first fold uses at least 50% of data for training


def load_features() -> tuple[pd.DataFrame, pd.Series]:
    """Load cached data and build features. Returns (X, y)."""
    fetcher = AlpacaFetcher()
    name    = f"{SYMBOL.replace('/', '_')}_{TIMEFRAME}"
    try:
        raw = fetcher.load(name)
    except FileNotFoundError:
        logger.info("No cache — fetching fresh data...")
        raw = fetcher.fetch_historical(SYMBOL, TIMEFRAME, days_back=CONFIG.data.days_back)
        fetcher.save(raw, name)

    eng = FeatureEngineer(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        forward_n=FORWARD_N,
        direction_threshold=THRESHOLD,
    )
    featured = eng.build(raw)
    X = featured[eng.feature_cols]
    y = featured["target_direction"]
    return X, y


def walk_forward_folds(
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = N_FOLDS,
    min_train_frac: float = MIN_TRAIN_FRAC,
) -> list[tuple]:
    """
    Generate (X_train, X_val, y_train, y_val) tuples for walk-forward CV.
    Each fold expands the training window; val window is the next chunk.
    """
    n = len(X)
    # Val window size: divide remaining data (after min_train) into n_folds chunks
    remaining = int(n * (1 - min_train_frac))
    val_size  = remaining // n_folds
    min_train = n - remaining

    folds = []
    for i in range(n_folds):
        train_end = min_train + i * val_size
        val_end   = train_end + val_size
        if val_end > n:
            break
        X_tr = X.iloc[:train_end]
        X_v  = X.iloc[train_end:val_end]
        y_tr = y.iloc[:train_end]
        y_v  = y.iloc[train_end:val_end]
        folds.append((X_tr, X_v, y_tr, y_v))

    return folds


def evaluate_params(params: dict, folds: list) -> float:
    """Train LightGBM with given params on each fold, return mean MCC."""
    mccs = []

    for X_tr, X_v, y_tr, y_v in folds:
        y_tr_mapped = y_tr.map(LABEL_MAP)
        y_v_mapped  = y_v.map(LABEL_MAP)

        model = lgb.LGBMClassifier(
            n_estimators=2000,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            **params,
        )
        model.fit(
            X_tr, y_tr_mapped,
            eval_set=[(X_v, y_v_mapped)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=9999),   # silence per-round logs
            ],
        )

        preds = pd.Series(model.predict(X_v)).map(LABEL_MAP_INV)
        mcc   = float(matthews_corrcoef(y_v.values, preds.values))
        mccs.append(mcc)

    return float(np.mean(mccs))


def objective(trial: optuna.Trial, folds: list) -> float:
    params = {
        "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves":         trial.suggest_int("num_leaves", 15, 127),
        "max_depth":          trial.suggest_int("max_depth", 3, 9),
        "min_child_samples":  trial.suggest_int("min_child_samples", 20, 150),
        "subsample":          trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_gain_to_split":  trial.suggest_float("min_gain_to_split", 1e-4, 0.1, log=True),
        "reg_alpha":          trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":         trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }
    return evaluate_params(params, folds)


def main():
    logger.info("Loading features...")
    X, y = load_features()
    logger.info(f"Dataset: {len(X):,} rows × {len(X.columns)} features")

    folds = walk_forward_folds(X, y)
    logger.info(f"Walk-forward folds: {len(folds)}")
    for i, (X_tr, X_v, y_tr, y_v) in enumerate(folds):
        logger.info(
            f"  Fold {i+1}: train={len(X_tr):,} ({X_tr.index[0].date()}→{X_tr.index[-1].date()})  "
            f"val={len(X_v):,} ({X_v.index[0].date()}→{X_v.index[-1].date()})"
        )

    logger.info(f"\nStarting Optuna search — {N_TRIALS} trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, folds),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best = study.best_trial
    logger.info(f"\n{'='*60}")
    logger.info(f"Best MCC (mean across {len(folds)} folds): {best.value:.4f}")
    logger.info(f"Best params:")
    for k, v in best.params.items():
        logger.info(f"  {k}: {v}")

    # Save results
    out = {
        "best_mcc":    round(best.value, 4),
        "best_params": best.params,
        "n_trials":    N_TRIALS,
        "n_folds":     len(folds),
    }
    with open("hyperparams_best.json", "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"\nSaved → hyperparams_best.json")
    logger.info("Paste best_params into models/trainer.py to use them.")

    # Show top 10 trials for reference
    logger.info(f"\nTop 10 trials by MCC:")
    trials_df = study.trials_dataframe().sort_values("value", ascending=False)
    print(trials_df[["number", "value"] + [c for c in trials_df.columns if c.startswith("params_")]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
