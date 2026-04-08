"""
run_pipeline.py
---------------
Demo script — runs the full data pipeline and prints a summary.
Run from the project root:

    pip install ccxt pandas numpy scikit-learn pyarrow
    python run_pipeline.py

For a quick offline test (no API call), set USE_SYNTHETIC=True below.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---- local imports ----
sys.path.insert(0, str(Path(__file__).parent))
from config import CONFIG
from features.engineer import FeatureEngineer

DATASET_DIR = Path(__file__).parent / "data" / "processed"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# ---- logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_pipeline")

# ---- flip to True to run without hitting an exchange ----
USE_SYNTHETIC = False


def make_synthetic_ohlcv(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic-ish synthetic BTC/USDT 1h candles for offline testing.
    Uses geometric Brownian motion with a drift and fat-tailed noise.
    """
    rng = np.random.default_rng(seed)
    dt  = 1 / (24 * 365)         # 1h in years
    mu  = 0.5                    # annualised drift
    sigma = 0.8                  # annualised vol

    # Geometric Brownian motion
    shocks = rng.standard_t(df=4, size=n) * sigma * np.sqrt(dt)
    log_prices = np.cumsum(mu * dt + shocks) + np.log(30_000)
    prices = np.exp(log_prices)

    # Build OHLCV
    noise = rng.uniform(0.995, 1.005, size=(n, 4))
    high  = prices * noise[:, 0].clip(min=1.001)
    low   = prices * noise[:, 1].clip(max=0.999)
    open_ = prices * noise[:, 2]
    vol   = rng.lognormal(mean=10, sigma=0.5, size=n) * (1 + np.abs(shocks) * 10)

    index = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": prices, "volume": vol},
        index=index,
    )
    # Ensure OHLC consistency
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"]  = df[["open", "low",  "close"]].min(axis=1)
    return df


def main():
    logger.info("=" * 60)
    logger.info("Crypto ML Trading Bot — Data Pipeline Demo")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Raw data
    # ------------------------------------------------------------------
    if USE_SYNTHETIC:
        logger.info("Using synthetic data (USE_SYNTHETIC=True)")
        raw = make_synthetic_ohlcv(n=5000)
    else:
        logger.info(f"Fetching live data: {CONFIG.data.symbol} {CONFIG.data.timeframe}")
        from data.pipeline import DataPipeline
        pipe = DataPipeline(
            symbol=CONFIG.data.symbol,
            timeframe=CONFIG.data.timeframe,
            exchange_id=CONFIG.data.exchange,
            forward_n=CONFIG.data.forward_n,
            direction_threshold=CONFIG.data.threshold,
        )
        X_train, X_val, y_train, y_val = pipe.build_dataset(
            days_back=CONFIG.data.days_back,
            val_frac=CONFIG.data.val_frac,
        )
        _print_summary(X_train, X_val, y_train, y_val)
        return

    logger.info(f"Raw data shape: {raw.shape}")
    logger.info(f"  Date range:  {raw.index[0]}  →  {raw.index[-1]}")
    logger.info(f"  Price range: ${raw['close'].min():,.0f} – ${raw['close'].max():,.0f}")

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    eng = FeatureEngineer(
        symbol=CONFIG.data.symbol,
        timeframe=CONFIG.data.timeframe,
        forward_n=CONFIG.data.forward_n,
        direction_threshold=CONFIG.data.threshold,
    )
    featured = eng.build(raw)

    logger.info(f"\nFeatured data shape: {featured.shape}")
    logger.info(f"  Feature columns: {len(eng.feature_cols)}")

    # ------------------------------------------------------------------
    # 3. Train / val split
    # ------------------------------------------------------------------
    X_train, X_val, y_train, y_val = eng.split(featured, val_frac=CONFIG.data.val_frac)
    X_train_s, X_val_s = eng.scale(X_train, X_val)

    _print_summary(X_train_s, X_val_s, y_train, y_val)

    # ------------------------------------------------------------------
    # 4. Save dataset to disk
    # ------------------------------------------------------------------
    tag = "synthetic" if USE_SYNTHETIC else f"{CONFIG.data.symbol.replace('/', '_')}_{CONFIG.data.timeframe}"
    X_train_s.to_parquet(DATASET_DIR / f"{tag}_X_train.parquet")
    X_val_s.to_parquet(  DATASET_DIR / f"{tag}_X_val.parquet")
    y_train.to_frame().to_parquet(DATASET_DIR / f"{tag}_y_train.parquet")
    y_val.to_frame().to_parquet(  DATASET_DIR / f"{tag}_y_val.parquet")
    # Also save the full featured frame for reference / reuse
    featured.to_parquet(DATASET_DIR / f"{tag}_featured_full.parquet")
    logger.info(f"\nDataset saved → {DATASET_DIR}/")
    logger.info(f"  {tag}_X_train.parquet  {tag}_X_val.parquet")
    logger.info(f"  {tag}_y_train.parquet  {tag}_y_val.parquet")
    logger.info(f"  {tag}_featured_full.parquet")

    # ------------------------------------------------------------------
    # 5. Feature preview
    # ------------------------------------------------------------------
    logger.info("\nTop 10 features (first 3 train rows):")
    preview_cols = eng.feature_cols[:10]
    print(X_train_s[preview_cols].head(3).to_string())

    logger.info("\nTarget distribution (train):")
    dist = y_train.value_counts().sort_index()
    for cls, cnt in dist.items():
        label = {-1: "SELL", 0: "HOLD", 1: "BUY"}.get(cls, str(cls))
        bar   = "█" * int(cnt / len(y_train) * 40)
        print(f"  {label:4s} ({cls:+d}): {cnt:5d}  {bar}  {cnt/len(y_train):.1%}")

    logger.info("\nPipeline complete. Next step: train a model on X_train / y_train.")
    logger.info("Run: python train_model.py")


def _print_summary(X_train, X_val, y_train, y_val):
    logger.info("\n--- Dataset Summary ---")
    logger.info(f"  X_train: {X_train.shape}   y_train: {y_train.shape}")
    logger.info(f"  X_val:   {X_val.shape}     y_val:   {y_val.shape}")
    logger.info(f"  Features: {X_train.shape[1]}")
    logger.info(f"  Train period: {X_train.index[0].date()} → {X_train.index[-1].date()}")
    logger.info(f"  Val period:   {X_val.index[0].date()}   → {X_val.index[-1].date()}")


if __name__ == "__main__":
    main()
