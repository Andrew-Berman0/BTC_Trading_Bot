"""
data/pipeline.py
----------------
Orchestrates the full data pipeline end-to-end:
  1. Fetch / update OHLCV from exchange
  2. Engineer features
  3. Return train/val splits ready for a model

Also provides a live_snapshot() method for the inference loop.

Usage:
    pipe = DataPipeline(symbol="BTC/USDT", timeframe="1h")

    # One-time historical build
    X_train, X_val, y_train, y_val = pipe.build_dataset(days_back=730)

    # Incremental update (append new candles)
    pipe.update_dataset()

    # Live inference snapshot
    X_live = pipe.live_snapshot()
"""

import logging
from pathlib import Path

import pandas as pd

from data.fetcher import OHLCVFetcher
from features.engineer import FeatureEngineer

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    End-to-end pipeline: raw data → feature matrix → train/val splits.

    Parameters
    ----------
    symbol      : e.g. 'BTC/USDT'
    timeframe   : candle interval, e.g. '1h'
    exchange_id : any ccxt exchange id
    forward_n   : candles ahead for target construction
    direction_threshold : minimum return to be classified as BUY/SELL
    api_key / api_secret : exchange credentials (optional for public data)
    """

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        exchange_id: str = "binance",
        forward_n: int = 4,
        direction_threshold: float = 0.005,
        api_key: str | None = None,
        api_secret: str | None = None,
    ):
        self.symbol    = symbol
        self.timeframe = timeframe
        self.name      = f"{symbol.replace('/', '_')}_{timeframe}"

        self.fetcher = OHLCVFetcher(
            exchange_id=exchange_id,
            api_key=api_key,
            api_secret=api_secret,
        )
        self.engineer = FeatureEngineer(
            symbol=symbol,
            timeframe=timeframe,
            forward_n=forward_n,
            direction_threshold=direction_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_dataset(
        self,
        days_back: int = 730,
        target: str = "target_direction",
        val_frac: float = 0.2,
        scale: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Full historical build from scratch.

        Returns
        -------
        X_train, X_val, y_train, y_val
        """
        logger.info(f"Building dataset: {self.symbol} {self.timeframe} ({days_back}d)")

        # 1. Fetch
        raw = self.fetcher.fetch_historical(
            self.symbol, self.timeframe, days_back=days_back
        )
        self.fetcher.save(raw, self.name)

        # 2. Engineer
        featured = self.engineer.build(raw)

        # 3. Split
        X_train, X_val, y_train, y_val = self.engineer.split(
            featured, target=target, val_frac=val_frac
        )

        # 4. Scale (optional but recommended for LSTM)
        if scale:
            X_train, X_val = self.engineer.scale(X_train, X_val)

        self._log_class_balance(y_train, "train")
        self._log_class_balance(y_val,   "val")

        return X_train, X_val, y_train, y_val

    def update_dataset(
        self,
        target: str = "target_direction",
        val_frac: float = 0.2,
        scale: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Append only new candles to the existing saved dataset, then rebuild
        features and return fresh splits. Call this on a schedule (e.g. daily).
        """
        logger.info(f"Updating dataset: {self.name}")
        raw = self.fetcher.update(self.symbol, self.timeframe, self.name)
        featured = self.engineer.build(raw)
        X_train, X_val, y_train, y_val = self.engineer.split(
            featured, target=target, val_frac=val_frac
        )
        if scale:
            X_train, X_val = self.engineer.scale(X_train, X_val)
        return X_train, X_val, y_train, y_val

    def live_snapshot(self, lookback_candles: int = 500) -> pd.DataFrame:
        """
        Fetch the latest N candles, compute features, and return the
        most recent row — ready to feed into a trained model for inference.

        Returns
        -------
        pd.DataFrame with shape (1, n_features)
        """
        raw = self.fetcher.fetch_latest(
            self.symbol, self.timeframe, lookback_candles=lookback_candles
        )
        X_live = self.engineer.transform_live(raw)
        logger.info(f"Live snapshot at {X_live.index[-1]} — {X_live.shape[1]} features")
        return X_live

    def get_feature_names(self) -> list[str]:
        return self.engineer.feature_cols

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_class_balance(self, y: pd.Series, split: str):
        counts = y.value_counts().sort_index()
        total  = len(y)
        dist   = {k: f"{v} ({v/total:.1%})" for k, v in counts.items()}
        logger.info(f"Class balance [{split}]: {dist}")
