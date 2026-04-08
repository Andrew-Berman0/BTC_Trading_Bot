"""
data/fng_fetcher.py
-------------------
Fetches the Crypto Fear & Greed Index from alternative.me (free, no API key).

The index is published daily as a value 0-100:
  0-25  = Extreme Fear
  25-45 = Fear
  45-55 = Neutral
  55-75 = Greed
  75-100 = Extreme Greed

We derive several features from it:
  fng_value      : raw 0-100 score
  fng_norm       : normalized to [-1, 1]  (fear = negative, greed = positive)
  fng_change_1d  : 1-day change in score (momentum)
  fng_change_7d  : 7-day change (trend)
  fng_ma_7       : 7-day rolling mean (smoothed regime)
  fng_extreme    : 1 if extreme fear (<25) or extreme greed (>75), else 0
                   (extremes are contrarian signals)
"""

import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

FNG_URL  = "https://api.alternative.me/fng/"
FNG_FILE = DATA_DIR / "fng.parquet"
FNG_MAX_DAYS = 0   # 0 = fetch all available history


class FearGreedFetcher:

    def fetch(self, days: int = 3000) -> pd.DataFrame:
        """
        Fetch Fear & Greed history. Default 3000 covers all available data (back to 2018-02-01).
        Returns a daily DatetimeIndex DataFrame with fng_value column.
        """
        logger.info(f"Fetching Fear & Greed Index ({days} days)...")
        for attempt in range(3):
            try:
                params = {"limit": days, "format": "json"}
                resp = requests.get(FNG_URL, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()["data"]
                break
            except Exception as e:
                logger.warning(f"FNG fetch attempt {attempt+1} failed: {e}")
                time.sleep(2)
        else:
            raise RuntimeError("Failed to fetch Fear & Greed Index after 3 attempts")

        df = pd.DataFrame(data)[["value", "timestamp"]]
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
        df["value"]     = df["value"].astype(int)
        df = df.rename(columns={"value": "fng_value"})
        df = df.set_index("timestamp").sort_index()

        # Normalize to daily frequency (one row per day)
        df = df[~df.index.duplicated(keep="last")]
        df.index = df.index.normalize()   # floor to midnight UTC

        logger.info(f"Fetched {len(df)} days of F&G data ({df.index[0].date()} → {df.index[-1].date()})")
        df.to_parquet(FNG_FILE)
        return df

    def load_or_fetch(self) -> pd.DataFrame:
        """Load cached data if fresh (updated today), otherwise re-fetch all history."""
        if FNG_FILE.exists():
            cached = pd.read_parquet(FNG_FILE)
            last   = cached.index[-1].date()
            today  = pd.Timestamp.utcnow().date()
            if last >= today:
                logger.info(f"F&G cache is current (last={last}, {len(cached)} days)")
                return cached
            logger.info(f"F&G cache stale (last={last}), refreshing...")
        return self.fetch()

    def align_to_ohlcv(self, fng: pd.DataFrame, ohlcv_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Reindex daily F&G data to match a higher-frequency OHLCV index.
        Uses forward-fill — each 4h candle gets the most recent daily F&G value.
        Then derives trading features from the aligned series.
        """
        # Reindex: forward-fill daily value into each candle
        fng_aligned = fng["fng_value"].reindex(
            fng["fng_value"].index.union(ohlcv_index)
        ).ffill().reindex(ohlcv_index)

        out = pd.DataFrame(index=ohlcv_index)
        out["fng_value"]     = fng_aligned
        out["fng_norm"]      = (fng_aligned - 50) / 50          # [-1, 1]
        out["fng_change_1d"] = fng_aligned.diff(1)              # daily momentum
        out["fng_change_7d"] = fng_aligned.diff(7)              # weekly trend
        out["fng_ma_7"]      = fng_aligned.rolling(7).mean()    # smoothed regime
        out["fng_extreme"]   = (
            (fng_aligned <= 25) | (fng_aligned >= 75)
        ).astype(int)

        return out
