"""
data/funding_fetcher.py
-----------------------
Fetches BTC perpetual futures funding rates from Bybit's public API.
No API key required. No US geo-restrictions (use from Hetzner/EU server).

Funding rates are published every 8h and represent the cost of holding
a leveraged position. Extreme positive funding = overcrowded longs =
bearish signal. Extreme negative funding = overcrowded shorts = bullish.

Features derived:
  funding_rate      : raw 8h rate (e.g. 0.0001 = 0.01%)
  funding_annualized: annualized rate (rate * 3 * 365)
  funding_ma_3      : 3-period (24h) rolling mean — smoothed positioning
  funding_ma_7      : 7-period (56h) rolling mean — regime
  funding_extreme   : 1 if |rate| > 2x historical std (crowded trade signal)
  funding_sign      : sign of rate — direction of carry pressure
"""

import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

FUNDING_FILE   = DATA_DIR / "funding_rates.parquet"
BYBIT_URL      = "https://api.bybit.com/v5/market/funding/history"
SYMBOL         = "BTCUSDT"
LIMIT_PER_PAGE = 200   # Bybit max per page


class FundingRateFetcher:

    def fetch_historical(self, days: int = 1500) -> pd.DataFrame:
        """
        Fetch up to `days` days of 8h funding rate history from Bybit.
        Paginates automatically. Returns DataFrame indexed by UTC timestamp.
        """
        logger.info(f"Fetching BTC funding rates ({days} days) from Bybit...")
        end_ms   = int(pd.Timestamp.utcnow().timestamp() * 1000)
        start_ms = end_ms - days * 86_400_000

        all_records = []
        cursor = end_ms

        while True:
            for attempt in range(3):
                try:
                    resp = requests.get(
                        BYBIT_URL,
                        params={
                            "category": "linear",
                            "symbol":   SYMBOL,
                            "endTime":  cursor,
                            "limit":    LIMIT_PER_PAGE,
                        },
                        timeout=15,
                    )
                    resp.raise_for_status()
                    result = resp.json()["result"]["list"]
                    break
                except Exception as e:
                    logger.warning(f"Funding fetch attempt {attempt+1} failed: {e}")
                    time.sleep(2)
            else:
                raise RuntimeError("Failed to fetch funding rates after 3 attempts")

            if not result:
                break

            all_records.extend(result)
            earliest_ts = int(result[-1]["fundingRateTimestamp"])
            if earliest_ts <= start_ms or len(result) < LIMIT_PER_PAGE:
                break
            cursor = earliest_ts - 1
            time.sleep(0.1)

        if not all_records:
            logger.warning("No funding rate data returned")
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df["timestamp"]    = pd.to_datetime(df["fundingRateTimestamp"].astype(int), unit="ms", utc=True)
        df["funding_rate"] = df["fundingRate"].astype(float)
        df = df[["timestamp", "funding_rate"]].set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="last")]

        logger.info(f"Fetched {len(df)} funding rate records ({df.index[0].date()} → {df.index[-1].date()})")
        df.to_parquet(FUNDING_FILE)
        return df

    def load_or_fetch(self, days: int = 1500) -> pd.DataFrame:
        """Load cached data if recent, otherwise re-fetch."""
        if FUNDING_FILE.exists():
            cached = pd.read_parquet(FUNDING_FILE)
            last   = cached.index[-1]
            age_h  = (pd.Timestamp.utcnow() - last).total_seconds() / 3600
            if age_h < 9:   # funding publishes every 8h, 9h = safe buffer
                logger.info(f"Funding cache current (last={last}, {len(cached)} records)")
                return cached
            logger.info(f"Funding cache stale ({age_h:.1f}h old), refreshing...")
        return self.fetch_historical(days)

    def align_to_ohlcv(self, funding: pd.DataFrame, ohlcv_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Reindex 8h funding rates to match the OHLCV candle index via forward-fill.
        Derives 6 trading features from the aligned series.
        """
        rate = funding["funding_rate"].reindex(
            funding["funding_rate"].index.union(ohlcv_index)
        ).ffill().reindex(ohlcv_index)

        # Historical std for extreme detection
        hist_std = rate.expanding().std().shift(1)

        out = pd.DataFrame(index=ohlcv_index)
        out["funding_rate"]       = rate
        out["funding_annualized"] = rate * 3 * 365              # annualized carry cost
        out["funding_ma_3"]       = rate.rolling(3).mean()      # 24h smoothed
        out["funding_ma_7"]       = rate.rolling(7).mean()      # 56h regime
        out["funding_extreme"]    = (rate.abs() > 2 * hist_std).astype(int)
        out["funding_sign"]       = rate.apply(
            lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
        )
        return out
