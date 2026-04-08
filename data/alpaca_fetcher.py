"""
data/alpaca_fetcher.py
----------------------
Fetches OHLCV bars from Alpaca's Market Data API (free tier, no geo-restrictions).
Replaces the ccxt/Binance fetcher.

Supported crypto symbols: BTC/USD, ETH/USD, SOL/USD, etc.
"""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TIMEFRAME_MAP = {
    "1m":  TimeFrame(1,  TimeFrameUnit.Minute),
    "5m":  TimeFrame(5,  TimeFrameUnit.Minute),
    "15m": TimeFrame(15, TimeFrameUnit.Minute),
    "1h":  TimeFrame(1,  TimeFrameUnit.Hour),
    "4h":  TimeFrame(4,  TimeFrameUnit.Hour),
    "1d":  TimeFrame(1,  TimeFrameUnit.Day),
}


class AlpacaFetcher:
    """
    Fetches crypto OHLCV bars from Alpaca.
    No API key required for crypto data on the free tier.
    """

    def __init__(self):
        # Crypto data is public — no keys needed
        self.client = CryptoHistoricalDataClient()

    def fetch_historical(
        self,
        symbol: str = "BTC/USD",
        timeframe: str = "1h",
        days_back: int = 730,
    ) -> pd.DataFrame:
        if timeframe not in TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe '{timeframe}'. Choose from: {list(TIMEFRAME_MAP)}")

        end   = datetime.now(timezone.utc)
        start = end - timedelta(days=days_back)

        logger.info(f"Fetching {symbol} {timeframe} from {start.date()} to {end.date()} via Alpaca")

        request = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TIMEFRAME_MAP[timeframe],
            start=start,
            end=end,
        )
        bars = self.client.get_crypto_bars(request)
        df = bars.df

        # Alpaca returns a MultiIndex (symbol, timestamp) — drop symbol level
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level="symbol")

        df.index = pd.to_datetime(df.index, utc=True)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="last")]

        logger.info(f"Fetched {len(df):,} bars ({df.index[0]} → {df.index[-1]})")
        return df

    def fetch_latest(self, symbol: str = "BTC/USD", timeframe: str = "1h", lookback: int = 500) -> pd.DataFrame:
        end   = datetime.now(timezone.utc)
        start = end - timedelta(hours=lookback)
        request = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TIMEFRAME_MAP[timeframe],
            start=start,
            end=end,
        )
        bars = self.client.get_crypto_bars(request)
        df = bars.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level="symbol")
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        df.sort_index(inplace=True)
        return df

    def save(self, df: pd.DataFrame, name: str) -> Path:
        path = DATA_DIR / f"{name}.parquet"
        df.to_parquet(path)
        logger.info(f"Saved {len(df):,} rows → {path}")
        return path

    def load(self, name: str) -> pd.DataFrame:
        path = DATA_DIR / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No data at {path}")
        return pd.read_parquet(path)

    def update(self, symbol: str, timeframe: str, name: str) -> pd.DataFrame:
        try:
            existing = self.load(name)
            last_ts  = existing.index[-1]
            days_needed = max(1, (datetime.now(timezone.utc) - last_ts).days + 1)
            logger.info(f"Updating from {last_ts.date()} ({days_needed} days)")
            new = self.fetch_historical(symbol, timeframe, days_back=days_needed)
            combined = pd.concat([existing, new])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
            self.save(combined, name)
            return combined
        except FileNotFoundError:
            logger.info("No existing file — fetching from scratch")
            df = self.fetch_historical(symbol, timeframe)
            self.save(df, name)
            return df
