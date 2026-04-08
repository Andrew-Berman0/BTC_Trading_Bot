"""
data/fetcher.py
---------------
Fetches OHLCV candle data from any ccxt-supported exchange.
Supports both historical backfill and live incremental updates.

Usage:
    fetcher = OHLCVFetcher(exchange_id="binance")
    df = fetcher.fetch_historical("BTC/USDT", "1h", days_back=365)
    fetcher.save(df, "BTC_USDT_1h")
"""

import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ccxt
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TIMEFRAME_MS = {
    "1m":  60_000,
    "5m":  300_000,
    "15m": 900_000,
    "1h":  3_600_000,
    "4h":  14_400_000,
    "1d":  86_400_000,
}

# ---------------------------------------------------------------------------
# Core fetcher class
# ---------------------------------------------------------------------------
class OHLCVFetcher:
    """
    Wraps ccxt to fetch, validate, and store OHLCV candle data.

    Parameters
    ----------
    exchange_id : str
        Any ccxt exchange id, e.g. 'binance', 'kraken', 'coinbase'
    api_key : str, optional
        Only needed for private endpoints (order book depth >5 levels, etc.)
    api_secret : str, optional
    rate_limit_sleep : float
        Extra sleep (seconds) between paginated requests. Binance allows
        ~1200 req/min on public endpoints; 0.1s is safe.
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: str | None = None,
        api_secret: str | None = None,
        rate_limit_sleep: float = 0.1,
    ):
        self.exchange_id = exchange_id
        self.rate_limit_sleep = rate_limit_sleep

        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,   # ccxt built-in throttle
            }
        )
        self.exchange.load_markets()
        logger.info(f"Connected to {exchange_id}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_historical(
        self,
        symbol: str,
        timeframe: str = "1h",
        days_back: int = 365,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Paginate backwards through candles and return a clean DataFrame.

        Parameters
        ----------
        symbol    : e.g. 'BTC/USDT'
        timeframe : '1m', '5m', '15m', '1h', '4h', '1d'
        days_back : how far back to fetch (ignored if `since` is set)
        since     : explicit start datetime (UTC)
        until     : explicit end datetime (UTC); defaults to now

        Returns
        -------
        pd.DataFrame with columns: open, high, low, close, volume
        index: DatetimeIndex (UTC)
        """
        self._validate_symbol(symbol)
        self._validate_timeframe(timeframe)

        until_dt = until or datetime.now(timezone.utc)
        since_dt = since or (until_dt - timedelta(days=days_back))

        since_ms = int(since_dt.timestamp() * 1000)
        until_ms = int(until_dt.timestamp() * 1000)
        tf_ms    = TIMEFRAME_MS[timeframe]
        limit    = 1000   # Binance max per request

        logger.info(
            f"Fetching {symbol} {timeframe} from {since_dt.date()} to {until_dt.date()}"
        )

        all_candles: list[list] = []
        cursor = since_ms

        while cursor < until_ms:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=cursor, limit=limit
                )
            except ccxt.NetworkError as e:
                logger.warning(f"Network error, retrying: {e}")
                time.sleep(2)
                continue
            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error: {e}")
                raise

            if not candles:
                break

            all_candles.extend(candles)
            last_ts = candles[-1][0]

            # Advance cursor; stop if exchange returned fewer than limit
            if len(candles) < limit:
                break
            cursor = last_ts + tf_ms
            time.sleep(self.rate_limit_sleep)

        if not all_candles:
            logger.warning("No candles returned.")
            return pd.DataFrame()

        df = self._to_dataframe(all_candles)
        df = df[df.index <= until_dt]   # trim any overshoot
        df = self._clean(df)

        logger.info(f"Fetched {len(df):,} candles ({df.index[0]} → {df.index[-1]})")
        return df

    def fetch_latest(
        self,
        symbol: str,
        timeframe: str = "1h",
        lookback_candles: int = 500,
    ) -> pd.DataFrame:
        """
        Fetch the most recent N candles. Used in the live loop.
        """
        self._validate_symbol(symbol)
        candles = self.exchange.fetch_ohlcv(
            symbol, timeframe, limit=lookback_candles
        )
        df = self._to_dataframe(candles)
        return self._clean(df)

    def fetch_order_book(
        self,
        symbol: str,
        depth: int = 20,
    ) -> dict:
        """
        Fetch order book snapshot. Returns dict with 'bids', 'asks',
        'mid_price', 'spread', 'imbalance'.
        """
        ob = self.exchange.fetch_order_book(symbol, limit=depth)
        bids = np.array(ob["bids"])  # [[price, qty], ...]
        asks = np.array(ob["asks"])

        mid   = (bids[0, 0] + asks[0, 0]) / 2
        spread = asks[0, 0] - bids[0, 0]
        bid_vol = bids[:, 1].sum()
        ask_vol = asks[:, 1].sum()
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)   # [-1, 1]

        return {
            "bids":      bids,
            "asks":      asks,
            "mid_price": mid,
            "spread":    spread,
            "spread_bps": spread / mid * 10_000,
            "imbalance": imbalance,
            "timestamp": pd.Timestamp.utcnow(),
        }

    def save(self, df: pd.DataFrame, name: str, fmt: str = "parquet") -> Path:
        """
        Persist DataFrame to disk.

        Parameters
        ----------
        name : filename stem, e.g. 'BTC_USDT_1h'
        fmt  : 'parquet' (default, fast + typed) or 'csv'
        """
        if df.empty:
            logger.warning("DataFrame is empty, skipping save.")
            return Path()

        path = DATA_DIR / f"{name}.{fmt}"
        if fmt == "parquet":
            df.to_parquet(path)
        else:
            df.to_csv(path)
        logger.info(f"Saved {len(df):,} rows → {path}")
        return path

    def load(self, name: str, fmt: str = "parquet") -> pd.DataFrame:
        """Load previously saved data."""
        path = DATA_DIR / f"{name}.{fmt}"
        if not path.exists():
            raise FileNotFoundError(f"No data file at {path}")
        if fmt == "parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path, index_col=0, parse_dates=True)

    def update(self, symbol: str, timeframe: str, name: str) -> pd.DataFrame:
        """
        Append new candles to an existing saved dataset.
        Avoids re-downloading data you already have.
        """
        try:
            existing = self.load(name)
            last_ts  = existing.index[-1]
            since_dt = last_ts + pd.Timedelta(TIMEFRAME_MS[timeframe], unit="ms")
            logger.info(f"Updating from {since_dt}")
            new = self.fetch_historical(symbol, timeframe, since=since_dt)
            if new.empty:
                logger.info("Already up to date.")
                return existing
            combined = pd.concat([existing, new])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
            self.save(combined, name)
            return combined
        except FileNotFoundError:
            logger.info("No existing file — fetching from scratch.")
            df = self.fetch_historical(symbol, timeframe)
            self.save(df, name)
            return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_dataframe(self, candles: list[list]) -> pd.DataFrame:
        df = pd.DataFrame(
            candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df.astype(float)

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates, sort, and flag anomalies."""
        df = df[~df.index.duplicated(keep="last")]
        df.sort_index(inplace=True)

        # Drop zero-volume candles (exchange outages, holidays)
        zero_vol = (df["volume"] == 0).sum()
        if zero_vol:
            logger.warning(f"Dropping {zero_vol} zero-volume candles")
            df = df[df["volume"] > 0]

        # Sanity check: OHLC consistency
        bad_ohlc = (
            (df["high"] < df["low"]) |
            (df["close"] > df["high"] * 1.5) |
            (df["close"] < df["low"] * 0.5)
        ).sum()
        if bad_ohlc:
            logger.warning(f"{bad_ohlc} candles failed OHLC sanity check — dropping")
            mask = (
                (df["high"] >= df["low"]) &
                (df["close"] <= df["high"] * 1.5) &
                (df["close"] >= df["low"] * 0.5)
            )
            df = df[mask]

        return df

    def _validate_symbol(self, symbol: str):
        if symbol not in self.exchange.markets:
            raise ValueError(
                f"'{symbol}' not found on {self.exchange_id}. "
                f"Example valid symbols: BTC/USDT, ETH/USDT, SOL/USDT"
            )

    def _validate_timeframe(self, timeframe: str):
        if timeframe not in TIMEFRAME_MS:
            raise ValueError(
                f"Unsupported timeframe '{timeframe}'. "
                f"Choose from: {list(TIMEFRAME_MS.keys())}"
            )
