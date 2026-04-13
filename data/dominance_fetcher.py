"""
data/dominance_fetcher.py
-------------------------
Computes a BTC dominance proxy from the BTC/ETH price ratio.

True BTC dominance (BTC market cap / total market cap) requires paid APIs
for full history. The BTC/ETH ratio is a strong free proxy — when BTC
outperforms ETH, dominance typically rises, and vice versa. Correlation
with true dominance is ~0.85+ on daily timeframes.

Features derived:
  btc_eth_ratio      : BTC price / ETH price (dominance proxy)
  btc_eth_ratio_norm : z-score normalized over 90-day rolling window
  btc_eth_ma_7       : 7-day rolling mean (trend)
  btc_eth_ma_30      : 30-day rolling mean (regime)
  btc_eth_momentum   : 14-period rate of change
  btc_dominance_up   : 1 if ratio > 30-day MA (BTC outperforming = risk-off)
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DOMINANCE_FILE = DATA_DIR / "btc_eth_ratio.parquet"


class DominanceFetcher:

    def fetch(self, days: int = 1825) -> pd.DataFrame:
        """
        Fetch BTC and ETH price history from Alpaca and compute ratio.
        Returns a DataFrame indexed by UTC timestamp with ratio features.
        """
        from data.alpaca_fetcher import AlpacaFetcher
        fetcher = AlpacaFetcher()

        logger.info(f"Fetching BTC/ETH ratio history ({days} days)...")
        btc = fetcher.fetch_historical("BTC/USD", timeframe="1d", days_back=days)
        eth = fetcher.fetch_historical("ETH/USD", timeframe="1d", days_back=days)

        # Align on common dates
        ratio = (btc["close"] / eth["close"]).dropna()
        ratio.name = "btc_eth_ratio"

        df = ratio.to_frame()
        df.to_parquet(DOMINANCE_FILE)
        logger.info(f"Computed {len(df)} BTC/ETH ratio points ({df.index[0].date()} → {df.index[-1].date()})")
        return df

    def load_or_fetch(self, days: int = 1825) -> pd.DataFrame:
        if DOMINANCE_FILE.exists():
            cached = pd.read_parquet(DOMINANCE_FILE)
            last  = cached.index[-1].date()
            today = pd.Timestamp.utcnow().date()
            if last >= today:
                logger.info(f"Dominance cache current (last={last}, {len(cached)} days)")
                return cached
            logger.info(f"Dominance cache stale (last={last}), refreshing...")
        return self.fetch(days)

    def align_to_ohlcv(self, dominance: pd.DataFrame, ohlcv_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Reindex daily ratio to match candle index via forward-fill.
        Derives 6 features.
        """
        ratio = dominance["btc_eth_ratio"].reindex(
            dominance["btc_eth_ratio"].index.union(ohlcv_index)
        ).ffill().reindex(ohlcv_index)

        out = pd.DataFrame(index=ohlcv_index)
        out["btc_eth_ratio"]      = ratio
        out["btc_eth_ratio_norm"] = (
            (ratio - ratio.rolling(90).mean()) /
            (ratio.rolling(90).std() + 1e-9)
        )
        out["btc_eth_ma_7"]       = ratio.rolling(7).mean()
        out["btc_eth_ma_30"]      = ratio.rolling(30).mean()
        out["btc_eth_momentum"]   = ratio.pct_change(14)
        out["btc_dominance_up"]   = (ratio > out["btc_eth_ma_30"]).astype(int)
        return out
