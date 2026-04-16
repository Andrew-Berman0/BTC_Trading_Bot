"""
features/engineer.py
--------------------
Transforms raw OHLCV data into ML-ready features.

Three feature groups:
  1. Technical indicators  — price-derived signals (RSI, MACD, Bollinger, ATR)
  2. Microstructure        — volume, spread proxies, VWAP deviation
  3. Regime / time         — rolling stats, autocorrelation, hour-of-day

Also handles:
  - Target column creation (forward return, binary direction)
  - Walk-forward train/val split (no future leakage)
  - Feature scaling

Usage:
    eng = FeatureEngineer(symbol="BTC/USDT", timeframe="1h")
    df  = eng.build(raw_df)
    X_train, X_val, y_train, y_val = eng.split(df, target="direction_4h")
"""

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

# Lazy import to avoid circular deps
def _get_fng_fetcher():
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.fng_fetcher import FearGreedFetcher
    return FearGreedFetcher()

def _get_funding_fetcher():
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.funding_fetcher import FundingRateFetcher
    return FundingRateFetcher()

def _get_dominance_fetcher():
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.dominance_fetcher import DominanceFetcher
    return DominanceFetcher()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FeatureEngineer:
    """
    Parameters
    ----------
    symbol      : informational only, used in logging
    timeframe   : '1h', '4h', etc.  (informational)
    forward_n   : candles ahead for target return calculation
    direction_threshold : min |return| to count as BUY/SELL (else HOLD=0)
    """

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        forward_n: int = 4,
        direction_threshold: float = 0.005,   # 0.5%
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.forward_n = forward_n
        self.direction_threshold = direction_threshold
        self.scaler = RobustScaler()
        self.feature_cols: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full feature pipeline on a raw OHLCV DataFrame.
        Returns a new DataFrame with all features + target columns appended.
        NaNs from indicator warmup are dropped.
        """
        out = df.copy()
        out = self._add_returns(out)
        out = self._add_technical(out)
        out = self._add_microstructure(out)
        out = self._add_regime(out)
        out = self._add_time_features(out)
        out = self._add_fear_greed(out)
        out = self._add_funding_rates(out)
        out = self._add_dominance(out)
        out = self._add_targets(out)

        # Drop warmup NaNs (longest indicator window is ~200 candles)
        before = len(out)
        out.dropna(inplace=True)
        dropped = before - len(out)
        if dropped:
            logger.info(f"Dropped {dropped} warmup rows; {len(out):,} rows remaining")

        # Record which columns are features (not raw OHLCV, not targets)
        raw_cols = {"open", "high", "low", "close", "volume"}
        target_cols = {c for c in out.columns if c.startswith("target_") or c.startswith("fwd_")}
        self.feature_cols = [c for c in out.columns if c not in raw_cols | target_cols]

        logger.info(f"Built {len(self.feature_cols)} features on {len(out):,} rows")
        return out

    def split(
        self,
        df: pd.DataFrame,
        target: str = "target_direction",
        val_frac: float = 0.2,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Walk-forward split — val is always the LAST val_frac of the data.
        Never shuffle; never let val data precede train data in time.

        Returns: X_train, X_val, y_train, y_val
        """
        n = len(df)
        split_idx = int(n * (1 - val_frac))
        train = df.iloc[:split_idx]
        val   = df.iloc[split_idx:]

        X_train = train[self.feature_cols]
        X_val   = val[self.feature_cols]
        y_train = train[target]
        y_val   = val[target]

        logger.info(
            f"Split: train={len(train):,} ({train.index[0].date()}→{train.index[-1].date()})  "
            f"val={len(val):,} ({val.index[0].date()}→{val.index[-1].date()})"
        )
        return X_train, X_val, y_train, y_val

    def scale(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fit RobustScaler on train only, transform both.
        RobustScaler is better than StandardScaler for financial data
        (handles outlier spikes without distorting the bulk of the data).
        """
        X_train_s = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        X_val_s = pd.DataFrame(
            self.scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index,
        )
        return X_train_s, X_val_s

    def transform_live(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build features on a live candle window and scale with the
        already-fitted scaler. Returns the last row as a 1-row DataFrame.
        Stores the full unscaled featured DataFrame as self.last_featured
        so the ATR circuit breaker can reuse it without rebuilding.
        """
        self.last_featured = self.build(df)
        X = self.last_featured[self.feature_cols].tail(1)
        return pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index,
        )

    # ------------------------------------------------------------------
    # Feature groups
    # ------------------------------------------------------------------

    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        df["log_ret_1"]  = np.log(df["close"] / df["close"].shift(1))
        df["log_ret_4"]  = np.log(df["close"] / df["close"].shift(4))
        df["log_ret_24"] = np.log(df["close"] / df["close"].shift(24))
        df["hl_ratio"]   = (df["high"] - df["low"]) / df["close"]   # range / price
        return df

    def _add_technical(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df["close"]
        h = df["high"]
        l = df["low"]
        v = df["volume"]

        # --- RSI (7 only — rsi_14 was r=0.94 with bb_pct/rsi_7, redundant) ---
        df["rsi_7"] = self._rsi(c, 7)

        # --- MACD (histogram only — macd_cross had near-zero importance) ---
        ema12  = c.ewm(span=12, adjust=False).mean()
        ema26  = c.ewm(span=26, adjust=False).mean()
        macd   = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        df["macd_hist"] = macd - signal

        # --- Bollinger Bands (20, 2σ) — keep pct/width only, drop raw bands ---
        sma20    = c.rolling(20).mean()
        std20    = c.rolling(20).std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        df["bb_pct"]   = (c - bb_lower) / (bb_upper - bb_lower)
        df["bb_width"] = (bb_upper - bb_lower) / sma20

        # --- ATR (14) ---
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        df["atr_14"]   = tr.ewm(span=14, adjust=False).mean()
        df["atr_norm"] = df["atr_14"] / c

        # --- EMA diffs only — raw EMAs were r>0.99 with each other and vwap ---
        ema_9  = c.ewm(span=9,   adjust=False).mean()
        ema_21 = c.ewm(span=21,  adjust=False).mean()
        ema_50 = c.ewm(span=50,  adjust=False).mean()
        ema_200= c.ewm(span=200, adjust=False).mean()
        df["ema_9_21_diff"]   = (ema_9  - ema_21)  / c
        df["ema_21_50_diff"]  = (ema_21 - ema_50)  / c
        df["ema_50_200_diff"] = (ema_50 - ema_200) / c

        # --- Stochastic (smoothed line only — stoch_k was r=0.92 with stoch_d) ---
        low_14  = l.rolling(14).min()
        high_14 = h.rolling(14).max()
        stoch_k = 100 * (c - low_14) / (high_14 - low_14 + 1e-9)
        df["stoch_d"] = stoch_k.rolling(3).mean()

        return df

    def _add_microstructure(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df["close"]
        v = df["volume"]
        h = df["high"]
        l = df["low"]

        # --- VWAP deviation only — raw vwap_24 was r>0.999 with all EMAs ---
        typical  = (h + l + c) / 3
        vwap_24  = (typical * v).rolling(24).sum() / v.rolling(24).sum()
        df["vwap_dev"] = (c - vwap_24) / vwap_24

        # --- Volume features ---
        df["vol_sma_20"]   = v.rolling(20).mean()
        df["vol_ratio"]    = v / df["vol_sma_20"]
        candle_sign        = np.sign(df["close"] - df["open"])    # replaces dropped oc_ratio
        df["vol_delta"]    = v * candle_sign
        df["vol_delta_ma"] = df["vol_delta"].rolling(10).mean()

        # --- Amihud illiquidity (price impact per unit volume) ---
        df["amihud"] = df["log_ret_1"].abs() / (v * c + 1e-9)
        df["amihud"] = df["amihud"].rolling(20).mean()

        # --- High-low spread proxy (Corwin-Schultz) ---
        beta  = (np.log(h / l) ** 2).rolling(2).sum()
        gamma = (np.log(h.rolling(2).max() / l.rolling(2).min())) ** 2
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2)) - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))
        df["cs_spread"] = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
        df["cs_spread"]  = df["cs_spread"].clip(lower=0)   # numerical artifacts

        return df

    def _add_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        r = df["log_ret_1"]

        # --- Rolling volatility ---
        df["vol_20"]  = r.rolling(20).std()
        df["vol_5"]   = r.rolling(5).std()
        df["vol_ratio_5_20"] = df["vol_5"] / (df["vol_20"] + 1e-9)  # vol expansion

        # --- Autocorrelation of returns (mean reversion vs momentum) ---
        df["autocorr_10"] = r.rolling(20).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 10 else np.nan, raw=False
        )

        # --- Hurst exponent (simplified R/S) ---
        df["hurst"] = r.rolling(100).apply(self._hurst_rs, raw=True)

        # --- Rolling Sharpe (return / vol) ---
        df["sharpe_20"] = (
            r.rolling(20).mean() / (r.rolling(20).std() + 1e-9) * np.sqrt(24)
        )

        # --- Max drawdown over 20 candles ---
        rolling_max = df["close"].rolling(20).max()
        df["dd_20"] = (df["close"] - rolling_max) / rolling_max

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        idx = df.index
        # Cyclically encode so 23:00 and 00:00 are adjacent
        hour  = idx.hour
        dow   = idx.dayofweek
        month = idx.month

        df["hour_sin"]  = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"]  = np.cos(2 * np.pi * hour / 24)
        df["dow_sin"]   = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"]   = np.cos(2 * np.pi * dow / 7)
        df["month_sin"] = np.sin(2 * np.pi * month / 12)
        df["month_cos"] = np.cos(2 * np.pi * month / 12)

        return df

    def _add_funding_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge 8h Binance funding rates into the candle DataFrame via forward-fill.
        Adds 6 features. Fails gracefully if Binance is unreachable.
        """
        try:
            fetcher  = _get_funding_fetcher()
            raw      = fetcher.load_or_fetch()
            features = fetcher.align_to_ohlcv(raw, df.index)
            for col in features.columns:
                df[col] = features[col]
            # funding_annualized is r=1.0 with funding_rate (just a rescaling)
            # funding_extreme and funding_sign had near-zero importance
            _drop = ["funding_annualized", "funding_extreme", "funding_sign"]
            df.drop(columns=_drop, errors="ignore", inplace=True)
            kept = [c for c in features.columns if c not in _drop]
            logger.info(f"Added {len(kept)} funding rate features")
        except Exception as e:
            logger.warning(f"Funding rate fetch failed, skipping: {e}")
        return df

    def _add_dominance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge BTC/ETH ratio (dominance proxy) into the candle DataFrame.
        Adds 6 features. Fails gracefully if Alpaca is unreachable.
        """
        try:
            fetcher  = _get_dominance_fetcher()
            raw      = fetcher.load_or_fetch()
            features = fetcher.align_to_ohlcv(raw, df.index)
            for col in features.columns:
                df[col] = features[col]
            # btc_eth_ma_30 was r=0.998 with btc_eth_ma_7; btc_dominance_up had importance=17
            # btc_eth_ratio was r=0.9995 with btc_eth_ma_7; ratio_norm + ma_7 cover the same info
            _drop = ["btc_eth_ma_30", "btc_dominance_up", "btc_eth_ratio"]
            df.drop(columns=_drop, errors="ignore", inplace=True)
            kept = [c for c in features.columns if c not in _drop]
            logger.info(f"Added {len(kept)} BTC dominance features")
        except Exception as e:
            logger.warning(f"Dominance fetch failed, skipping: {e}")
        return df

    def _add_fear_greed(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge daily Fear & Greed Index into the candle DataFrame.
        Forward-fills the daily value to every candle.
        Adds 6 features: fng_value, fng_norm, fng_change_1d,
                         fng_change_7d, fng_ma_7, fng_extreme.
        Fails gracefully — if the API is down, skips without crashing.
        """
        try:
            fetcher = _get_fng_fetcher()
            fng_raw = fetcher.load_or_fetch()
            fng_features = fetcher.align_to_ohlcv(fng_raw, df.index)
            for col in fng_features.columns:
                df[col] = fng_features[col]
            # fng_value is r=1.0 with fng_norm; fng_norm is r=0.99 with fng_ma_7 (lower importance)
            # fng_extreme had importance=11
            _drop = ["fng_value", "fng_norm", "fng_extreme"]
            df.drop(columns=_drop, errors="ignore", inplace=True)
            kept = [c for c in fng_features.columns if c not in _drop]
            logger.info(f"Added {len(kept)} Fear & Greed features")
        except Exception as e:
            logger.warning(f"Fear & Greed fetch failed, skipping: {e}")
        return df

    def _add_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Two target columns:
          fwd_return_n    : log return forward n candles (regression target)
          target_direction: -1 / 0 / 1 (classification target)
        """
        n = self.forward_n
        df[f"fwd_return_{n}"] = np.log(df["close"].shift(-n) / df["close"])
        thresh = self.direction_threshold
        df["target_direction"] = np.where(
            df[f"fwd_return_{n}"] >  thresh,  1,
            np.where(df[f"fwd_return_{n}"] < -thresh, -1, 0)
        )
        return df

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain  = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _hurst_rs(ts: np.ndarray) -> float:
        """
        Simplified R/S Hurst exponent for a return series.
        H < 0.5  → mean-reverting
        H ≈ 0.5  → random walk
        H > 0.5  → trending
        """
        n = len(ts)
        if n < 20:
            return np.nan
        mean_adj = ts - ts.mean()
        cum_dev  = np.cumsum(mean_adj)
        R = cum_dev.max() - cum_dev.min()
        S = ts.std(ddof=1)
        if S == 0:
            return np.nan
        rs = R / S
        if rs <= 0:
            return np.nan
        return np.log(rs) / np.log(n)
