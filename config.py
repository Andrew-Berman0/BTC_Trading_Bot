"""
config.py
---------
Central configuration for the trading bot.
Edit this file to change symbols, timeframes, risk parameters, etc.
Never commit API keys — use environment variables or a .env file.
"""

import os
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    exchange:    str   = "binance"
    symbol:      str   = "BTC/USD"
    timeframe:   str   = "4h"
    days_back:   int   = 1825      # 5 years for initial training
    val_frac:    float = 0.2
    forward_n:   int   = 2         # predict next 8h (2 candles ahead)
    threshold:   float = 0.005     # 0.5% minimum move to call BUY/SELL


@dataclass
class ModelConfig:
    model_type:  str   = "lgbm"    # 'lgbm' | 'lstm' | 'ensemble'
    target:      str   = "target_direction"
    n_estimators:int   = 2000
    learning_rate:float= 0.05
    max_depth:   int   = 6
    num_leaves:  int   = 31
    # LSTM specific
    seq_len:     int   = 60        # lookback window in candles
    hidden_size: int   = 64
    num_layers:  int   = 2
    dropout:     float = 0.2


@dataclass
class RiskConfig:
    max_position_pct:  float = 0.01    # max 1% of equity per trade while unproven
    max_drawdown_pct:  float = 0.10    # kill switch at 10% drawdown
    stop_loss_pct:     float = 0.015   # 1.5% stop loss per trade
    take_profit_pct:   float = 0.030   # 3% take profit (2:1 R/R)
    min_signal_prob:   float = 0.65    # only trade if model confidence >= 65%
    max_open_trades:   int   = 3


@dataclass
class ExecutionConfig:
    paper_trade:   bool  = True     # ALWAYS start with paper trading
    order_type:    str   = "market" # 'market' | 'limit'
    slippage_bps:  float = 5.0      # assume 5bps slippage on market orders
    api_key:       str   = field(default_factory=lambda: os.getenv("EXCHANGE_API_KEY", ""))
    api_secret:    str   = field(default_factory=lambda: os.getenv("EXCHANGE_API_SECRET", ""))


@dataclass
class BotConfig:
    data:      DataConfig      = field(default_factory=DataConfig)
    model:     ModelConfig     = field(default_factory=ModelConfig)
    risk:      RiskConfig      = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    log_level: str             = "INFO"
    log_dir:   str             = "logs"


# Singleton — import this directly
CONFIG = BotConfig()
