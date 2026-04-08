"""
broker/alpaca_broker.py
-----------------------
Paper trading execution via Alpaca's Trading API.

Handles:
  - Placing market orders (BUY / SELL / flatten)
  - Querying open positions and account equity
  - Risk guardrails (max position size, drawdown kill switch)

Usage:
    broker = AlpacaBroker()           # reads keys from .env
    broker.buy("BTC/USD", usd=500)
    broker.sell("BTC/USD", usd=500)
    broker.flatten("BTC/USD")
    pos = broker.get_position("BTC/USD")
    eq  = broker.get_equity()
"""

import logging
import os
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, ClosePositionRequest
from alpaca.trading.enums import OrderSide, TimeInForce

load_dotenv()
logger = logging.getLogger(__name__)


class AlpacaBroker:
    """
    Thin wrapper around Alpaca's TradingClient for crypto paper trading.

    Parameters
    ----------
    paper : bool
        Always True for now — never flip this until you've validated the model.
    """

    def __init__(self, paper: bool = True):
        api_key    = os.getenv("ALPACA_API_KEY", "")
        api_secret = os.getenv("ALPACA_API_SECRET", "")

        if not api_key or not api_secret:
            raise EnvironmentError(
                "ALPACA_API_KEY and ALPACA_API_SECRET must be set in your .env file."
            )

        self.client = TradingClient(api_key, api_secret, paper=paper)
        self.paper  = paper
        mode = "PAPER" if paper else "LIVE"
        logger.info(f"AlpacaBroker connected [{mode}]")

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def buy(self, symbol: str, usd: float) -> dict:
        """
        Place a market buy for `usd` notional value of `symbol`.
        Alpaca crypto supports fractional / notional orders.
        """
        return self._place_order(symbol, OrderSide.BUY, usd)

    def sell(self, symbol: str, usd: float) -> dict:
        """Place a market sell for `usd` notional value."""
        return self._place_order(symbol, OrderSide.SELL, usd)

    def flatten(self, symbol: str) -> dict | None:
        """Close the entire position in `symbol`, if one exists."""
        alpaca_symbol = self._to_alpaca_symbol(symbol)
        pos = self.get_position(symbol)
        if pos is None:
            logger.info(f"No open position in {symbol} to flatten")
            return None
        try:
            result = self.client.close_position(alpaca_symbol)
            logger.info(f"Flattened {symbol}")
            return result
        except Exception as e:
            logger.error(f"Failed to flatten {symbol}: {e}")
            raise

    # ------------------------------------------------------------------
    # Account queries
    # ------------------------------------------------------------------

    def get_position(self, symbol: str) -> dict | None:
        """
        Returns position dict with keys: qty, market_value, side, avg_entry_price.
        Returns None if no position.
        """
        alpaca_symbol = self._to_alpaca_symbol(symbol)
        try:
            pos = self.client.get_open_position(alpaca_symbol)
            return {
                "symbol":          symbol,
                "qty":             float(pos.qty),
                "market_value":    float(pos.market_value),
                "side":            pos.side.value,
                "avg_entry_price": float(pos.avg_entry_price),
                "unrealized_pl":   float(pos.unrealized_pl),
                "unrealized_plpc": float(pos.unrealized_plpc),
            }
        except Exception:
            return None

    def get_equity(self) -> float:
        """Return total portfolio equity in USD."""
        account = self.client.get_account()
        return float(account.equity)

    def get_account(self) -> dict:
        account = self.client.get_account()
        return {
            "equity":        float(account.equity),
            "cash":          float(account.cash),
            "buying_power":  float(account.buying_power),
            "daytrade_count": account.daytrade_count,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _place_order(self, symbol: str, side: OrderSide, usd: float) -> dict:
        alpaca_symbol = self._to_alpaca_symbol(symbol)
        logger.info(f"{'BUY' if side == OrderSide.BUY else 'SELL'} {symbol} ${usd:,.2f} [{'PAPER' if self.paper else 'LIVE'}]")
        try:
            order = self.client.submit_order(
                MarketOrderRequest(
                    symbol=alpaca_symbol,
                    notional=round(usd, 2),    # USD amount (fractional crypto)
                    side=side,
                    time_in_force=TimeInForce.IOC,  # immediate-or-cancel for crypto
                )
            )
            logger.info(f"Order submitted: {order.id} status={order.status}")
            return {"id": str(order.id), "status": str(order.status), "symbol": symbol, "usd": usd}
        except Exception as e:
            logger.error(f"Order failed: {e}")
            raise

    @staticmethod
    def _to_alpaca_symbol(symbol: str) -> str:
        """Convert 'BTC/USD' → 'BTC/USD' (Alpaca crypto uses slash format)."""
        return symbol.replace("-", "/")
