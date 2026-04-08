"""
bot.py
------
Main entry point for the paper trading bot.

Two modes:
  python bot.py --train     Build dataset + train model, then start loop
  python bot.py             Load saved model, start live loop immediately

The loop runs on the candle close of each bar (default: 1h).
Every bar it:
  1. Fetches latest candles from Alpaca
  2. Engineers features
  3. Asks the model for a signal + confidence
  4. Executes via Alpaca paper trading if confidence >= threshold
  5. Enforces risk rules (stop-loss, drawdown kill switch)

Press Ctrl+C to stop.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from config import CONFIG
from data.alpaca_fetcher import AlpacaFetcher
from features.engineer import FeatureEngineer
from models.trainer import ModelTrainer
from broker.alpaca_broker import AlpacaBroker

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bot")

# ---------------------------------------------------------------------------
SYMBOL    = "BTC/USD"           # Alpaca uses BTC/USD (not BTC/USDT)
TIMEFRAME = CONFIG.data.timeframe
FORWARD_N = CONFIG.data.forward_n
THRESHOLD = CONFIG.data.threshold
MIN_PROB  = CONFIG.risk.min_signal_prob    # 0.60 default
MAX_DRAWDOWN = CONFIG.risk.max_drawdown_pct

CANDLE_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
RETRAIN_EVERY_DAYS = 7       # retrain weekly
STATE_FILE = Path(__file__).parent / "bot_state.json"
# ---------------------------------------------------------------------------


def save_state(last_retrain: datetime):
    """Persist bot state so it survives power outages."""
    with open(STATE_FILE, "w") as f:
        json.dump({"last_retrain": last_retrain.isoformat()}, f)


def load_state() -> datetime:
    """Load last retrain time from disk, or return epoch if no state exists."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            data = json.load(f)
        ts = datetime.fromisoformat(data["last_retrain"])
        logger.info(f"Resumed from saved state — last retrain: {ts.date()}")
        return ts
    return datetime.fromtimestamp(0, tz=timezone.utc)


def maybe_retrain(
    trainer: ModelTrainer,
    eng: FeatureEngineer,
    fetcher: AlpacaFetcher,
    last_retrain: datetime,
) -> tuple[ModelTrainer, FeatureEngineer, datetime]:
    """
    Retrain if RETRAIN_EVERY_DAYS have passed since last retrain.
    Only swaps in the new model if val F1 improves.
    Returns updated (trainer, eng, last_retrain).
    """
    now = datetime.now(timezone.utc)
    if (now - last_retrain).days < RETRAIN_EVERY_DAYS:
        return trainer, eng, last_retrain

    logger.info(f"=== WEEKLY RETRAIN (last: {last_retrain.date()}) ===")
    name = f"{SYMBOL.replace('/', '_')}_{TIMEFRAME}"

    try:
        raw = fetcher.update(SYMBOL, TIMEFRAME, name)

        new_eng = FeatureEngineer(
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            forward_n=FORWARD_N,
            direction_threshold=THRESHOLD,
        )
        featured = new_eng.build(raw)
        X_train, X_val, y_train, y_val = new_eng.split(featured, val_frac=CONFIG.data.val_frac)
        X_train_s, X_val_s = new_eng.scale(X_train, X_val)

        new_trainer = ModelTrainer(model_path="lgbm_model_candidate.pkl")
        metrics     = new_trainer.train(X_train_s, y_train, X_val_s, y_val)
        new_sharpe  = metrics["sharpe"]
        new_mcc     = metrics["mcc"]
        old_sharpe  = trainer.current_sharpe()

        logger.info(f"Retrain complete — new Sharpe: {new_sharpe:.4f}  MCC: {new_mcc:.4f}  |  old Sharpe: {old_sharpe:.4f}")

        if new_sharpe >= old_sharpe:
            new_trainer.model_path = ModelTrainer().model_path
            new_trainer.save(
                mcc=new_mcc, sharpe=new_sharpe,
                precision=metrics["precision"], f1=metrics["f1"],
                optimal_threshold=metrics.get("optimal_threshold", 0.65),
                n_train=len(X_train),
            )
            logger.info("New model promoted (Sharpe improved).")
            last_retrain = now
            save_state(last_retrain)
            return new_trainer, new_eng, last_retrain
        else:
            logger.info("New model did not improve Sharpe — keeping current model.")
            last_retrain = now
            save_state(last_retrain)
            return trainer, eng, last_retrain

    except Exception as e:
        logger.error(f"Retrain failed: {e}", exc_info=True)
        return trainer, eng, last_retrain


def build_and_train() -> tuple[ModelTrainer, FeatureEngineer]:
    """Fetch data, engineer features, train model, save everything."""
    logger.info("=== BUILD & TRAIN MODE ===")

    fetcher = AlpacaFetcher()
    raw = fetcher.fetch_historical(SYMBOL, TIMEFRAME, days_back=CONFIG.data.days_back)
    name = f"{SYMBOL.replace('/', '_')}_{TIMEFRAME}"
    fetcher.save(raw, name)

    eng = FeatureEngineer(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        forward_n=FORWARD_N,
        direction_threshold=THRESHOLD,
    )
    featured = eng.build(raw)
    X_train, X_val, y_train, y_val = eng.split(featured, val_frac=CONFIG.data.val_frac)
    X_train_s, X_val_s = eng.scale(X_train, X_val)

    trainer = ModelTrainer()
    metrics = trainer.train(X_train_s, y_train, X_val_s, y_val)
    trainer.save(
        mcc=metrics["mcc"], sharpe=metrics["sharpe"],
        precision=metrics["precision"], f1=metrics["f1"],
        optimal_threshold=metrics.get("optimal_threshold", 0.65),
        n_train=len(X_train_s),
    )
    save_state(datetime.now(timezone.utc))

    logger.info("Training complete. Starting live loop...")
    return trainer, eng


def load_trained() -> tuple[ModelTrainer, FeatureEngineer]:
    """Load a pre-trained model and a fresh FeatureEngineer."""
    trainer = ModelTrainer()
    trainer.load()

    eng = FeatureEngineer(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        forward_n=FORWARD_N,
        direction_threshold=THRESHOLD,
    )
    # Rebuild scaler by re-loading the raw data and re-fitting
    fetcher = AlpacaFetcher()
    name = f"{SYMBOL.replace('/', '_')}_{TIMEFRAME}"
    try:
        raw = fetcher.load(name)
    except FileNotFoundError:
        logger.warning("No cached data found — fetching fresh.")
        raw = fetcher.fetch_historical(SYMBOL, TIMEFRAME, days_back=CONFIG.data.days_back)
        fetcher.save(raw, name)

    featured = eng.build(raw)
    X_train, X_val, _, _ = eng.split(featured, val_frac=CONFIG.data.val_frac)
    eng.scale(X_train, X_val)     # fits the scaler in-place
    logger.info("Model + scaler ready.")
    return trainer, eng


def run_loop(trainer: ModelTrainer, eng: FeatureEngineer):
    """
    Main trading loop. Wakes up after each candle close, makes a decision.
    """
    fetcher = AlpacaFetcher()
    broker  = AlpacaBroker(paper=CONFIG.execution.paper_trade)

    account      = broker.get_account()
    start_equity = account["equity"]
    last_retrain = load_state()
    threshold    = trainer.optimal_threshold()
    logger.info(f"Starting equity: ${start_equity:,.2f}  |  Confidence threshold: {threshold:.2f}")

    candle_secs = CANDLE_SECONDS.get(TIMEFRAME, 3600)
    label_map   = {-1: "SELL", 0: "HOLD", 1: "BUY"}

    while True:
        try:
            # ---- 0. Weekly retrain check ----
            trainer, eng, last_retrain = maybe_retrain(trainer, eng, fetcher, last_retrain)

            # ---- 1. Wait for candle close ----
            now = datetime.now(timezone.utc)
            secs_into_candle = now.timestamp() % candle_secs
            secs_to_close    = candle_secs - secs_into_candle + 2   # +2s buffer
            logger.info(f"Next candle close in {secs_to_close:.0f}s — sleeping...")
            time.sleep(secs_to_close)

            # ---- 2. Fetch latest candles ----
            raw_live = fetcher.fetch_latest(SYMBOL, TIMEFRAME, lookback=2000)
            X_live   = eng.transform_live(raw_live)

            # ---- 3. Volatility circuit breaker ----
            # If current ATR is > 3x its 50-period median, the market is in
            # an extreme regime the model was not trained on. Force HOLD.
            atr_col = "atr_norm" if "atr_norm" in X_live.columns else None
            if atr_col:
                current_atr  = float(X_live[atr_col].iloc[-1])
                featured_all = eng.build(raw_live)
                median_atr   = float(featured_all[atr_col].median())
                if current_atr > 3 * median_atr:
                    logger.warning(
                        f"VOLATILITY CIRCUIT BREAKER — ATR {current_atr:.4f} is "
                        f"{current_atr/median_atr:.1f}x median. Forcing HOLD."
                    )
                    signal, confidence = 0, 0.0
                else:
                    # ---- 3b. Predict ----
                    signal, confidence = trainer.predict_signal(X_live, min_prob=threshold)
            else:
                signal, confidence = trainer.predict_signal(X_live, min_prob=threshold)
            logger.info(f"Signal: {label_map[signal]:4s}  confidence: {confidence:.1%}")

            # ---- 4. Risk check: drawdown kill switch ----
            equity = broker.get_equity()
            drawdown = (equity - start_equity) / start_equity
            if drawdown < -MAX_DRAWDOWN:
                logger.warning(f"KILL SWITCH — drawdown {drawdown:.1%} exceeded {MAX_DRAWDOWN:.1%}. Flattening all.")
                broker.flatten(SYMBOL)
                break

            # ---- 5. Execute ----
            position   = broker.get_position(SYMBOL)
            in_long    = position is not None and position["side"] == "long"
            in_short   = position is not None and position["side"] == "short"
            trade_size = equity * CONFIG.risk.max_position_pct

            if signal == 1 and not in_long:
                if in_short:
                    broker.flatten(SYMBOL)
                broker.buy(SYMBOL, usd=trade_size)

            elif signal == -1 and not in_short:
                if in_long:
                    broker.flatten(SYMBOL)
                # Alpaca crypto supports short selling in paper mode
                broker.sell(SYMBOL, usd=trade_size)

            elif signal == 0 and (in_long or in_short):
                broker.flatten(SYMBOL)

            else:
                logger.info("No trade — signal matches current position or HOLD.")

            # ---- 6. Log state ----
            pos = broker.get_position(SYMBOL)
            if pos:
                logger.info(
                    f"Position: {pos['side'].upper()} ${pos['market_value']:,.2f} "
                    f"| P&L: ${pos['unrealized_pl']:+,.2f} ({float(pos['unrealized_plpc'])*100:+.2f}%)"
                )
            logger.info(f"Equity: ${equity:,.2f}  Drawdown: {drawdown:+.2%}")

        except KeyboardInterrupt:
            logger.info("Shutting down — flattening position.")
            broker.flatten(SYMBOL)
            break
        except Exception as e:
            logger.error(f"Loop error: {e}", exc_info=True)
            logger.info("Sleeping 60s before retry...")
            time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description="Crypto paper trading bot")
    parser.add_argument("--train", action="store_true", help="Fetch data, train model, then run")
    args = parser.parse_args()

    if args.train:
        trainer, eng = build_and_train()
    else:
        trainer, eng = load_trained()

    run_loop(trainer, eng)


if __name__ == "__main__":
    main()
