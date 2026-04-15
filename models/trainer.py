"""
models/trainer.py
-----------------
Trains a LightGBM classifier on the engineered dataset.

Target: target_direction ∈ {-1 (SELL), 0 (HOLD), 1 (BUY)}

Evaluation hierarchy:
  1. MCC (Matthews Correlation Coefficient) — primary metric for training
     evaluation and model selection. Best for imbalanced multi-class problems.
     Cannot be gamed by predicting the majority class.
  2. Simulated Sharpe ratio — used for model promotion during weekly retraining.
     Simulates what the val-set signals would have returned, risk-adjusted.
  3. Precision / F1 / classification report — logged for reference only.
"""

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    classification_report,
    matthews_corrcoef,
    precision_score,
    f1_score,
)

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "models" / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# LightGBM maps -1/0/1 → 0/1/2 internally
LABEL_MAP     = {-1: 0, 0: 1, 1: 2}
LABEL_MAP_INV = {0: -1, 1: 0, 2: 1}


class ModelTrainer:

    def __init__(self, model_path: str = "lgbm_model.pkl"):
        self.model_path = MODEL_DIR / model_path
        self.model: lgb.LGBMClassifier | None = None
        self.feature_cols: list[str] = []

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val:   pd.DataFrame,
        y_val:   pd.Series,
    ) -> dict:
        """
        Train LightGBM with early stopping on val loss.
        Returns a dict with mcc, sharpe, precision, f1.
        """
        self.feature_cols = list(X_train.columns)

        y_tr = y_train.map(LABEL_MAP)
        y_v  = y_val.map(LABEL_MAP)

        self.model = lgb.LGBMClassifier(
            n_estimators=2000,          # capped by early stopping
            learning_rate=0.011,        # tuned: slower learning generalises better
            max_depth=6,                # tuned: confirmed optimal
            num_leaves=67,              # tuned
            min_child_samples=139,      # tuned: higher = less overfit on small leaves
            subsample=0.999,            # tuned
            colsample_bytree=0.650,     # tuned
            min_gain_to_split=0.056,    # tuned
            reg_alpha=0.010,            # tuned: L1 regularisation
            reg_lambda=0.896,           # tuned: L2 regularisation
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

        self.model.fit(
            X_train, y_tr,
            eval_set=[(X_val, y_v)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )

        preds     = pd.Series(self.model.predict(X_val)).map(LABEL_MAP_INV)
        y_orig    = y_val.values

        # --- Classification report (reference only) ---
        report = classification_report(
            y_orig, preds,
            target_names=["SELL", "HOLD", "BUY"],
            output_dict=True,
        )
        logger.info("\n" + classification_report(y_orig, preds, target_names=["SELL", "HOLD", "BUY"]))

        # --- Primary metric: MCC ---
        mcc = float(matthews_corrcoef(y_orig, preds))

        # --- Promotion metric: simulated Sharpe ---
        sharpe = self._simulate_sharpe(y_val, preds)

        # --- Reference metrics ---
        precision = float(precision_score(y_orig, preds, average="macro", zero_division=0))
        f1        = float(f1_score(y_orig, preds, average="macro", zero_division=0))

        fi = self._feature_importance()
        logger.info(f"Top 10 features:\n{fi.head(10).to_string()}")
        logger.info(
            f"MCC: {mcc:.4f}  |  Simulated Sharpe: {sharpe:.4f}  |  "
            f"Precision: {precision:.4f}  |  F1: {f1:.4f}"
        )

        # Calibrate optimal confidence threshold on val set
        optimal_threshold, threshold_sharpe = self._calibrate_threshold(X_val, y_val)
        logger.info(
            f"Optimal confidence threshold: {optimal_threshold:.2f}  "
            f"(Sharpe at threshold: {threshold_sharpe:.4f})"
        )

        return {
            **report,
            "mcc":                mcc,
            "sharpe":             sharpe,
            "precision":          precision,
            "f1":                 f1,
            "optimal_threshold":  optimal_threshold,
            "threshold_sharpe":   threshold_sharpe,
        }

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns probability array shape (n, 3): [P(SELL), P(HOLD), P(BUY)]"""
        if self.model is None:
            raise RuntimeError("Model not trained or loaded.")
        return self.model.predict_proba(X[self.feature_cols])

    def predict_signal(self, X: pd.DataFrame, min_prob: float = 0.60) -> tuple[int, float]:
        """
        Returns (signal, confidence) where signal ∈ {-1, 0, 1}.
        Returns (0, prob) if max confidence < min_prob — stay flat.
        """
        proba      = self.predict_proba(X)[0]
        best_idx   = int(np.argmax(proba))
        confidence = float(proba[best_idx])
        signal     = LABEL_MAP_INV[best_idx]

        if min_prob > 0.0 and confidence < min_prob:
            return 0, confidence
        return signal, confidence

    def _calibrate_threshold(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        min_trades: int = 30,
    ) -> tuple[float, float]:
        """
        Grid-search confidence thresholds on the val set.
        Returns (best_threshold, best_sharpe).

        Includes threshold=0 (trade every signal, no filtering) as a
        candidate so that if filtering never helps, we fall back cleanly
        rather than forcing a threshold that hurts performance.
        """
        proba      = self.predict_proba(X_val)
        best_threshold = 0.0
        best_sharpe    = float("-inf")

        # 0.0 = no filtering (trade every argmax signal)
        thresholds = [0.0] + list(np.arange(0.50, 0.86, 0.05))

        for threshold in thresholds:
            max_proba  = proba.max(axis=1)
            best_class = proba.argmax(axis=1)
            signals    = np.where(max_proba >= threshold,
                                  pd.Series(best_class).map(LABEL_MAP_INV),
                                  0)
            n_trades = (signals != 0).sum()
            if n_trades < min_trades:
                continue

            preds_s = pd.Series(signals, index=y_val.index)
            sharpe  = self._simulate_sharpe(y_val, preds_s)
            if sharpe > best_sharpe:
                best_sharpe    = sharpe
                best_threshold = round(float(threshold), 2)

        return best_threshold, best_sharpe

    def save(self, mcc: float = 0.0, sharpe: float = 0.0,
             precision: float = 0.0, f1: float = 0.0,
             optimal_threshold: float = 0.60, n_train: int = 0):
        with open(self.model_path, "wb") as f:
            pickle.dump({"model": self.model, "feature_cols": self.feature_cols}, f)
        meta = {
            "saved_at":           datetime.now(timezone.utc).isoformat(),
            "mcc":                round(mcc, 4),
            "sharpe":             round(sharpe, 4),
            "precision":          round(precision, 4),
            "f1":                 round(f1, 4),
            "optimal_threshold":  round(optimal_threshold, 2),
            "n_train":            n_train,
        }
        with open(self.model_path.with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(
            f"Model saved → {self.model_path}  "
            f"(MCC={mcc:.4f}, Sharpe={sharpe:.4f}, threshold={optimal_threshold:.2f}, n_train={n_train})"
        )

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"No saved model at {self.model_path}")
        with open(self.model_path, "rb") as f:
            data = pickle.load(f)
        self.model        = data["model"]
        self.feature_cols = data["feature_cols"]
        meta_path = self.model_path.with_suffix(".json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            logger.info(
                f"Model loaded ← {self.model_path}  "
                f"(saved={meta['saved_at']}, MCC={meta.get('mcc', '?')}, "
                f"Sharpe={meta.get('sharpe', '?')}, n_train={meta['n_train']})"
            )
        else:
            logger.info(f"Model loaded ← {self.model_path}")

    def current_sharpe(self) -> float:
        """Return the simulated Sharpe of the current saved model, or -inf."""
        meta_path = self.model_path.with_suffix(".json")
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f).get("sharpe", float("-inf"))
        return float("-inf")

    def current_mcc(self) -> float:
        """Return the MCC of the current saved model, or -1."""
        meta_path = self.model_path.with_suffix(".json")
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f).get("mcc", -1.0)
        return -1.0

    def optimal_threshold(self) -> float:
        """Return the calibrated confidence threshold, or 0.65 as fallback."""
        meta_path = self.model_path.with_suffix(".json")
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f).get("optimal_threshold", 0.65)
        return 0.65

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _simulate_sharpe(y_true: pd.Series, y_pred: pd.Series,
                         annualize: float = np.sqrt(6 * 365)) -> float:
        """
        Simulate long-only strategy returns on the val set and compute annualized Sharpe.

        Matches live bot behaviour — no shorting:
          - BUY  (+1) → enter long (if flat)
          - SELL (-1) → exit long (if in position), otherwise do nothing
          - HOLD ( 0) → keep current state

        Returns are accumulated only while in a long position.
        """
        actual_returns = y_true.values * 0.005   # ~0.5% per label as return proxy
        signals        = y_pred.values

        strategy_rets = []
        in_long = False
        for sig, ret in zip(signals, actual_returns):
            if sig == 1:
                in_long = True
            elif sig == -1:
                in_long = False
            if in_long:
                strategy_rets.append(ret)

        active = np.array(strategy_rets)
        if len(active) < 10:
            return 0.0

        mean = active.mean()
        std  = active.std(ddof=1)
        if std == 0:
            return 0.0

        return float((mean / std) * annualize)

    def _feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame({
            "feature":    self.feature_cols,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
