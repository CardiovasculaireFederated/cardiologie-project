# client/sentinel_agent.py

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("sentinel_agent")


class SentinelAgent:
    """
    Decision engine that evaluates training stability and update quality.
    """

    DEFAULT_CONFIG = {
        "min_epochs": 3,
        "min_acc_delta": 0.002,
        "max_loss_increase": 0.5,
        "patience": 2,
        "min_baseline_acc": 0.5,
    }

    def __init__(self, patience: int = 3, min_delta: float = 1e-3):
        """
        Args:
            patience: number of epochs without improvement allowed (legacy update)
            min_delta: minimum improvement to be considered progress (legacy update)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: Optional[float] = None
        self.bad_epochs = 0

    def update(self, val_loss: float) -> bool:
        """
        Legacy early-stopping helper for existing trainer integration.

        Returns:
            True  -> continue training
            False -> stop training
        """
        if not math.isfinite(val_loss):
            logger.error("Sentinel: invalid loss value detected")
            return False

        if self.best_loss is None:
            self.best_loss = val_loss
            return True

        improvement = self.best_loss - val_loss

        if improvement > self.min_delta:
            logger.info(
                "Sentinel: improvement detected (loss %.4f -> %.4f)",
                self.best_loss,
                val_loss,
            )
            self.best_loss = val_loss
            self.bad_epochs = 0
            return True

        self.bad_epochs += 1
        logger.warning(
            "Sentinel: no improvement (%s/%s)",
            self.bad_epochs,
            self.patience,
        )

        if self.bad_epochs >= self.patience:
            logger.error("Sentinel: early stopping triggered")
            return False

        return True

    def evaluate_training(
        self,
        train_losses: List[float],
        val_accuracies: List[float],
        current_epoch: int,
        round_id: int,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Decide whether to accept, early_stop, or reject a local update.
        """
        cfg = dict(self.DEFAULT_CONFIG)
        if config:
            cfg.update(config)

        min_epochs = int(cfg["min_epochs"])
        min_acc_delta = float(cfg["min_acc_delta"])
        max_loss_increase = float(cfg["max_loss_increase"])
        patience = int(cfg["patience"])
        min_baseline_acc = float(cfg["min_baseline_acc"])

        last_train_loss, loss_delta = self._last_and_delta(train_losses)
        last_val_acc, acc_delta = self._last_and_delta(val_accuracies)

        metrics = {
            "last_train_loss": last_train_loss,
            "last_val_acc": last_val_acc,
            "acc_delta": acc_delta,
            "loss_delta": loss_delta,
        }

        if current_epoch < 0:
            return self._decision(
                "reject",
                "invalid_current_epoch",
                metrics,
                round_id,
            )

        if not self._has_valid_metrics(last_train_loss, last_val_acc):
            return self._decision(
                "reject",
                "invalid_metrics",
                metrics,
                round_id,
            )

        if last_val_acc < min_baseline_acc:
            return self._decision(
                "reject",
                "accuracy_below_baseline",
                metrics,
                round_id,
            )

        acc_drop = acc_delta <= -(min_acc_delta * 2.0)
        loss_spike = loss_delta >= max_loss_increase

        if acc_drop or loss_spike:
            reason = "accuracy_drop" if acc_drop else "loss_spike"
            return self._decision("reject", reason, metrics, round_id)

        if current_epoch < min_epochs:
            return self._decision(
                "early_stop",
                "min_epochs_not_reached",
                metrics,
                round_id,
            )

        acc_improving = acc_delta >= min_acc_delta
        loss_improving = loss_delta <= 0.0

        if acc_improving and (loss_improving or loss_delta <= max_loss_increase):
            return self._decision("accept", "training_improving", metrics, round_id)

        acc_plateau, loss_plateau = self._plateau_detected(
            train_losses,
            val_accuracies,
            min_acc_delta,
            patience,
        )

        if acc_plateau and loss_plateau:
            return self._decision(
                "early_stop",
                "plateau_detected",
                metrics,
                round_id,
            )

        return self._decision(
            "early_stop",
            "no_clear_improvement",
            metrics,
            round_id,
        )

    def _last_and_delta(self, values: List[float]) -> Tuple[float, float]:
        if not values:
            return float("nan"), float("nan")
        if len(values) == 1:
            return float(values[-1]), 0.0
        return float(values[-1]), float(values[-1] - values[-2])

    def _plateau_detected(
        self,
        train_losses: List[float],
        val_accuracies: List[float],
        min_acc_delta: float,
        patience: int,
    ) -> Tuple[bool, bool]:
        if patience <= 0:
            return False, False

        acc_deltas = self._recent_deltas(val_accuracies, patience)
        loss_deltas = self._recent_deltas(train_losses, patience)

        acc_plateau = bool(acc_deltas) and all(
            abs(delta) < min_acc_delta for delta in acc_deltas
        )
        loss_plateau = bool(loss_deltas) and all(delta >= 0.0 for delta in loss_deltas)

        return acc_plateau, loss_plateau

    def _recent_deltas(self, values: List[float], n: int) -> List[float]:
        if len(values) < n + 1:
            return []
        start = len(values) - n - 1
        deltas = []
        for idx in range(start, len(values) - 1):
            deltas.append(float(values[idx + 1] - values[idx]))
        return deltas

    def _has_valid_metrics(self, last_train_loss: float, last_val_acc: float) -> bool:
        return math.isfinite(last_train_loss) and math.isfinite(last_val_acc)

    def _decision(
        self,
        decision: str,
        reason: str,
        metrics: Dict[str, float],
        round_id: int,
    ) -> Dict[str, Any]:
        logger.info(
            "Sentinel decision=%s round=%s reason=%s metrics=%s",
            decision,
            round_id,
            reason,
            metrics,
        )
        return {
            "decision": decision,
            "reason": reason,
            "metrics": metrics,
        }
