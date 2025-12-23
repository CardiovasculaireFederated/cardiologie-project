# client/sentinel_agent.py

import logging

logger = logging.getLogger("sentinel_agent")


class SentinelAgent:
    """
    Monitors training stability and decides early stopping.
    """

    def __init__(self, patience: int = 3, min_delta: float = 1e-3):
        """
        Args:
            patience: number of epochs without improvement allowed
            min_delta: minimum improvement to be considered progress
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.bad_epochs = 0

    def update(self, val_loss: float) -> bool:
        """
        Update sentinel state.

        Returns:
            True  -> continue training
            False -> stop training
        """

        if self.best_loss is None:
            self.best_loss = val_loss
            return True

        improvement = self.best_loss - val_loss

        if improvement > self.min_delta:
            logger.info(
                f"Sentinel: improvement detected "
                f"(loss {self.best_loss:.4f} → {val_loss:.4f})"
            )
            self.best_loss = val_loss
            self.bad_epochs = 0
            return True

        self.bad_epochs += 1
        logger.warning(
            f"Sentinel: no improvement "
            f"({self.bad_epochs}/{self.patience})"
        )

        if self.bad_epochs >= self.patience:
            logger.error("Sentinel: early stopping triggered")
            return False

        return True
