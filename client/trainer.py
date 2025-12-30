# client/trainer.py

from client.sentinel_agent import SentinelAgent
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


def train_model(model, train_loader, val_loader, epochs, learning_rate, device):
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    sentinel = SentinelAgent(patience=3, min_delta=1e-3)

    best_val_acc = 0.0
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(epochs):
        # ---- Train ----
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, loss_fn, device
        )

        # ---- Validate ----
        val_loss, val_acc = evaluate(
            model, val_loader, loss_fn, device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"TrainLoss={train_loss:.4f} Acc={train_acc:.4f} | "
            f"ValLoss={val_loss:.4f} Acc={val_acc:.4f}"
        )

        # ---- Best model ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # ---- Sentinel check ----
        if not sentinel.update(val_loss):
            logger.warning(
                f"Training stopped early at epoch {epoch+1}"
            )
            break

    return {
        "weights": model.state_dict(),
        "best_val_acc": best_val_acc,
        "history": history,
    }
