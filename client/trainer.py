# client/trainer.py

import logging
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from client.sentinel_agent import SentinelAgent

logger = logging.getLogger(__name__)


def train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).view(-1, 1)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        preds = torch.sigmoid(logits).round()
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).view(-1, 1)

            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            preds = torch.sigmoid(logits).round()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> Dict[str, Any]:
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
