import copy
import logging
import random
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

logger = logging.getLogger("scout_agent")


class ScoutAgent:
    def __init__(
        self,
        candidate_lrs: Optional[List[float]] = None,
        base_epochs: int = 5,
        quick_epochs: int = 1,
        subset_ratio: float = 0.05,
        max_batches: int = 10,
        random_seed: int = 42,
    ) -> None:
        self.candidate_lrs = candidate_lrs or [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
        self.base_epochs = base_epochs
        self.quick_epochs = max(1, quick_epochs)
        self.subset_ratio = max(0.01, min(subset_ratio, 1.0))
        self.max_batches = max(1, max_batches)
        self.random_seed = random_seed

    def suggest_hyperparameters(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        history: List[Dict[str, Any]],
        device: torch.device,
    ) -> Dict[str, Any]:
        best_lr = self._search_best_lr(model, train_loader, val_loader, device)
        if best_lr is None:
            logger.warning("ScoutAgent: fallback to heuristic defaults")
            best_lr = self._heuristic_lr(history)

        epochs = self._heuristic_epochs(history)
        return {"learning_rate": best_lr, "epochs": epochs}

    def _search_best_lr(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ) -> Optional[float]:
        try:
            train_subset = self._build_subset_loader(train_loader)
            val_subset = self._build_subset_loader(val_loader)
        except Exception as exc:
            logger.error(f"ScoutAgent: failed to build subset loaders: {exc}")
            return None

        initial_weights = copy.deepcopy(model.state_dict())
        best_lr: Optional[float] = None
        best_acc = -1.0

        for lr in self.candidate_lrs:
            model.load_state_dict(initial_weights)
            val_acc = self._quick_train_eval(
                model=model,
                train_loader=train_subset,
                val_loader=val_subset,
                learning_rate=lr,
                device=device,
            )
            logger.info(f"ScoutAgent: lr={lr} -> val_acc={val_acc:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                best_lr = lr

        model.load_state_dict(initial_weights)
        return best_lr

    def _quick_train_eval(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float,
        device: torch.device,
    ) -> float:
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.BCEWithLogitsLoss()

        for _ in range(self.quick_epochs):
            model.train()
            for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device).view(-1, 1)
                optimizer.zero_grad()
                logits = model(x_batch)
                loss = loss_fn(logits, y_batch)
                loss.backward()
                optimizer.step()
                if batch_idx + 1 >= self.max_batches:
                    break

        return self._quick_evaluate(model, val_loader, device)

    def _quick_evaluate(
        self,
        model: torch.nn.Module,
        val_loader: DataLoader,
        device: torch.device,
    ) -> float:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (x_batch, y_batch) in enumerate(val_loader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device).view(-1, 1)
                logits = model(x_batch)
                preds = torch.sigmoid(logits).round()
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
                if batch_idx + 1 >= self.max_batches:
                    break
        return correct / max(total, 1)

    def _build_subset_loader(self, loader: DataLoader) -> DataLoader:
        dataset = loader.dataset
        total = len(dataset)
        subset_size = max(1, int(total * self.subset_ratio))
        subset_size = min(subset_size, total)

        rng = random.Random(self.random_seed)
        if subset_size < total:
            indices = rng.sample(range(total), subset_size)
        else:
            indices = list(range(total))

        subset = Subset(dataset, indices)
        batch_size = loader.batch_size or 32
        return DataLoader(subset, batch_size=batch_size, shuffle=True)

    def _heuristic_lr(self, history: List[Dict[str, Any]]) -> float:
        if len(history) < 2:
            return 1e-3

        last = history[-1]
        prev = history[-2]
        last_acc = float(last.get("best_val_acc") or 0.0)
        prev_acc = float(prev.get("best_val_acc") or 0.0)
        delta = last_acc - prev_acc
        lr = float(last.get("learning_rate") or 1e-3)

        if delta > 0.01:
            return lr
        if -0.01 <= delta <= 0.01:
            return lr
        return max(lr * 0.5, 1e-5)

    def _heuristic_epochs(self, history: List[Dict[str, Any]]) -> int:
        if len(history) < 2:
            return self.base_epochs

        last = history[-1]
        prev = history[-2]
        last_acc = float(last.get("best_val_acc") or 0.0)
        prev_acc = float(prev.get("best_val_acc") or 0.0)
        delta = last_acc - prev_acc
        epochs = int(last.get("epochs") or self.base_epochs)

        if -0.01 <= delta <= 0.01:
            return min(epochs + 2, 15)
        return epochs


def suggest_hyperparameters(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    agent = ScoutAgent()
    return {
        "learning_rate": agent._heuristic_lr(history),
        "epochs": agent._heuristic_epochs(history),
    }
