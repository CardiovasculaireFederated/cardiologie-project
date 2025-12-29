import logging
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger("model_evaluator")


class ModelEvaluatorAgent:
    """
    Evaluates a trained model on a held-out dataset.
    """

    def evaluate(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        model.eval()
        model.to(device)

        correct = 0
        total = 0

        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device).view(-1, 1)
                logits = model(x_batch)
                preds = torch.sigmoid(logits).round()
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        accuracy = correct / max(total, 1)
        xai_score = 0.0
        try:
            xai_score = self._compute_xai_score(model, data_loader, device)
        except Exception as exc:
            logger.warning(f"ModelEvaluator: XAI score failed: {exc}")

        logger.info(
            f"ModelEvaluator: accuracy={accuracy:.4f} xai_score={xai_score:.4f}"
        )
        return {"accuracy": accuracy, "xai_score": xai_score}

    def _compute_xai_score(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        device: torch.device,
    ) -> float:
        batch = self._get_first_batch(data_loader)
        if batch is None:
            return 0.0

        x_batch, _ = batch
        x_batch = x_batch.to(device)
        x_batch = x_batch.detach().clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)

        logits = model(x_batch)
        probs = torch.sigmoid(logits)
        score = probs.mean()
        score.backward()

        gradients = x_batch.grad
        if gradients is None:
            return 0.0

        grad_mean = gradients.abs().mean().item()
        return grad_mean / (grad_mean + 1.0)

    @staticmethod
    def _get_first_batch(
        data_loader: DataLoader,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        try:
            return next(iter(data_loader))
        except StopIteration:
            return None
