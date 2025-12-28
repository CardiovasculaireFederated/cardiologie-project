import logging
from typing import Dict

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
        logger.info(f"ModelEvaluator: accuracy={accuracy:.4f}")
        return {"accuracy": accuracy}
