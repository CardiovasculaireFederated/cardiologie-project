# client/data_validator.py

import torch
import logging

logger = logging.getLogger("data_validator")


class DataValidatorAgent:
    """
    Validates local dataset before training.
    """

    def __init__(
        self,
        min_samples: int = 100,
        expected_features: int = 13,
        max_nan_ratio: float = 0.05,
        allow_single_label: bool = False,
    ):
        self.min_samples = min_samples
        self.expected_features = expected_features
        self.max_nan_ratio = max_nan_ratio
        self.allow_single_label = allow_single_label

    def validate(self, train_loader):
        dataset = train_loader.dataset
        samples = len(dataset)

        #  Dataset size
        if samples < self.min_samples:
            return False, f"Not enough samples ({samples})"

        X, y = dataset.tensors

        #  Feature shape
        if X.shape[1] != self.expected_features:
            return False, (
                f"Invalid feature count "
                f"(expected {self.expected_features}, got {X.shape[1]})"
            )

        #  Labels sanity
        unique_labels, label_counts = torch.unique(y, return_counts=True)
        if unique_labels.numel() < 2:
            label_value = unique_labels[0].item() if unique_labels.numel() else None
            label_count = label_counts[0].item() if label_counts.numel() else 0
            message = (
                f"Only one label present (label={label_value}, count={label_count})"
            )
            if self.allow_single_label:
                logger.warning(message)
            else:
                return False, message

        #  NaN / Inf ratio
        nan_ratio = torch.isnan(X).float().mean().item()
        inf_ratio = torch.isinf(X).float().mean().item()

        if nan_ratio > self.max_nan_ratio or inf_ratio > self.max_nan_ratio:
            return False, (
                f"Too many NaN/Inf values "
                f"(nan={nan_ratio:.2%}, inf={inf_ratio:.2%})"
            )

        # 5Class balance (warning only)
        pos_ratio = y.mean().item()
        if pos_ratio < 0.1 or pos_ratio > 0.9:
            logger.warning(
                f"Class imbalance detected (positive ratio={pos_ratio:.2f})"
            )

        return True, "Data validation passed"
