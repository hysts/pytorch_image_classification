from typing import Tuple

import torch
import torch.nn as nn


class MixupLoss:
    def __init__(self, reduction: str):
        self.loss_func = nn.CrossEntropyLoss(reduction=reduction)

    def __call__(
            self, predictions: torch.Tensor,
            targets: Tuple[torch.Tensor, torch.Tensor, float]) -> torch.Tensor:
        targets1, targets2, lam = targets
        return lam * self.loss_func(predictions, targets1) + (
            1 - lam) * self.loss_func(predictions, targets2)
