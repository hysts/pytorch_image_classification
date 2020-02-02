from typing import List, Tuple

import numpy as np
import torch
import yacs.config


def mixup(
    batch: Tuple[torch.Tensor, torch.Tensor], alpha: float
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, float]]:
    data, targets = batch
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    data = data * lam + shuffled_data * (1 - lam)
    targets = (targets, shuffled_targets, lam)

    return data, targets


class MixupCollator:
    def __init__(self, config: yacs.config.CfgNode):
        self.alpha = config.augmentation.mixup.alpha

    def __call__(
        self, batch: List[Tuple[torch.Tensor, int]]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, float]]:
        batch = torch.utils.data.dataloader.default_collate(batch)
        batch = mixup(batch, self.alpha)
        return batch
