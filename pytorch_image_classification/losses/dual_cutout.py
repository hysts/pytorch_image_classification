import torch
import torch.nn as nn
import torch.nn.functional as F
import yacs.config


class DualCutoutLoss:
    def __init__(self, config: yacs.config.CfgNode, reduction: str):
        self.alpha = config.augmentation.cutout.dual_cutout_alpha
        self.loss_func = nn.CrossEntropyLoss(reduction=reduction)

    def __call__(self, predictions: torch.Tensor,
                 targets: torch.Tensor) -> torch.Tensor:
        predictions1, predictions2 = predictions[:, 0], predictions[:, 1]
        return (self.loss_func(predictions1, targets) + self.loss_func(
            predictions2, targets)) * 0.5 + self.alpha * F.mse_loss(
                predictions1, predictions2)
