import torch
import torch.nn as nn
import torch.nn.functional as F
import yacs.config


class DualCutoutLoss:
    def __init__(self, config: yacs.config.CfgNode, reduction: str):
        self.alpha = config.augmentation.cutout.dual_cutout_alpha
        self.loss_func = nn.CrossEntropyLoss(reduction=reduction)

    def __call__(self, preds: torch.Tensor,
                 targets: torch.Tensor) -> torch.Tensor:
        preds1, preds2 = preds[:, 0], preds[:, 1]
        return (self.loss_func(preds1, targets) + self.loss_func(
            preds2, targets)) * 0.5 + self.alpha * F.mse_loss(preds1, preds2)
