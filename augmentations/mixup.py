import numpy as np
import torch
import torch.nn as nn


def mixup(data, targets, alpha, n_classes):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    data = data * lam + shuffled_data * (1 - lam)
    targets = (targets, shuffled_targets, lam)

    return data, targets


def mixup_criterion(preds, targets):
    targets1, targets2, lam = targets
    criterion = nn.CrossEntropyLoss(reduction='mean')
    return lam * criterion(preds, targets1) + (1 - lam) * criterion(
        preds, targets2)
