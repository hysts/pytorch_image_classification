import numpy as np
import torch
import torch.nn as nn


def ricap(data, targets, beta, n_classes):
    image_h, image_w = data.shape[2:]
    ratio = np.random.beta(beta, beta, size=2)
    w0, h0 = np.round(np.array([image_w, image_h]) * ratio).astype(np.int)
    w1, h1 = image_w - w0, image_h - h0
    ws = [w0, w1, w0, w1]
    hs = [h0, h0, h1, h1]

    patches = []
    labels = []
    label_weights = []
    for w, h in zip(ws, hs):
        indices = torch.randperm(data.size(0))
        x0 = np.random.randint(0, image_w - w + 1)
        y0 = np.random.randint(0, image_h - h + 1)
        patches.append(data[indices, :, y0:y0 + h, x0:x0 + w])
        labels.append(targets[indices])
        label_weights.append(h * w / (image_h * image_w))

    data = torch.cat(
        [torch.cat(patches[:2], dim=3),
         torch.cat(patches[2:], dim=3)], dim=2)
    targets = (labels, label_weights)

    return data, targets


def ricap_criterion(preds, targets):
    target_list, weights = targets
    criterion = nn.CrossEntropyLoss(reduction='mean')
    return sum([
        weight * criterion(preds, targets)
        for targets, weight in zip(target_list, weights)
    ])
