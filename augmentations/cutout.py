import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Cutout:
    def __init__(self, mask_size, p, cutout_inside, mask_color=0):
        self.p = p
        self.mask_size = mask_size
        self.cutout_inside = cutout_inside
        self.mask_color = mask_color

        self.mask_size_half = mask_size // 2
        self.offset = 1 if mask_size % 2 == 0 else 0

    def __call__(self, image):
        image = np.asarray(image).copy()

        if np.random.random() > self.p:
            return image

        h, w = image.shape[:2]

        if self.cutout_inside:
            cxmin, cxmax = self.mask_size_half, w + self.offset - self.mask_size_half
            cymin, cymax = self.mask_size_half, h + self.offset - self.mask_size_half
        else:
            cxmin, cxmax = 0, w + self.offset
            cymin, cymax = 0, h + self.offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - self.mask_size_half
        ymin = cy - self.mask_size_half
        xmax = xmin + self.mask_size
        ymax = ymin + self.mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax] = self.mask_color
        return image


class DualCutout:
    def __init__(self, mask_size, p, cutout_inside, mask_color=0):
        self.cutout = Cutout(mask_size, p, cutout_inside, mask_color)

    def __call__(self, image):
        return np.hstack([self.cutout(image), self.cutout(image)])


class DualCutoutCriterion:
    def __init__(self, alpha):
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def __call__(self, preds, targets):
        preds1, preds2 = preds
        return (self.criterion(preds1, targets) + self.criterion(
            preds2, targets)) * 0.5 + self.alpha * F.mse_loss(preds1, preds2)
