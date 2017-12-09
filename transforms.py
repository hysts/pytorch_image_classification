# coding: utf-8

import numpy as np
import torch


def to_tensor():
    def _to_tensor(image):
        if len(image.shape) == 3:
            return torch.from_numpy(
                image.transpose(2, 0, 1).astype(np.float32))
        else:
            return torch.from_numpy(image[None, :, :].astype(np.float32))

    return _to_tensor


def normalize(mean, std):
    mean = np.array(mean)
    std = np.array(std)

    def _normalize(image):
        image = np.asarray(image).astype(np.float32) / 255.
        image = (image - mean) / std
        return image

    return _normalize


def cutout(mask_size, p, cutout_inside, mask_color=0):
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    def _cutout(image):
        image = np.asarray(image).copy()

        if np.random.random() > p:
            return image

        h, w = image.shape[:2]

        if cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax] = mask_color
        return image

    return _cutout


def random_erasing(p, area_ratio_range, min_aspect_ratio, max_attempt=20):
    sl, sh = area_ratio_range
    rl, rh = min_aspect_ratio, 1. / min_aspect_ratio

    def _random_erasing(image):
        image = np.asarray(image).copy()

        if np.random.random() > p:
            return image

        h, w = image.shape[:2]
        image_area = h * w

        for _ in range(max_attempt):
            mask_area = np.random.uniform(sl, sh) * image_area
            aspect_ratio = np.random.uniform(rl, rh)
            mask_h = int(np.sqrt(mask_area * aspect_ratio))
            mask_w = int(np.sqrt(mask_area / aspect_ratio))

            if mask_w < w and mask_h < h:
                x0 = np.random.randint(0, w - mask_w)
                y0 = np.random.randint(0, h - mask_h)
                x1 = x0 + mask_w
                y1 = y0 + mask_h
                image[y0:y1, x0:x1] = np.random.uniform(0, 1)
                break

        return image

    return _random_erasing
