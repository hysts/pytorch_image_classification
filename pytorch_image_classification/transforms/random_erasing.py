import numpy as np
import yacs.config


class RandomErasing:
    def __init__(self, config: yacs.config.CfgNode):
        aug_config = config.augmentation.random_erasing
        self.p = aug_config.prob
        self.max_attempt = aug_config.max_attempt
        self.sl, self.sh = aug_config.area_ratio_range
        self.rl = aug_config.min_aspect_ratio
        self.rh = 1. / aug_config.min_aspect_ratio

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image).copy()

        if np.random.random() > self.p:
            return image

        h, w = image.shape[:2]
        image_area = h * w

        for _ in range(self.max_attempt):
            mask_area = np.random.uniform(self.sl, self.sh) * image_area
            aspect_ratio = np.random.uniform(self.rl, self.rh)
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
