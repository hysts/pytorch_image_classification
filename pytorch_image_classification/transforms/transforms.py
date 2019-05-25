from typing import Tuple, Union

import numpy as np
import torch
import torchvision
import yacs.config


class Normalize:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image).astype(np.float32) / 255.
        image = (image - self.mean) / self.std
        return image


class RandomCrop:
    def __init__(self, config: yacs.config.CfgNode):
        self.transform = torchvision.transforms.RandomCrop(
            config.dataset.image_size,
            padding=config.augmentation.random_crop.padding,
            fill=config.augmentation.random_crop.fill,
            padding_mode=config.augmentation.random_crop.padding_mode)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return self.transform(data)


class RandomHorizontalFlip:
    def __init__(self, config: yacs.config.CfgNode):
        self.transform = torchvision.transforms.RandomHorizontalFlip(
            config.augmentation.random_horizontal_flip.prob)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return self.transform(data)


class ToTensor:
    def __call__(self, data: Union[np.ndarray, Tuple[np.ndarray, ...]]
                 ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if isinstance(data, tuple):
            return tuple([self._to_tensor(image) for image in data])
        else:
            return self._to_tensor(data)

    @staticmethod
    def _to_tensor(data: np.ndarray) -> torch.Tensor:
        if len(data.shape) == 3:
            return torch.from_numpy(data.transpose(2, 0, 1).astype(np.float32))
        else:
            return torch.from_numpy(data[None, :, :].astype(np.float32))
