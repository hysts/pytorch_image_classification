from typing import Callable

import torch
import yacs.config

from .cutmix import CutMixCollator
from .mixup import MixupCollator
from .ricap import RICAPCollator


def create_collator(config: yacs.config.CfgNode) -> Callable:
    if config.augmentation.use_mixup:
        return MixupCollator(config)
    elif config.augmentation.use_ricap:
        return RICAPCollator(config)
    elif config.augmentation.use_cutmix:
        return CutMixCollator(config)
    else:
        return torch.utils.data.dataloader.default_collate
