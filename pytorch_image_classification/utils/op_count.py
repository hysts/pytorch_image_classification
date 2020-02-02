from typing import Tuple
import thop
import torch
import torch.nn as nn
import yacs.config


def count_op(config: yacs.config.CfgNode, model: nn.Module) -> Tuple[str, str]:
    data = torch.zeros((1, config.dataset.n_channels,
                        config.dataset.image_size, config.dataset.image_size),
                       dtype=torch.float32,
                       device=torch.device(config.device))
    return thop.clever_format(thop.profile(model, (data, ), verbose=False))
