from typing import Callable

import torch.nn as nn


def create_initializer(mode: str) -> Callable:
    if mode in ['kaiming_fan_out', 'kaiming_fan_in']:
        mode = mode[8:]

        def initializer(module):
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight.data,
                                        mode=mode,
                                        nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight.data, 1)
                nn.init.constant_(module.bias.data, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data,
                                        mode=mode,
                                        nonlinearity='relu')
                nn.init.constant_(module.bias.data, 0)
    else:
        raise ValueError()

    return initializer
