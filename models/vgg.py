# coding: utf-8

import torch
import torch.nn as nn


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
        module.bias.data.zero_()
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.uniform_()
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()

        input_shape = config['input_shape']
        n_classes = config['n_classes']

        self.use_bn = config['use_bn']
        n_channels = config['n_channels']
        n_layers = config['n_layers']

        self.stage1 = self._make_stage(input_shape[1], n_channels[0],
                                       n_layers[0])
        self.stage2 = self._make_stage(n_channels[0], n_channels[1],
                                       n_layers[1])
        self.stage3 = self._make_stage(n_channels[1], n_channels[2],
                                       n_layers[2])
        self.stage4 = self._make_stage(n_channels[2], n_channels[3],
                                       n_layers[3])
        self.stage5 = self._make_stage(n_channels[3], n_channels[4],
                                       n_layers[4])

        # compute conv feature size
        with torch.no_grad():
            self.feature_size = self._forward_conv(
                torch.zeros(*input_shape)).view(-1).shape[0]

        self.fc = nn.Linear(self.feature_size, n_classes)

        # initialize weights
        self.apply(initialize_weights)

    def _make_stage(self, in_channels, out_channels, n_blocks):
        stage = nn.Sequential()
        for index in range(n_blocks):
            if index == 0:
                conv = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            else:
                conv = nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            stage.add_module('conv{}'.format(index), conv)
            if self.use_bn:
                stage.add_module('bn{}'.format(index),
                                 nn.BatchNorm2d(out_channels))
            stage.add_module('relu', nn.ReLU(inplace=True))
        stage.add_module('pool', nn.MaxPool2d(kernel_size=2, stride=2))
        return stage

    def _forward_conv(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
