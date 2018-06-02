# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,  # downsample with first conv
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride > 1:
            self.shortcut = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        y = self.bn1(x)
        y = self.conv1(y)

        y = F.relu(self.bn2(y), inplace=True)
        y = self.conv2(y)

        y = self.bn3(y)

        if y.size(1) != x.size(1):
            y += F.pad(
                self.shortcut(x), (0, 0, 0, 0, 0, y.size(1) - x.size(1)),
                'constant', 0)
        else:
            y += self.shortcut(x)
        return y


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride):
        super(BottleneckBlock, self).__init__()

        bottleneck_channels = out_channels // self.expansion

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,  # downsample with 3x3 conv
            padding=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)

        self.bn4 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()  # identity
        if stride > 1:
            self.shortcut = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        y = self.bn1(x)
        y = self.conv1(y)

        y = F.relu(self.bn2(y), inplace=True)
        y = self.conv2(y)

        y = F.relu(self.bn3(y), inplace=True)
        y = self.conv3(y)

        y = self.bn4(y)

        if y.size(1) != x.size(1):
            y += F.pad(
                self.shortcut(x), (0, 0, 0, 0, 0, y.size(1) - x.size(1)),
                'constant', 0)
        else:
            y += self.shortcut(x)
        return y


class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()

        input_shape = config['input_shape']
        n_classes = config['n_classes']

        base_channels = config['base_channels']
        block_type = config['block_type']
        depth = config['depth']

        alpha = config['pyramid_alpha']

        assert block_type in ['basic', 'bottleneck']
        if block_type == 'basic':
            block = BasicBlock
            n_blocks_per_stage = (depth - 2) // 6
            assert n_blocks_per_stage * 6 + 2 == depth
        else:
            block = BottleneckBlock
            n_blocks_per_stage = (depth - 2) // 9
            assert n_blocks_per_stage * 9 + 2 == depth

        n_channels = [base_channels]
        for _ in range(n_blocks_per_stage * 3):
            num = n_channels[-1] + alpha / (n_blocks_per_stage * 3)
            n_channels.append(num)
        n_channels = [int(np.round(c)) * block.expansion for c in n_channels]

        self.conv = nn.Conv2d(
            input_shape[1],
            n_channels[0],
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(n_channels[0])

        self.stage1 = self._make_stage(
            n_channels[:n_blocks_per_stage + 1],
            n_blocks_per_stage,
            block,
            stride=1)
        self.stage2 = self._make_stage(
            n_channels[n_blocks_per_stage:n_blocks_per_stage * 2 + 1],
            n_blocks_per_stage,
            block,
            stride=2)
        self.stage3 = self._make_stage(
            n_channels[n_blocks_per_stage * 2:],
            n_blocks_per_stage,
            block,
            stride=2)

        self.bn2 = nn.BatchNorm2d(n_channels[-1])

        # compute conv feature size
        with torch.no_grad():
            self.feature_size = self._forward_conv(
                torch.zeros(*input_shape)).view(-1).shape[0]

        self.fc = nn.Linear(self.feature_size, n_classes)

        # initialize weights
        self.apply(initialize_weights)

    def _make_stage(self, n_channels, n_blocks, block, stride):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = 'block{}'.format(index + 1)
            if index == 0:
                stage.add_module(
                    block_name,
                    block(
                        n_channels[index],
                        n_channels[index + 1],
                        stride=stride))
            else:
                stage.add_module(
                    block_name,
                    block(n_channels[index], n_channels[index + 1], stride=1))
        return stage

    def _forward_conv(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.relu(
            self.bn2(x),
            inplace=True)  # apply BN and ReLU before average pooling
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
