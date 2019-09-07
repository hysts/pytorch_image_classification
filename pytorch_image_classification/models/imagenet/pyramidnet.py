import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..initializer import create_initializer


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,  # downsample with first conv
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,
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
            y += F.pad(self.shortcut(x),
                       (0, 0, 0, 0, 0, y.size(1) - x.size(1)), 'constant', 0)
        else:
            y += self.shortcut(x)
        return y


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        bottleneck_channels = out_channels // self.expansion

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels,
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
        self.conv3 = nn.Conv2d(bottleneck_channels,
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
            y += F.pad(self.shortcut(x),
                       (0, 0, 0, 0, 0, y.size(1) - x.size(1)), 'constant', 0)
        else:
            y += self.shortcut(x)
        return y


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()

        model_config = config.model.pyramidnet
        initial_channels = model_config.initial_channels
        block_type = model_config.block_type
        n_blocks = model_config.n_blocks
        alpha = model_config.alpha

        assert block_type in ['basic', 'bottleneck']
        if block_type == 'basic':
            block = BasicBlock
        else:
            block = BottleneckBlock

        n_channels = [initial_channels]
        depth = sum(n_blocks)
        rate = alpha / depth
        for _ in range(depth):
            num = n_channels[-1] + rate
            n_channels.append(num)
        n_channels = [int(np.round(c)) * block.expansion for c in n_channels]
        n_channels[0] //= block.expansion

        self.conv = nn.Conv2d(config.dataset.n_channels,
                              n_channels[0],
                              kernel_size=7,
                              stride=2,
                              padding=3,
                              bias=False)
        self.bn = nn.BatchNorm2d(n_channels[0])

        accs = [n_blocks[0]]
        for i in range(1, 4):
            accs.append(accs[-1] + n_blocks[i])
        self.stage1 = self._make_stage(n_channels[:accs[0] + 1],
                                       n_blocks[0],
                                       block,
                                       stride=1)
        self.stage2 = self._make_stage(n_channels[accs[0]:accs[1] + 1],
                                       n_blocks[1],
                                       block,
                                       stride=2)
        self.stage3 = self._make_stage(n_channels[accs[1]:accs[2] + 1],
                                       n_blocks[2],
                                       block,
                                       stride=2)
        self.stage4 = self._make_stage(n_channels[accs[2]:accs[3] + 1],
                                       n_blocks[3],
                                       block,
                                       stride=2)

        self.bn_last = nn.BatchNorm2d(n_channels[-1])

        # compute conv feature size
        with torch.no_grad():
            dummy_data = torch.zeros(
                (1, config.dataset.n_channels, config.dataset.image_size,
                 config.dataset.image_size),
                dtype=torch.float32)
            self.feature_size = self._forward_conv(dummy_data).view(
                -1).shape[0]

        self.fc = nn.Linear(self.feature_size, config.dataset.n_classes)

        # initialize weights
        initializer = create_initializer(config.model.init_mode)
        self.apply(initializer)

    def _make_stage(self, n_channels, n_blocks, block, stride):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = f'block{index + 1}'
            if index == 0:
                stage.add_module(
                    block_name,
                    block(n_channels[index],
                          n_channels[index + 1],
                          stride=stride))
            else:
                stage.add_module(
                    block_name,
                    block(n_channels[index], n_channels[index + 1], stride=1))
        return stage

    def _forward_conv(self, x):
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.relu(self.bn_last(x),
                   inplace=True)  # apply BN and ReLU before average pooling
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
