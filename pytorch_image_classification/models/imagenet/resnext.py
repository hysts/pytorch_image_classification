import torch
import torch.nn as nn
import torch.nn.functional as F

from ..initializer import create_initializer


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, stage_index,
                 base_channels, cardinality):
        super().__init__()

        bottleneck_channels = cardinality * base_channels * 2**stage_index

        self.conv1 = nn.Conv2d(in_channels,
                               bottleneck_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,  # downsample with 3x3 conv
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        self.conv3 = nn.Conv2d(bottleneck_channels,
                               out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()  # identity
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  # downsample
                    padding=0,
                    bias=False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))  # BN

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = F.relu(self.bn2(self.conv2(y)), inplace=True)
        y = self.bn3(self.conv3(y))  # not apply ReLU
        y += self.shortcut(x)
        y = F.relu(y, inplace=True)  # apply ReLU after addition
        return y


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()

        model_config = config.model.resnext
        initial_channels = model_config.initial_channels
        n_blocks = model_config.n_blocks
        self.base_channels = model_config.base_channels
        self.cardinality = model_config.cardinality

        block = BottleneckBlock

        n_channels = [
            initial_channels,
            initial_channels * block.expansion,
            initial_channels * 2 * block.expansion,
            initial_channels * 4 * block.expansion,
            initial_channels * 8 * block.expansion,
        ]

        self.conv = nn.Conv2d(config.dataset.n_channels,
                              n_channels[0],
                              kernel_size=7,
                              stride=2,
                              padding=3,
                              bias=False)
        self.bn = nn.BatchNorm2d(n_channels[0])

        self.stage1 = self._make_stage(n_channels[0],
                                       n_channels[1],
                                       n_blocks[0],
                                       0,
                                       stride=1)
        self.stage2 = self._make_stage(n_channels[1],
                                       n_channels[2],
                                       n_blocks[1],
                                       1,
                                       stride=2)
        self.stage3 = self._make_stage(n_channels[2],
                                       n_channels[3],
                                       n_blocks[2],
                                       2,
                                       stride=2)
        self.stage4 = self._make_stage(n_channels[3],
                                       n_channels[4],
                                       n_blocks[3],
                                       3,
                                       stride=2)

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

    def _make_stage(self, in_channels, out_channels, n_blocks, stage_index,
                    stride):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = f'block{index + 1}'
            if index == 0:
                stage.add_module(
                    block_name,
                    BottleneckBlock(
                        in_channels,
                        out_channels,
                        stride,  # downsample
                        stage_index,
                        self.base_channels,
                        self.cardinality))
            else:
                stage.add_module(
                    block_name,
                    BottleneckBlock(
                        out_channels,
                        out_channels,
                        1,  # no downsampling
                        stage_index,
                        self.base_channels,
                        self.cardinality))
        return stage

    def _forward_conv(self, x):
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
