import torch
import torch.nn as nn
import torch.nn.functional as F

from ..initializer import create_initializer
from ..functions.shake_shake_function import get_alpha_beta, shake_function


class ResidualPath(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(x, inplace=False)
        x = F.relu(self.bn1(self.conv1(x)), inplace=False)
        x = self.bn2(self.conv2(x))
        return x


class SkipConnection(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels // 2,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels,
                               out_channels // 2,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.stride = stride

    def forward(self, x):
        x = F.relu(x, inplace=False)
        y1 = F.avg_pool2d(x, kernel_size=1, stride=self.stride, padding=0)
        y1 = self.conv1(y1)

        y2 = F.pad(x[:, :, 1:, 1:], (0, 1, 0, 1))
        y2 = F.avg_pool2d(y2, kernel_size=1, stride=self.stride, padding=0)
        y2 = self.conv2(y2)

        z = torch.cat([y1, y2], dim=1)
        z = self.bn(z)

        return z


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, shake_config):
        super().__init__()

        self.shake_config = shake_config

        self.residual_path1 = ResidualPath(in_channels, out_channels, stride)
        self.residual_path2 = ResidualPath(in_channels, out_channels, stride)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                'skip', SkipConnection(in_channels, out_channels, stride))

    def forward(self, x):
        x1 = self.residual_path1(x)
        x2 = self.residual_path2(x)

        if self.training:
            shake_config = self.shake_config
        else:
            shake_config = (False, False, False)

        alpha, beta = get_alpha_beta(x.size(0), shake_config, x.device)
        y = shake_function(x1, x2, alpha, beta)

        return self.shortcut(x) + y


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()

        model_config = config.model.shake_shake
        depth = model_config.depth
        initial_channels = model_config.initial_channels
        self.shake_config = [
            model_config.shake_forward,
            model_config.shake_backward,
            model_config.shake_image,
        ]

        block = BasicBlock
        n_blocks_per_stage = (depth - 2) // 6
        assert n_blocks_per_stage * 6 + 2 == depth

        n_channels = [
            initial_channels,
            initial_channels * 2,
            initial_channels * 4,
        ]

        self.conv = nn.Conv2d(config.dataset.n_channels,
                              16,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=False)
        self.bn = nn.BatchNorm2d(16)

        self.stage1 = self._make_stage(16,
                                       n_channels[0],
                                       n_blocks_per_stage,
                                       block,
                                       stride=1)
        self.stage2 = self._make_stage(n_channels[0],
                                       n_channels[1],
                                       n_blocks_per_stage,
                                       block,
                                       stride=2)
        self.stage3 = self._make_stage(n_channels[1],
                                       n_channels[2],
                                       n_blocks_per_stage,
                                       block,
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

    def _make_stage(self, in_channels, out_channels, n_blocks, block, stride):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = f'block{index + 1}'
            if index == 0:
                stage.add_module(
                    block_name,
                    block(in_channels,
                          out_channels,
                          stride=stride,
                          shake_config=self.shake_config))
            else:
                stage.add_module(
                    block_name,
                    block(out_channels,
                          out_channels,
                          stride=1,
                          shake_config=self.shake_config))
        return stage

    def _forward_conv(self, x):
        x = self.bn(self.conv(x))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
