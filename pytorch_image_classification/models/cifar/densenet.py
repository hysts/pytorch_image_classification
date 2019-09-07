import torch
import torch.nn as nn
import torch.nn.functional as F

from ..initializer import create_initializer


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate):
        super().__init__()

        self.drop_rate = drop_rate

        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=False)

    def forward(self, x):
        y = self.conv(F.relu(self.bn(x), inplace=True))
        if self.drop_rate > 0:
            y = F.dropout(y,
                          p=self.drop_rate,
                          training=self.training,
                          inplace=False)
        return torch.cat([x, y], dim=1)


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate):
        super().__init__()

        self.drop_rate = drop_rate

        bottleneck_channels = out_channels * 4

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels,
                               bottleneck_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)

        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)

    def forward(self, x):
        y = self.conv1(F.relu(self.bn1(x), inplace=True))
        if self.drop_rate > 0:
            y = F.dropout(y,
                          p=self.drop_rate,
                          training=self.training,
                          inplace=False)
        y = self.conv2(F.relu(self.bn2(y), inplace=True))
        if self.drop_rate > 0:
            y = F.dropout(y,
                          p=self.drop_rate,
                          training=self.training,
                          inplace=False)
        return torch.cat([x, y], dim=1)


class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate):
        super().__init__()

        self.drop_rate = drop_rate

        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              bias=False)

    def forward(self, x):
        x = self.conv(F.relu(self.bn(x), inplace=True))
        if self.drop_rate > 0:
            x = F.dropout(x,
                          p=self.drop_rate,
                          training=self.training,
                          inplace=False)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()

        model_config = config.model.densenet
        depth = model_config.depth
        block_type = model_config.block_type
        self.growth_rate = model_config.growth_rate
        self.drop_rate = model_config.drop_rate
        self.compression_rate = model_config.compression_rate

        assert block_type in ['basic', 'bottleneck']
        if block_type == 'basic':
            block = BasicBlock
            n_blocks_per_stage = (depth - 4) // 3
            assert n_blocks_per_stage * 3 + 4 == depth
        else:
            block = BottleneckBlock
            n_blocks_per_stage = (depth - 4) // 6
            assert n_blocks_per_stage * 6 + 4 == depth

        in_channels = [2 * self.growth_rate]
        for index in range(3):
            denseblock_out_channels = int(in_channels[-1] +
                                          n_blocks_per_stage *
                                          self.growth_rate)
            if index < 2:
                transitionblock_out_channels = int(denseblock_out_channels *
                                                   self.compression_rate)
            else:
                transitionblock_out_channels = denseblock_out_channels
            in_channels.append(transitionblock_out_channels)

        self.conv = nn.Conv2d(config.dataset.n_channels,
                              in_channels[0],
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=False)
        self.stage1 = self._make_stage(in_channels[0], n_blocks_per_stage,
                                       block, True)
        self.stage2 = self._make_stage(in_channels[1], n_blocks_per_stage,
                                       block, True)
        self.stage3 = self._make_stage(in_channels[2], n_blocks_per_stage,
                                       block, False)
        self.bn = nn.BatchNorm2d(in_channels[3])

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

    def _make_stage(self, in_channels, n_blocks, block, add_transition_block):
        stage = nn.Sequential()
        for index in range(n_blocks):
            stage.add_module(
                f'block{index + 1}',
                block(in_channels + index * self.growth_rate, self.growth_rate,
                      self.drop_rate))
        if add_transition_block:
            in_channels = int(in_channels + n_blocks * self.growth_rate)
            out_channels = int(in_channels * self.compression_rate)
            stage.add_module(
                'transition',
                TransitionBlock(in_channels, out_channels, self.drop_rate))
        return stage

    def _forward_conv(self, x):
        x = self.conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.relu(self.bn(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
