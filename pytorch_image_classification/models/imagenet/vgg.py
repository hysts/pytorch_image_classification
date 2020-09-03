import torch
import torch.nn as nn
import torch.nn.functional as F

from ..initializer import create_initializer


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()

        model_config = config.model.vgg
        self.use_bn = model_config.use_bn
        n_channels = model_config.n_channels
        n_layers = model_config.n_layers

        self.stage1 = self._make_stage(config.dataset.n_channels,
                                       n_channels[0], n_layers[0])
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
            dummy_data = torch.zeros(
                (1, config.dataset.n_channels, config.dataset.image_size,
                 config.dataset.image_size),
                dtype=torch.float32)
            self.feature_size = self._forward_conv(dummy_data).view(
                -1).shape[0]

        self.fc1 = nn.Linear(self.feature_size, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, config.dataset.n_classes)

        # initialize weights
        initializer = create_initializer(config.model.init_mode)
        self.apply(initializer)

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
            stage.add_module(f'conv{index}', conv)
            if self.use_bn:
                stage.add_module(f'bn{index}', nn.BatchNorm2d(out_channels))
            stage.add_module(f'relu{index}', nn.ReLU(inplace=True))
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
        x = F.dropout(F.relu(self.fc1(x), inplace=True),
                      training=self.training)
        x = F.dropout(F.relu(self.fc2(x), inplace=True),
                      training=self.training)
        x = self.fc3(x)
        return x
