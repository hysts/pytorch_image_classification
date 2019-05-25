import importlib

import torch
import torch.nn as nn
import torch.distributed as dist


def create_model(config):
    module = importlib.import_module(
        f'pytorch_image_classification.models.{config.model.name}')
    model = getattr(module, 'Network')(config)
    device = torch.device(config.device)
    model.to(device)
    return model


def apply_data_parallel_wrapper(config, model):
    local_rank = config.train.dist.local_rank
    if dist.is_available() and dist.is_initialized():
        if config.train.dist.use_sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model,
                                                    device_ids=[local_rank],
                                                    output_device=local_rank)
    else:
        if config.device == 'cuda':
            model = nn.DataParallel(model)
    return model
