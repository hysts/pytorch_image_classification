from typing import Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import yacs.config

from torch.utils.data import DataLoader

from pytorch_image_classification import create_collator
from pytorch_image_classification.datasets import create_dataset


def worker_init_fn(worker_id: int) -> None:
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def create_dataloader(
        config: yacs.config.CfgNode,
        is_train: bool) -> Union[Tuple[DataLoader, DataLoader], DataLoader]:
    if is_train:
        train_dataset, val_dataset = create_dataset(config, is_train)

        if dist.is_available() and dist.is_initialized():
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset)
        else:
            train_sampler = torch.utils.data.sampler.RandomSampler(
                train_dataset, replacement=False)
            val_sampler = torch.utils.data.sampler.SequentialSampler(
                val_dataset)

        train_collator = create_collator(config)

        train_batch_sampler = torch.utils.data.sampler.BatchSampler(
            train_sampler,
            batch_size=config.train.batch_size,
            drop_last=config.train.dataloader.drop_last)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=config.train.dataloader.num_workers,
            collate_fn=train_collator,
            pin_memory=config.train.dataloader.pin_memory,
            worker_init_fn=worker_init_fn)

        val_batch_sampler = torch.utils.data.sampler.BatchSampler(
            val_sampler,
            batch_size=config.validation.batch_size,
            drop_last=config.validation.dataloader.drop_last)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_sampler=val_batch_sampler,
            num_workers=config.validation.dataloader.num_workers,
            pin_memory=config.validation.dataloader.pin_memory,
            worker_init_fn=worker_init_fn)

        return train_loader, val_loader
    else:
        dataset = create_dataset(config, is_train)
        if dist.is_available() and dist.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = None
        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.test.batch_size,
            num_workers=config.test.dataloader.num_workers,
            sampler=sampler,
            shuffle=False,
            drop_last=False,
            pin_memory=config.test.dataloader.pin_memory)
        return test_loader
