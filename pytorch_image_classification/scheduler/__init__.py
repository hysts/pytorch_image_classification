import torch

from .combined_scheduler import CombinedScheduler
from .components import (
    ConstantScheduler,
    CosineScheduler,
    ExponentialScheduler,
    LinearScheduler,
)
from .multistep_scheduler import MultistepScheduler
from .sgdr import SGDRScheduler


def _create_warmup(config, warmup_steps):
    warmup_type = config.scheduler.warmup.type
    if warmup_type == 'none' or warmup_steps == 0:
        return None

    warmup_start_factor = config.scheduler.warmup.start_factor

    if warmup_type == 'linear':
        lr_end = 1
        lr_start = warmup_start_factor
        scheduler = LinearScheduler(warmup_steps, lr_start, lr_end)
    elif warmup_type == 'exponential':
        scheduler = ExponentialScheduler(warmup_steps, config.train.base_lr,
                                         config.scheduler.warmup.exponent,
                                         warmup_start_factor)
    else:
        raise ValueError()

    return scheduler


def _create_main_scheduler(config, main_steps):
    scheduler_type = config.scheduler.type

    if scheduler_type == 'constant':
        scheduler = ConstantScheduler(main_steps, 1)
    elif scheduler_type == 'multistep':
        lr_decay = config.scheduler.lr_decay
        scheduler = MultistepScheduler(main_steps, 1, lr_decay,
                                       config.scheduler.milestones)
    elif scheduler_type == 'linear':
        lr_start = 1
        lr_end = config.scheduler.lr_min_factor
        scheduler = LinearScheduler(main_steps, lr_start, lr_end)
    elif scheduler_type == 'cosine':
        scheduler = CosineScheduler(main_steps, 1,
                                    config.scheduler.lr_min_factor)
    elif scheduler_type == 'sgdr':
        scheduler = SGDRScheduler(main_steps, 1, config.scheduler.T0,
                                  config.scheduler.T_mul,
                                  config.scheduler.lr_min_factor)
    else:
        raise ValueError()

    return scheduler


def create_scheduler(config, optimizer, steps_per_epoch):
    warmup_epochs = config.scheduler.warmup.epochs
    main_epochs = config.scheduler.epochs - warmup_epochs

    warmup_scheduler = _create_warmup(config, warmup_epochs)
    main_scheduler = _create_main_scheduler(config, main_epochs)

    scheduler_func = CombinedScheduler([warmup_scheduler, main_scheduler])
    scheduler_func.multiply_steps(steps_per_epoch)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_func)

    return scheduler
