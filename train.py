#!/usr/bin/env python

import argparse
import pathlib
import time

import apex
import numpy as np
import torch
import torch.distributed as dist
import torchvision
from tensorboardX import SummaryWriter

from pytorch_image_classification import (
    apply_data_parallel_wrapper,
    create_dataloader,
    create_loss,
    create_model,
    create_optimizer,
    create_scheduler,
    get_default_config,
    update_config,
)
from pytorch_image_classification.utils import (
    AverageMeter,
    CheckPointer,
    compute_accuracy,
    create_logger,
    find_config_diff,
    get_env_info,
    get_rank,
    save_config,
    set_seed,
    setup_cudnn,
)


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    if args.config is not None:
        config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    if not torch.cuda.is_available():
        config.device = 'cpu'
        config.train.dataloader.pin_memory = False
    if args.resume != '':
        config_path = pathlib.Path(args.resume) / 'config.yaml'
        config.merge_from_file(config_path.as_posix())
        config.merge_from_list(['train.resume', True])
    config.merge_from_list(['train.dist.local_rank', args.local_rank])
    config = update_config(config)
    config.freeze()
    return config


def subdivide_batch(config, data, targets):
    subdivision = config.train.subdivision

    if subdivision == 1:
        return [data], [targets]

    data_chunks = data.chunk(subdivision)
    if config.augmentation.use_mixup or config.augmentation.use_cutmix:
        targets1, targets2, lam = targets
        target_chunks = [(chunk1, chunk2, lam) for chunk1, chunk2 in zip(
            targets1.chunk(subdivision), targets2.chunk(subdivision))]
    elif config.augmentation.use_ricap:
        target_list, weights = targets
        target_list_chunks = list(
            zip(*[target.chunk(subdivision) for target in target_list]))
        target_chunks = [(chunk, weights) for chunk in target_list_chunks]
    else:
        target_chunks = targets.chunk(subdivision)
    return data_chunks, target_chunks


def send_targets_to_device(config, targets, device):
    if config.augmentation.use_mixup or config.augmentation.use_cutmix:
        t1, t2, lam = targets
        targets = (t1.to(device), t2.to(device), lam)
    elif config.augmentation.use_ricap:
        labels, weights = targets
        labels = [label.to(device) for label in labels]
        targets = (labels, weights)
    else:
        targets = targets.to(device)
    return targets


def train(epoch, config, model, optimizer, scheduler, loss_func, train_loader,
          logger, writer):
    logger.info(f'Train {epoch}')

    device = torch.device(config.device)

    model.train()

    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    start = time.time()
    for step, (data, targets) in enumerate(train_loader):
        step += 1

        if get_rank() == 0 and step == 1:
            if config.tensorboard.train_images:
                image = torchvision.utils.make_grid(data,
                                                    normalize=True,
                                                    scale_each=True)
                writer.add_image('Train/Image', image, epoch)

        data = data.to(device)
        targets = send_targets_to_device(config, targets, device)

        data_chunks, target_chunks = subdivide_batch(config, data, targets)
        optimizer.zero_grad()
        outputs = []
        losses = []
        for data_chunk, target_chunk in zip(data_chunks, target_chunks):
            if config.augmentation.use_dual_cutout:
                w = data_chunk.size(3) // 2
                data1 = data_chunk[:, :, :, :w]
                data2 = data_chunk[:, :, :, w:]
                outputs1 = model(data1)
                outputs2 = model(data2)
                output_chunk = torch.cat(
                    (outputs1.unsqueeze(1), outputs2.unsqueeze(1)), dim=1)
            else:
                output_chunk = model(data_chunk)
            outputs.append(output_chunk)

            loss = loss_func(output_chunk, target_chunk)
            losses.append(loss)
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        outputs = torch.cat(outputs)

        if config.train.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(apex.amp.master_params(optimizer),
                                           config.train.gradient_clip)
        if config.train.subdivision > 1:
            for param in model.parameters():
                param.grad.data.div_(config.train.subdivision)
        optimizer.step()

        accuracy = compute_accuracy(config, outputs, targets)

        loss = sum(losses)
        if config.train.distributed:
            loss_all_reduce = dist.all_reduce(loss,
                                              op=dist.ReduceOp.SUM,
                                              async_op=True)
            acc_all_reduce = dist.all_reduce(accuracy,
                                             op=dist.ReduceOp.SUM,
                                             async_op=True)
            loss_all_reduce.wait()
            acc_all_reduce.wait()
            loss.div_(dist.get_world_size())
            accuracy.div_(dist.get_world_size())
        loss = loss.item()
        accuracy = accuracy.item()

        num = data.size(0)
        loss_meter.update(loss, num)
        accuracy_meter.update(accuracy, num)

        torch.cuda.synchronize()

        if get_rank() == 0:
            if step % config.train.log_period == 0 or step == len(
                    train_loader):
                logger.info(
                    f'Epoch {epoch} '
                    f'Step {step}/{len(train_loader)} '
                    f'lr {scheduler.get_lr()[0]:.6f} '
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                    'accuracy '
                    f'{accuracy_meter.val:.4f} ({accuracy_meter.avg:.4f})')

        scheduler.step()

    if get_rank() == 0:
        elapsed = time.time() - start
        logger.info(f'Elapsed {elapsed:.2f}')

        writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
        writer.add_scalar('Train/Accuracy', accuracy_meter.avg, epoch)
        writer.add_scalar('Train/Time', elapsed, epoch)
        writer.add_scalar('Train/LearningRate', scheduler.get_lr()[0], epoch)


def validate(epoch, config, model, loss_func, val_loader, logger, writer):
    logger.info(f'Val {epoch}')

    device = torch.device(config.device)

    model.eval()

    loss_meter = AverageMeter()
    correct_meter = AverageMeter()
    start = time.time()
    with torch.no_grad():
        for step, (data, targets) in enumerate(val_loader):
            if get_rank() == 0:
                if config.tensorboard.val_images:
                    if epoch == 0 and step == 0:
                        image = torchvision.utils.make_grid(data,
                                                            normalize=True,
                                                            scale_each=True)
                        writer.add_image('Val/Image', image, epoch)

            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            loss = loss_func(outputs, targets)

            _, preds = torch.max(outputs, dim=1)

            correct = preds.eq(targets).sum()
            num = data.size(0)

            if config.train.distributed:
                loss_all_reduce = dist.all_reduce(loss,
                                                  op=dist.ReduceOp.SUM,
                                                  async_op=True)
                correct_all_reduce = dist.all_reduce(correct,
                                                     op=dist.ReduceOp.SUM,
                                                     async_op=True)
                loss_all_reduce.wait()
                correct_all_reduce.wait()
                loss.div_(dist.get_world_size())
            loss = loss.item()
            correct = correct.item()

            loss_meter.update(loss, num)
            correct_meter.update(correct, 1)

            torch.cuda.synchronize()

        accuracy = correct_meter.sum / len(val_loader.dataset)

        logger.info(
            f'Epoch {epoch} loss {loss_meter.avg:.4f} accuracy {accuracy:.4f}')

        elapsed = time.time() - start
        logger.info(f'Elapsed {elapsed:.2f}')

    if get_rank() == 0:
        if epoch > 0:
            writer.add_scalar('Val/Loss', loss_meter.avg, epoch)
        writer.add_scalar('Val/Accuracy', accuracy, epoch)
        writer.add_scalar('Val/Time', elapsed, epoch)
        if config.tensorboard.model_params:
            for name, param in model.named_parameters():
                writer.add_histogram(name, param, epoch)


def main():
    config = load_config()

    set_seed(config)
    setup_cudnn(config)

    epoch_seeds = np.random.randint(np.iinfo(np.int32).max // 2,
                                    size=config.scheduler.epochs)

    if config.train.distributed:
        dist.init_process_group(backend=config.train.dist.backend,
                                init_method=config.train.dist.init_method,
                                rank=config.train.dist.node_rank,
                                world_size=config.train.dist.world_size)
        torch.cuda.set_device(config.train.dist.local_rank)

    output_dir = pathlib.Path(config.train.output_dir)
    if get_rank() == 0:
        if not config.train.resume and output_dir.exists():
            raise RuntimeError(
                f'Output directory `{output_dir.as_posix()}` already exists')
        output_dir.mkdir(exist_ok=True, parents=True)
        if not config.train.resume:
            save_config(config, output_dir / 'config.yaml')
            save_config(get_env_info(config), output_dir / 'env.yaml')
            diff = find_config_diff(config)
            if diff is not None:
                save_config(diff, output_dir / 'config_min.yaml')

    logger = create_logger(name=__name__,
                           distributed_rank=get_rank(),
                           output_dir=output_dir,
                           filename='log.txt')
    logger.info(config)
    logger.info(get_env_info(config))

    train_loader, val_loader = create_dataloader(config, is_train=True)

    model = create_model(config)
    n_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    logger.info(f'n_params: {n_params}')

    optimizer = create_optimizer(config, model)
    model, optimizer = apex.amp.initialize(model,
                                           optimizer,
                                           opt_level=config.train.precision)
    model = apply_data_parallel_wrapper(config, model)

    scheduler = create_scheduler(config,
                                 optimizer,
                                 steps_per_epoch=len(train_loader))
    checkpointer = CheckPointer(model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                checkpoint_dir=output_dir,
                                logger=logger,
                                distributed_rank=get_rank())

    start_epoch = config.train.start_epoch
    scheduler.last_epoch = start_epoch
    if config.train.resume:
        checkpoint_config, start_epoch = checkpointer.load()
        config.defrost()
        config.merge_from_other_cfg(checkpoint_config)
        config.train.start_epoch = start_epoch
        config.freeze()
    elif config.train.checkpoint != '':
        _, start_epoch = checkpointer.load(config.train.checkpoint)
        config.defrost()
        config.train.start_epoch = start_epoch
        config.freeze()
        for index in range(len(scheduler.base_lrs)):
            scheduler.base_lrs[index] = config.train.base_lr
        save_config(config, output_dir)

    if get_rank() == 0 and config.train.use_tensorboard:
        if start_epoch > 0:
            writer = SummaryWriter(output_dir.as_posix(),
                                   purge_step=start_epoch + 1)
        else:
            writer = SummaryWriter(output_dir.as_posix())
    else:
        writer = None

    train_loss, val_loss = create_loss(config)

    if (config.train.val_period > 0 and start_epoch == 0
            and config.train.val_first):
        validate(0, config, model, val_loss, val_loader, logger, writer)

    for epoch, seed in enumerate(epoch_seeds[start_epoch:], start_epoch):
        epoch += 1

        np.random.seed(seed)
        train(epoch, config, model, optimizer, scheduler, train_loss,
              train_loader, logger, writer)

        if config.train.val_period > 0 and (epoch %
                                            config.train.val_period == 0):
            validate(epoch, config, model, val_loss, val_loader, logger,
                     writer)

        if (epoch % config.train.checkpoint_period == 0) or (
                epoch == config.scheduler.epochs):
            checkpoint_config = {'epoch': epoch, 'config': config.as_dict()}
            checkpointer.save(f'checkpoint_{epoch:05d}', **checkpoint_config)

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    main()
