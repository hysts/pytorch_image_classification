# coding: utf-8

import importlib
import os
import shutil
import numpy as np
import torch
import torch.nn.functional as F


def str2bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise RuntimeError('Boolean value expected')


def load_model(config):
    module = importlib.import_module('models.{}'.format(config['arch']))
    Network = getattr(module, 'Network')
    return Network(config)


def save_checkpoint(state, outdir):
    model_path = os.path.join(outdir, 'model_state.pth')
    best_model_path = os.path.join(outdir, 'model_best_state.pth')
    torch.save(state, model_path)
    if state['best_epoch'] == state['epoch']:
        shutil.copy(model_path, best_model_path)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
        1 + np.cos(step / total_steps * np.pi))


def _get_optimizer(model_parameters, optim_config):
    if optim_config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            model_parameters,
            lr=optim_config['base_lr'],
            momentum=optim_config['momentum'],
            weight_decay=optim_config['weight_decay'],
            nesterov=optim_config['nesterov'])
    elif optim_config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            model_parameters,
            lr=optim_config['base_lr'],
            betas=optim_config['betas'],
            weight_decay=optim_config['weight_decay'])
    return optimizer


def _get_scheduler(optimizer, optim_config):
    if optim_config['optimizer'] == 'sgd':
        if optim_config['scheduler'] == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=optim_config['milestones'],
                gamma=optim_config['lr_decay'])
        elif optim_config['scheduler'] == 'cosine':
            total_steps = optim_config['epochs'] * \
                optim_config['steps_per_epoch']

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: cosine_annealing(
                    step,
                    total_steps,
                    1,  # since lr_lambda computes multiplicative factor
                    optim_config['lr_min'] / optim_config['base_lr']))
    else:
        scheduler = None
    return scheduler


def create_optimizer(model_parameters, optim_config):
    optimizer = _get_optimizer(model_parameters, optim_config)
    scheduler = _get_scheduler(optimizer, optim_config)
    return optimizer, scheduler


def onehot(label, n_classes):
    return torch.zeros(label.size(0), n_classes).scatter_(
        1, label.view(-1, 1), 1)


def mixup(data, targets, alpha, n_classes):
    indices = torch.randperm(data.size(0))
    data2 = data[indices]
    targets2 = targets[indices]

    targets = onehot(targets, n_classes)
    targets2 = onehot(targets2, n_classes)

    lam = torch.FloatTensor([np.random.beta(alpha, alpha)])
    data = data * lam + data2 * (1 - lam)
    targets = targets * lam + targets2 * (1 - lam)

    return data, targets


def cross_entropy_loss(input, target, size_average=True):
    input = F.log_softmax(input, dim=1)
    loss = -torch.sum(input * target)
    if size_average:
        return loss / input.size(0)
    else:
        return loss


class CrossEntropyLoss(object):
    def __init__(self, size_average=True):
        self.size_average = size_average

    def __call__(self, input, target):
        return cross_entropy_loss(input, target, self.size_average)
