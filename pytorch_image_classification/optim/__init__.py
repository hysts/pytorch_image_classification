import torch

from .adabound import AdaBound, AdaBoundW
from .lars import LARSOptimizer


def get_param_list(config, model):
    if config.train.no_weight_decay_on_bn:
        param_list = []
        for name, params in model.named_parameters():
            if 'conv.weight' in name:
                param_list.append({
                    'params': params,
                    'weight_decay': config.train.weight_decay,
                })
            else:
                param_list.append({
                    'params': params,
                    'weight_decay': 0,
                })
    else:
        param_list = [{
            'params': list(model.parameters()),
            'weight_decay': config.train.weight_decay,
        }]
    return param_list


def create_optimizer(config, model):
    params = get_param_list(config, model)

    if config.train.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params,
                                    lr=config.train.base_lr,
                                    momentum=config.train.momentum,
                                    nesterov=config.train.nesterov)
    elif config.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(params,
                                     lr=config.train.base_lr,
                                     betas=config.optim.adam.betas)
    elif config.train.optimizer == 'amsgrad':
        optimizer = torch.optim.Adam(params,
                                     lr=config.train.base_lr,
                                     betas=config.optim.adam.betas,
                                     amsgrad=True)
    elif config.train.optimizer == 'adabound':
        optimizer = AdaBound(params,
                             lr=config.train.base_lr,
                             betas=config.optim.adabound.betas,
                             final_lr=config.optim.adabound.final_lr,
                             gamma=config.optim.adabound.gamma)
    elif config.train.optimizer == 'adaboundw':
        optimizer = AdaBoundW(params,
                              lr=config.train.base_lr,
                              betas=config.optim.adabound.betas,
                              final_lr=config.optim.adabound.final_lr,
                              gamma=config.optim.adabound.gamma)
    elif config.train.optimizer == 'lars':
        optimizer = LARSOptimizer(params,
                                  lr=config.train.base_lr,
                                  momentum=config.train.momentum,
                                  eps=config.optim.lars.eps,
                                  thresh=config.optim.lars.threshold)
    else:
        raise ValueError()
    return optimizer
