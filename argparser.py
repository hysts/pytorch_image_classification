# coding: utf-8

import json
from collections import OrderedDict
import torch


def _args2config(args, keys, json_keys):
    if json_keys is None:
        json_keys = []

    args = vars(args)

    config = OrderedDict()
    for key in keys:
        value = args[key]
        if value is None:
            continue

        if key in json_keys and isinstance(value, str):
            value = json.loads(value)

        config[key] = value

    return config


def _get_model_config(args):
    keys = [
        'arch',
        'input_shape',
        'n_classes',
        # vgg
        'n_channels',
        'n_layers',
        'use_bn',
        #
        'base_channels',
        'block_type',
        'depth',
        # resnet_preact, se_resnet_preact
        'remove_first_relu',
        'add_last_bn',
        'preact_stage',
        # wrn
        'widening_factor',
        # densenet
        'growth_rate',
        'compression_rate',
        # wrn, densenet
        'drop_rate',
        # pyramidnet
        'pyramid_alpha',
        # resnext
        'cardinality',
        # shake_shake
        'shake_forward',
        'shake_backward',
        'shake_image',
        # se_resnet_preact
        'se_reduction',
    ]
    json_keys = ['preact_stage']
    config = _args2config(args, keys, json_keys)
    return config


def _check_optim_config(config):
    optimizer = config['optimizer']
    for key in ['base_lr', 'weight_decay']:
        message = 'Key `{}` must be specified.'.format(key)
        assert key in config.keys(), message
    if optimizer == 'sgd':
        for key in ['momentum', 'nesterov']:
            message = 'When using SGD, key `{}` must be specified.'.format(key)
            assert key in config.keys(), message
    elif optimizer == 'adam':
        for key in ['betas']:
            message = 'When using Adam, key `{}` must be specified.'.format(
                key)
            assert key in config.keys(), message

    scheduler = config['scheduler']
    if scheduler == 'multistep':
        for key in ['milestones', 'lr_decay']:
            message = 'Key `{}` must be specified.'.format(key)
            assert key in config.keys(), message
    elif scheduler == 'cosine':
        for key in ['lr_min']:
            message = 'Key `{}` must be specified.'.format(key)
            assert key in config.keys(), message


def _get_optim_config(args):
    keys = [
        'epochs',
        'batch_size',
        'optimizer',
        'base_lr',
        'weight_decay',
        'momentum',
        'nesterov',
        'scheduler',
        'milestones',
        'lr_decay',
        'lr_min',
        'betas',
    ]
    json_keys = ['milestones', 'betas']
    config = _args2config(args, keys, json_keys)

    _check_optim_config(config)

    return config


def _get_data_config(args):
    keys = [
        'dataset',
        'n_classes',
        'num_workers',
        'batch_size',
        'use_cutout',
        'cutout_size',
        'cutout_prob',
        'cutout_inside',
        'use_random_erasing',
        'random_erasing_prob',
        'random_erasing_area_ratio_range',
        'random_erasing_min_aspect_ratio',
        'random_erasing_max_attempt',
        'use_mixup',
        'mixup_alpha',
    ]
    json_keys = ['random_erasing_area_ratio_range']
    config = _args2config(args, keys, json_keys)
    config['use_gpu'] = True if args.gpu != '-1' else False
    return config


def _get_run_config(args):
    keys = [
        'outdir',
        'seed',
        'test_first',
        'gpu',
        'tensorboard',
        'tensorboard_train_images',
        'tensorboard_test_images',
        'tensorboard_model_params',
    ]
    config = _args2config(args, keys, None)

    config['use_gpu'] = True if args.gpu != '-1' else False
    return config


def _get_env_info(args):
    info = OrderedDict({
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version(),
    })

    def _get_device_info(device_id):
        name = torch.cuda.get_device_name(device_id)
        capability = torch.cuda.get_device_capability(device_id)
        capability = '{}.{}'.format(*capability)
        return name, capability

    for gpu_id in args.gpus:
        if gpu_id == -1:
            continue

        name, capability = _get_device_info(gpu_id)
        info['gpu{}'.format(gpu_id)] = OrderedDict({
            'name': name,
            'capability': capability,
        })

    return info


def _cleanup_args(args):
    # architecture
    if args.arch == 'vgg':
        args.base_channels = None
        args.depth = None
    if args.arch != 'vgg':
        args.n_channels = None
        args.n_layers = None
        args.use_bn = None
    if args.arch not in ['resnet', 'resnet_preact', 'densenet', 'pyramidnet',
                         'se_resnet_preact']:
        args.block_type = None
    if args.arch not in ['resnet_preact', 'se_resnet_preact']:
        args.remove_first_relu = None
        args.add_last_bn = None
        args.preact_stage = None
    if args.arch != 'wrn':
        args.widening_factor = None
    if args.arch != 'densenet':
        args.growth_rate = None
        args.compression_rate = None
    if args.arch not in ['wrn', 'densenet']:
        args.drop_rate = None
    if args.arch != 'pyramidnet':
        args.pyramid_alpha = None
    if args.arch != 'resnext':
        args.cardinality = None
    if args.arch != 'shake_shake':
        args.shake_forward = None
        args.shake_backward = None
        args.shake_image = None
    if args.arch != 'se_resnet_preact':
        args.se_reduction = None

    # optimizer
    if args.optimizer != 'sgd':
        args.momentum = None
        args.nesterov = None
        args.scheduler = 'none'
    if args.optimizer != 'adam':
        args.betas = None

    # scheduler
    if args.scheduler != 'multistep':
        args.milestones = None
        args.lr_decay = None
    if args.scheduler != 'cosine':
        args.lr_min = None

    # cutout
    if not args.use_cutout:
        args.cutout_size = None
        args.cutout_prob = None
        args.cutout_inside = None

    # random erasing
    if not args.use_random_erasing:
        args.random_erasing_prob = None
        args.random_erasing_area_ratio_range = None
        args.random_erasing_min_aspect_ratio = None
        args.random_erasing_max_attempt = None

    # mixup
    if not args.use_mixup:
        args.mixup_alpha = None

    # TensorBoard
    if not args.tensorboard:
        args.tensorboard_train_images = False
        args.tensorboard_test_images = False
        args.tensorboard_model_params = False

    # data
    if args.dataset == 'CIFAR10':
        args.input_shape = (1, 3, 32, 32)
        args.n_classes = 10
    elif args.dataset == 'CIFAR100':
        args.input_shape = (1, 3, 32, 32)
        args.n_classes = 100
    elif 'MNIST' in args.dataset:
        args.input_shape = (1, 1, 28, 28)
        args.n_classes = 10

    return args


def _set_default_values(args):
    if args.config is not None:
        with open(args.config, 'r') as fin:
            config = json.load(fin)

        d_args = vars(args)
        for config_key, default_config in config.items():
            if config_key == 'env_info':
                continue

            for key, default_value in default_config.items():
                if key not in d_args.keys() or d_args[key] is None:
                    setattr(args, key, default_value)

    return args


def get_config(args):
    if args.arch is None and args.config is None:
        raise RuntimeError(
            'One of args.arch and args.config must be specified')
    if args.config is None:
        args.config = 'configs/{}.json'.format(args.arch)

    args.gpus = list(map(lambda x: int(x), args.gpu.split(',')))

    args = _set_default_values(args)
    args = _cleanup_args(args)
    config = OrderedDict({
        'model_config': _get_model_config(args),
        'optim_config': _get_optim_config(args),
        'data_config': _get_data_config(args),
        'run_config': _get_run_config(args),
        'env_info': _get_env_info(args),
    })
    return config
