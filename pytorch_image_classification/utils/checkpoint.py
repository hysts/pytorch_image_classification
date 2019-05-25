import copy
import logging
import pathlib

import torch
import torch.nn as nn

from pytorch_image_classification import get_default_config
from pytorch_image_classification.config.config_node import ConfigNode


class CheckPointer:
    def __init__(
            self,
            model,
            optimizer=None,
            scheduler=None,
            checkpoint_dir=None,
            logger=None,
            distributed_rank=0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = pathlib.Path(
            checkpoint_dir) if checkpoint_dir is not None else None
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger
        self.distributed_rank = distributed_rank

    def save(self, name, **kwargs):
        if self.checkpoint_dir is None or self.distributed_rank != 0:
            return

        checkpoint = copy.deepcopy(kwargs)
        if isinstance(self.model,
                      (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            checkpoint['model'] = self.model.module.state_dict()
        else:
            checkpoint['model'] = self.model.state_dict()
        if self.optimizer is not None:
            checkpoint['optimizer'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            checkpoint['scheduler'] = self.scheduler.state_dict()

        outpath = self.checkpoint_dir / f'{name}.pth'
        self.logger.info(f'Saving checkpoint to {outpath.as_posix()}')
        torch.save(checkpoint, outpath)
        self.tag_last_checkpoint(outpath)

    def load(self, path=None):
        if path is None and self.has_checkpoint():
            path = self.get_checkpoint_filepath()
        if isinstance(path, str):
            path = pathlib.Path(path)
        if path is None or not path.exists():
            raise RuntimeError('Checkpoint not found.')

        self.logger.info(f'Loading checkpoint from {path.as_posix()}')
        checkpoint = self._load_checkpoint(path)

        self.load_checkpoint(checkpoint)
        if 'optimizer' in checkpoint.keys() and self.optimizer is not None:
            self.logger.info(f'Loading optimizer from {path.as_posix()}')
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint.keys() and self.scheduler is not None:
            self.logger.info(f'Loading scheduler from {path.as_posix()}')
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        default_config = get_default_config()
        if 'config' in checkpoint.keys():
            config = ConfigNode(checkpoint['config'])
        else:
            config = default_config
        return config, checkpoint.get('epoch', 0)

    def has_checkpoint(self):
        if self.checkpoint_dir is None:
            return False
        checkpoint_file = self.checkpoint_dir / 'last_checkpoint'
        return checkpoint_file.exists()

    def get_checkpoint_filepath(self):
        checkpoint_file = self.checkpoint_dir / 'last_checkpoint'
        try:
            with open(checkpoint_file, 'r') as fin:
                last_saved = fin.read()
                last_saved = last_saved.strip()
            last_saved = self.checkpoint_dir / last_saved
        except IOError:
            last_saved = None
        return last_saved

    def tag_last_checkpoint(self, last_filepath):
        outfile = self.checkpoint_dir / 'last_checkpoint'
        with open(outfile, 'w') as fout:
            fout.write(last_filepath.name)

    @staticmethod
    def _load_checkpoint(path):
        return torch.load(path, map_location='cpu')

    def load_checkpoint(self, checkpoint):
        if isinstance(self.model,
                      (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            self.model.module.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'])
