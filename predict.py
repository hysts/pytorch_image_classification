#!/usr/bin/env python

import argparse
import logging
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from dataloader import get_loader
import utils
from utils import AverageMeter
from argparser import get_config

logging.basicConfig(
    format='[%(asctime)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)


def predict(model, criterion, test_loader, device):
    model.eval()

    loss_meter = AverageMeter()
    correct_meter = AverageMeter()
    start = time.time()

    pred_raw_all = []
    pred_prob_all = []
    pred_label_all = []
    with torch.no_grad():
        for data, targets in tqdm.tqdm(test_loader):
            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets)

            pred_raw_all.append(outputs.cpu().numpy())
            pred_prob_all.append(F.softmax(outputs, dim=1).cpu().numpy())

            _, preds = torch.max(outputs, dim=1)
            pred_label_all.append(preds.cpu().numpy())

            loss_ = loss.item()
            correct_ = preds.eq(targets).sum().item()
            num = data.size(0)

            loss_meter.update(loss_, num)
            correct_meter.update(correct_, 1)

        accuracy = correct_meter.sum / len(test_loader.dataset)

        elapsed = time.time() - start
        logger.info('Elapsed {:.2f}'.format(elapsed))
        logger.info('Loss {:.4f} Accuracy {:.4f}'.format(
            loss_meter.avg, accuracy))

    preds = np.concatenate(pred_raw_all)
    probs = np.concatenate(pred_prob_all)
    labels = np.concatenate(pred_label_all)
    return preds, probs, labels, loss_meter.avg, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--outdir', type=str)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt)
    config = ckpt['config']
    state_dict = ckpt['state_dict']
    epoch = ckpt['epoch']

    if args.outdir is None:
        outdir = pathlib.Path(args.ckpt).parent
    else:
        outdir = pathlib.Path(args.outdir)
        outdir.mkdir(exist_ok=True, parents=True)

    use_gpu = args.device != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    data_config = config['data_config']
    data_config['batch_size'] = args.batch_size
    data_config['num_workers'] = args.num_workers
    data_config['use_gpu'] = use_gpu
    _, test_loader = get_loader(data_config)

    model = utils.load_model(config['model_config'])
    try:
        model.load_state_dict(state_dict)
    except Exception:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)
        model = model.module
    model.to(device)

    _, test_criterion = utils.get_criterion(config['data_config'])

    preds, probs, labels, loss, acc = predict(model, test_criterion,
                                              test_loader, device)

    outpath = outdir / 'preds_{:04}.npz'.format(epoch)
    np.savez(
        outpath, preds=preds, probs=probs, labels=labels, loss=loss, acc=acc)


if __name__ == '__main__':
    main()
