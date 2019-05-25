import torch


def compute_accuracy(config, outputs, targets, topk=(1, )):
    if config.augmentation.use_mixup or config.augmentation.use_cutmix:
        targets1, targets2, lam = targets
        acc = lam * accuracy(outputs, targets1)[0] + (1 - lam) * accuracy(
            outputs, targets2)[0]
    elif config.augmentation.use_ricap:
        acc = sum([
            weight * accuracy(outputs, labels)[0]
            for labels, weight in zip(*targets)
        ])
    elif config.augmentation.use_dual_cutout:
        outputs1, outputs2 = outputs[:, 0], outputs[:, 1]
        acc = accuracy((outputs1 + outputs2) / 2, targets)[0]
    else:
        acc = accuracy(outputs, targets, topk)[0]
    return acc


def accuracy(outputs, targets, topk=(1, )):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1 / batch_size))
    return res
