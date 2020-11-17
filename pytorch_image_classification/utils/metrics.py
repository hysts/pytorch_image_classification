import torch


def compute_accuracy(config, outputs, targets, augmentation, topk=(1, )):
    if augmentation:
        if config.augmentation.use_mixup or config.augmentation.use_cutmix:
            targets1, targets2, lam = targets
            accs1 = accuracy(outputs, targets1, topk)
            accs2 = accuracy(outputs, targets2, topk)
            accs = tuple([
                lam * acc1 + (1 - lam) * acc2
                for acc1, acc2 in zip(accs1, accs2)
            ])
        elif config.augmentation.use_ricap:
            weights = []
            accs_all = []
            for labels, weight in zip(*targets):
                weights.append(weight)
                accs_all.append(accuracy(outputs, labels, topk))
            accs = []
            for i in range(len(accs_all[0])):
                acc = 0
                for weight, accs_list in zip(weights, accs_all):
                    acc += weight * accs_list[i]
                accs.append(acc)
            accs = tuple(accs)
        elif config.augmentation.use_dual_cutout:
            outputs1, outputs2 = outputs[:, 0], outputs[:, 1]
            accs = accuracy((outputs1 + outputs2) / 2, targets, topk)
        else:
            accs = accuracy(outputs, targets, topk)
    else:
        accs = accuracy(outputs, targets, topk)
    return accs


def accuracy(outputs, targets, topk=(1, )):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1 / batch_size))
    return res
