import torch


class LARSOptimizer(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 lr,
                 momentum=0,
                 weight_decay=0,
                 eps=1e-9,
                 thresh=1e-2):

        if lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')
        if momentum < 0.0:
            raise ValueError(f'Invalid momentum value: {momentum}')
        if weight_decay < 0.0:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')

        defaults = dict(lr=lr,
                        momentum=momentum,
                        weight_decay=weight_decay,
                        eps=eps,
                        thresh=thresh)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            lr = group['lr']
            eps = group['eps']
            thresh = group['thresh']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data

                weight_norm = torch.norm(p.data)
                grad_norm = torch.norm(d_p)
                local_lr = weight_norm / (eps + grad_norm +
                                          weight_decay * weight_norm)
                local_lr = torch.where(weight_norm < thresh,
                                       torch.ones_like(local_lr), local_lr)

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.zeros_like(
                        p.data)
                else:
                    buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(lr * local_lr, d_p)
                p.data.add_(-1.0, buf)

        return loss
