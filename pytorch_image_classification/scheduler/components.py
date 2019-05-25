import numpy as np


class ConstantScheduler:
    def __init__(self, steps, lr):
        self.steps = steps
        self.lr = lr

    def __call__(self, step):
        return self.lr


class CosineScheduler:
    def __init__(self, steps, base_lr, lr_min_factor=1e-3):
        self.steps = steps
        self.base_lr = base_lr
        self.lr_min_factor = lr_min_factor

    def __call__(self, step):
        return self.base_lr * (self.lr_min_factor +
                               (1 - self.lr_min_factor) * 0.5 *
                               (1 + np.cos(step / self.steps * np.pi)))


class ExponentialScheduler:
    def __init__(self, steps, base_lr, exponent, lr_start_factor=1e-3):
        self.steps = steps
        self.base_lr = base_lr
        self.exponent = exponent
        self.lr_start_factor = lr_start_factor

    def __call__(self, step):
        return self.base_lr * (self.lr_start_factor +
                               (1 - self.lr_start_factor) *
                               (step / self.steps)**self.exponent)


class LinearScheduler:
    def __init__(self, steps, lr_start, lr_end):
        self.steps = steps
        self.lr_start = lr_start
        self.lr_end = lr_end

    def __call__(self, step):
        return self.lr_start + (self.lr_end -
                                self.lr_start) * step / self.steps
