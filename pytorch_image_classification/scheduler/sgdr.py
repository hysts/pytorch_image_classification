from pytorch_image_classification.scheduler.combined_scheduler import (
    CombinedScheduler, )
from pytorch_image_classification.scheduler.components import (
    CosineScheduler, )


class SGDRScheduler(CombinedScheduler):
    def __init__(self, steps, base_lr, T0, T_mul, lr_min_factor=1e-3):
        step_list = [T0]
        while sum(step_list) < steps:
            step_list.append(int(step_list[-1] * T_mul))
        assert sum(step_list) == steps
        schedulers = [
            CosineScheduler(step, base_lr, lr_min_factor) for step in step_list
        ]
        super().__init__(schedulers)
