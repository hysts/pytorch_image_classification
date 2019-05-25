from pytorch_image_classification.scheduler.combined_scheduler import (
    CombinedScheduler, )
from pytorch_image_classification.scheduler.components import (
    ConstantScheduler, )


class MultistepScheduler(CombinedScheduler):
    def __init__(self, steps, base_lr, gamma, milestones):
        lrs = [base_lr * gamma**index for index in range(len(milestones) + 1)]
        step_list = [
            step1 - step0
            for step0, step1 in zip([0] + milestones, milestones + [steps])
        ]
        schedulers = [
            ConstantScheduler(step, lr) for step, lr in zip(step_list, lrs)
        ]
        super().__init__(schedulers)
