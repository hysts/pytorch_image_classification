import bisect

import numpy as np


class CombinedScheduler:
    def __init__(self, schedulers=None):
        self.schedulers = []
        if schedulers is not None:
            for scheduler in schedulers:
                if scheduler is None:
                    continue
                elif hasattr(scheduler, 'schedulers'):
                    self.schedulers += scheduler.schedulers
                else:
                    self.schedulers.append(scheduler)

    def __call__(self, step):
        index = bisect.bisect_left(self._offsets, step) - 1
        index = max(0, min(index, len(self.schedulers) - 1))
        scheduler = self.schedulers[index]
        offset = self._offsets[index]
        return scheduler(step - offset)

    @property
    def _steps(self):
        return [scheduler.steps for scheduler in self.schedulers]

    @property
    def steps(self):
        return sum(self._steps)

    @property
    def _offsets(self):
        return np.cumsum(np.concatenate([[0], self._steps]))

    def multiply_steps(self, val):
        for scheduler in self.schedulers:
            scheduler.steps *= val
