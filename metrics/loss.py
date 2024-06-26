from typing import Callable
from handler.evaluator import MetricBase


class Loss(MetricBase):
    def __init__(self, loss_fn, output_transform: Callable = lambda x: x):
        super(Loss, self).__init__(output_transform)
        self._loss_fn = loss_fn

    def reset(self):
        self._sum = 0
        self._num_examples = 0

    def update(self, output):
        average_loss = self._loss_fn(*output)

        if len(average_loss.shape) != 0:
            raise ValueError("loss_fn did not return the average loss.")

        N = output[0].shape[0]
        self._sum += average_loss.item() * N
        self._num_examples += N

        return average_loss.item()

    def compute(self):
        if self._num_examples == 0:
            raise ValueError(
                "Loss must have at least one example before it can be computed."
            )
        return self._sum / self._num_examples
