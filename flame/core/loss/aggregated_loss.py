import torch
from torch import nn, Tensor
from typing import List, Dict, Callable

from . import loss


class AggregatedLoss(loss.LossBase):
    def __init__(
        self,
        verbose: bool = False,
        box_cls_weight: float = 1., box_reg_weight: float = 1.,
        rpn_cls_weight: float = 1., rpn_reg_weight: float = 1.,
        mask_weight: float = 1.,
        output_transform: Callable = lambda x: x,
    ) -> None:
        super(AggregatedLoss, self).__init__(output_transform)
        self.verbose = verbose
        self.box_cls_weight = box_cls_weight
        self.box_reg_weight = box_reg_weight
        self.rpn_cls_weight = rpn_cls_weight
        self.rpn_reg_weight = rpn_reg_weight
        self.mask_weight = mask_weight

    def forward(self, losses: Dict[str, Tensor]) -> Tensor:
        loss_classifier = losses['loss_classifier']
        loss_box_reg = losses['loss_box_reg']
        loss_objectness = losses['loss_objectness']
        loss_rpn_box_reg = losses['loss_rpn_box_reg']
        loss_mask = losses.get('loss_mask', 0.)

        loss: Tensor = self.box_cls_weight * loss_classifier\
            + self.box_reg_weight * loss_box_reg\
            + self.rpn_cls_weight * loss_objectness\
            + self.rpn_reg_weight *  loss_rpn_box_reg\
            + self.mask_weight * loss_mask\

        if self.verbose:
            from prettytable import PrettyTable
            verbose = PrettyTable(losses.keys())  # heading of table
            verbose.add_row([loss.item() for loss in losses.values()])
            print(verbose)

        return loss
