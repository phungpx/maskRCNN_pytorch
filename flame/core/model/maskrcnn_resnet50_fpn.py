from torch import nn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from typing import Tuple


class MaskrcnnResnet50FPN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        box_score_thresh: float,
        box_nms_thresh: float,
        anchor_sizes: Tuple[Tuple[int]] = ((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios: Tuple[Tuple[float]] = ((0.5, 1.0, 2.0),) * 5,
        pretrained: bool = False,
        pretrained_backbone: bool = False,
    ):
        super(MaskrcnnResnet50FPN, self).__init__()
        self.model = maskrcnn_resnet50_fpn(pretrained=pretrained, pretrained_backbone=pretrained_backbone)
        # anchor generator
        self.model.rpn.anchor_generator.sizes = anchor_sizes
        self.model.rpn.anchor_generator.aspect_ratios = aspect_ratios
        # roi_heads
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
        self.model.roi_heads.nms_thresh = box_nms_thresh
        self.model.roi_heads.score_thresh = box_score_thresh

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def forward(self, x, targets=None):
        return self.model(x, targets)
