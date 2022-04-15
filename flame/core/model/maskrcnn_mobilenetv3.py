import torchvision
import torch.nn as nn
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class MaskrcnnMobileNetV3(nn.Module):
    def __init__(self, num_classes, pretrained=False, pretrained_backbone=False):
        super(MaskrcnnMobileNetV3, self).__init__()
        self.model = self._maskrcnn_mobilenet_v3(num_classes, pretrained, pretrained_backbone)

    def _maskrcnn_mobilenet_v3(self, num_classes, pretrained, pretrained_backbone):
        fasterrcnn_mobilenetv3 = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
            pretrained=pretrained,
            pretrained_backbone=pretrained_backbone
        )
        mobilenet_v3 = fasterrcnn_mobilenetv3.backbone
        rpn_anchor_generator = fasterrcnn_mobilenetv3.rpn.anchor_generator
        roi_pool = fasterrcnn_mobilenetv3.roi_heads.box_roi_pool

        model = torchvision.models.detection.MaskRCNN(
            backbone=mobilenet_v3,
            min_size=320,
            max_size=640,
            num_classes=num_classes,
            rpn_anchor_generator=rpn_anchor_generator,
            box_roi_pool=roi_pool,
            mask_roi_pool=roi_pool
        )

        return model

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def forward(self, x, targets=None):
        return self.model(x, targets)
