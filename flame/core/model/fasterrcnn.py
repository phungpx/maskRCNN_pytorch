import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FasterRcnnMobilenetV3(nn.Module):
    def __init__(self, num_classes, pretrained=True, pretrained_backbone=True):
        super(FasterRcnnMobilenetV3, self).__init__()
        self.model = self.fasterrcnn_mobilenetv3(num_classes, pretrained, pretrained_backbone)

    def fasterrcnn_mobilenetv3(self, num_classes, pretrained, pretrained_backbone):
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=pretrained, pretrained_backbone=pretrained_backbone)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def forward(self, x, targets=None):
        return self.model(x, targets)


class FasterRcnnResnet50FPN(nn.Module):
    def __init__(self, num_classes, pretrained=True, pretrained_backbone=True):
        super(FasterRcnnResnet50FPN, self).__init__()
        self.model = self.fasterrcnn_resnet50_fpn(num_classes, pretrained, pretrained_backbone)

    def fasterrcnn_resnet50_fpn(self, num_classes, pretrained, pretrained_backbone):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained, pretrained_backbone=pretrained_backbone)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def forward(self, x, targets=None):
        return self.model(x, targets)
