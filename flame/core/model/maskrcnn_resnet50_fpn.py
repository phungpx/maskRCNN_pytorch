import torchvision
import torch.nn as nn
from flame.core.model.maskRCNN.mask_rcnn import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from flame.core.model.maskRCNN.mask_rcnn import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights


class MaskrcnnResnet50FPN(nn.Module):
    def __init__(
        self,
        version: int = 1,
        predictor_hidden_layer: int = 256,
        num_classes: int = 91,
        pretrained: bool = False,
        pretrained_backbone: bool = False,
        score_threshold: bool = 0.05,
        nms_threshold: bool = 0.5,
    ):
        super(MaskrcnnResnet50FPN, self).__init__()
        self.model = self.maskrcnn(
            num_classes,
            pretrained,
            pretrained_backbone,
            score_threshold,
            nms_threshold,
        )

    def maskrcnn(
        self,
        num_classes,
        pretrained,
        pretrained_backbone,
        score_threshold,
        nms_threshold,
    ):
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=pretrained,
            pretrained_backbone=pretrained_backbone,
            box_score_thresh=score_threshold,
            box_nms_thresh=nms_threshold,
        )

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            predictor_hidden_layer,
            num_classes,
        )

        return model

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def forward(self, x, targets=None):
        return self.model(x, targets)
