import torchvision
import torch.nn as nn
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class MaskrcnnResnet50FPN(nn.Module):
    def __init__(
        self,
        num_classes,
        pretrained=False,
        pretrained_backbone=False,
        score_threshold=0.05,
        nms_threshold=0.5,
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
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )

        return model

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def forward(self, x, targets=None):
        return self.model(x, targets)


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


class MaskrcnnMobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_pretrained,
        backbone_out_channels,
        anchor_sizes,
        anchor_aspect_ratios,
        roi_output_size,
        roi_sampling_ratio
    ) -> None:
        super(MaskrcnnMobileNetV2, self).__init__()
        self.model = self.maskrcnn_mobilenet_v2(
            num_classes=num_classes,
            backbone_pretrained=backbone_pretrained,
            backbone_out_channels=backbone_out_channels,
            anchor_sizes=anchor_sizes,
            anchor_aspect_ratios=anchor_aspect_ratios,
            roi_output_size=roi_output_size,
            roi_sampling_ratio=roi_sampling_ratio
        )

    def maskrcnn_mobilenet_v2(
        self,
        num_classes,
        backbone_pretrained,
        backbone_out_channels,
        anchor_sizes,
        anchor_aspect_ratios,
        roi_output_size,
        roi_sampling_ratio
    ):
        mobilenetv2_backbone = torchvision.models.mobilenet_v2(pretrained=backbone_pretrained).features
        mobilenetv2_backbone[-1] = torchvision.models.mobilenet.ConvBNReLU(320, backbone_out_channels)
        mobilenetv2_backbone.out_channels = backbone_out_channels
        anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=anchor_aspect_ratios)
        roi_align = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=roi_output_size,
            sampling_ratio=roi_sampling_ratio
        )
        model = MaskRCNN(
            backbone=mobilenetv2_backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_align
        )

        return model

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def forward(self, x, targets=None):
        return self.model(x, targets)
