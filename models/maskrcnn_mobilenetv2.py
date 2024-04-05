import torchvision
import torch.nn as nn
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class MaskrcnnMobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_pretrained,
        backbone_out_channels,
        anchor_sizes,
        anchor_aspect_ratios,
        roi_output_size,
        roi_sampling_ratio,
    ) -> None:
        super(MaskrcnnMobileNetV2, self).__init__()
        self.model = selfmodels.based_maskrcnn_mobilenet_v2(
            num_classes=num_classes,
            backbone_pretrained=backbone_pretrained,
            backbone_out_channels=backbone_out_channels,
            anchor_sizes=anchor_sizes,
            anchor_aspect_ratios=anchor_aspect_ratios,
            roi_output_size=roi_output_size,
            roi_sampling_ratio=roi_sampling_ratio,
        )

    def maskrcnn_mobilenet_v2(
        self,
        num_classes,
        backbone_pretrained,
        backbone_out_channels,
        anchor_sizes,
        anchor_aspect_ratios,
        roi_output_size,
        roi_sampling_ratio,
    ):
        mobilenetv2_backbone = torchvision.models.mobilenet_v2(
            pretrained=backbone_pretrained
        ).features
        mobilenetv2_backbone[-1] = torchvision.models.mobilenet.ConvBNReLU(
            320, backbone_out_channels
        )
        mobilenetv2_backbone.out_channels = backbone_out_channels
        anchor_generator = AnchorGenerator(
            sizes=anchor_sizes, aspect_ratios=anchor_aspect_ratios
        )
        roi_align = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0"],
            output_size=roi_output_size,
            sampling_ratio=roi_sampling_ratio,
        )
        model = MaskRCNN(
            backbone=mobilenetv2_backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_align,
        )

        return model

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def forward(self, x, targets=None):
        return self.model(x, targets)
