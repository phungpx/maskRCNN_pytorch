from torch import nn
from models.based_maskrcnn.anchor import AnchorGenerator
from models.based_maskrcnn.faster_rcnn import FasterRCNN
from models.based_maskrcnn.utils.block_utils import FrozenBatchNorm2d
from models.based_maskrcnn.utils._utils import _ovewrite_value_param
from models.based_maskrcnn.backbones.mobilenetv3 import (
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
)
from models.based_maskrcnn.backbones.backbone import (
    _mobilenet_extractor,
    _validate_trainable_layers,
)
from models.based_maskrcnn.faster_rcnn import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)

from typing import Optional, Tuple, Any


class FasterRCNNMobileNetV3LargeFPN(nn.Module):
    def __init__(
        self,
        pretrained: bool = False,
        backbone_pretrained: bool = False,
        num_classes: int = None,
        progress: bool = True,
        trainable_backbone_layers: Optional[int] = None,
        # Anchors parameters
        anchor_sizes: Tuple[Tuple[int]] = (
            (
                32,
                64,
                128,
                256,
                512,
            ),
        )
        * 3,
        aspect_ratios: Tuple[Tuple[float]] = ((0.5, 1.0, 2.0),) * 3,
        # transform parameters
        min_size: int = 320,
        max_size: int = 640,
        # RPN parameters
        rpn_pre_nms_top_n_test: int = 150,
        rpn_post_nms_top_n_test: int = 150,
        rpn_nms_thresh: float = 0.7,
        rpn_score_thresh: float = 0.05,
        # Box parameters
        box_score_thresh: float = 0.05,
        box_nms_thresh: float = 0.5,
        **kwargs: Any,
    ):
        super(FasterRCNNMobileNetV3LargeFPN, self).__init__()

        weights = (
            FasterRCNN_MobileNet_V3_Large_FPN_Weights.verify(
                FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
            )
            if pretrained
            else None
        )

        weights_backbone = (
            MobileNet_V3_Large_Weights.verify(MobileNet_V3_Large_Weights.IMAGENET1K_V1)
            if backbone_pretrained
            else None
        )

        if weights is not None:
            weights_backbone = None
            num_classes = _ovewrite_value_param(
                num_classes, len(weights.meta["categories"])
            )
        elif num_classes is None:
            num_classes = 91

        is_trained = (weights is not None) or (weights_backbone is not None)
        trainable_backbone_layers = _validate_trainable_layers(
            is_trained, trainable_backbone_layers, 6, 3
        )
        norm_layer = FrozenBatchNorm2d if is_trained else nn.BatchNorm2d

        backbone = mobilenet_v3_large(
            weights=weights_backbone, progress=progress, norm_layer=norm_layer
        )
        backbone = _mobilenet_extractor(backbone, True, trainable_backbone_layers)

        self.model = FasterRCNN(
            backbone,
            num_classes,
            rpn_anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios),
            min_size=min_size,
            max_size=max_size,
            # RPN parameters
            rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
            rpn_nms_thresh=rpn_nms_thresh,
            rpn_score_thresh=rpn_score_thresh,
            # Box parameters
            box_score_thresh=box_score_thresh,
            box_nms_thresh=box_nms_thresh,
            **kwargs,
        )

        if weights is not None:
            self.model.load_state_dict(weights.get_state_dict(progress=progress))

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def forward(self, x, targets=None):
        return self.model(x, targets)


if __name__ == "__main__":
    import cv2
    import time
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FasterRCNNMobileNetV3LargeFPN(
        # weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1,
        # weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1,
        # num_classes=2,
        # Anchors parameters
        anchor_sizes=(
            (
                32,
                64,
                128,
                256,
                512,
            ),
        )
        * 3,
        aspect_ratios=((0.25, 0.5, 1.0, 2.0, 4.0),) * 3,
        # Box parameters
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
    )
    model.eval().to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    image = cv2.imread("/home/phungpx/Downloads/dog.jpg")
    sample = torch.from_numpy(image).to(device)
    sample = sample.float().div(255.0)
    sample = sample.permute(2, 0, 1)

    t1 = time.time()
    with torch.no_grad():
        preds = model([sample])
    t2 = time.time()

    print(t2 - t1)
    # print(preds)
