from torch import nn
from .maskRCNN.rpn.rpn import RPNHead
from .maskRCNN.anchor import AnchorGenerator
from .maskRCNN.roi_align.poolers import MultiScaleRoIAlign
from .maskRCNN.utils._utils import _ovewrite_value_param
from .maskRCNN.utils.block_utils import FrozenBatchNorm2d
from .maskRCNN.backbones.mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights
from .maskRCNN.backbones.backbone import _mobilenet_extractor, _validate_trainable_layers

from .maskRCNN.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from .maskRCNN.mask_rcnn import MaskRCNN, MaskRCNNHeads, MaskRCNNPredictor

from typing import Optional, List, Tuple, Any


class MaskRCNNMobileNetV3LargeFPN(nn.Module):
    def __init__(
        self,
        pretrained: bool = False,
        backbone_pretrained: bool = False,
        num_classes: int = None,
        progress: bool = True,
        trainable_backbone_layers: Optional[int] = None,
        # Anchors parameters
        anchor_sizes: Tuple[Tuple[int]] = ((32, 64, 128, 256, 512,),) * 3,
        aspect_ratios: Tuple[Tuple[float]] = ((0.5, 1.0, 2.0),) * 3,
        # transform parameters
        min_size: int = 320,
        max_size: int = 640,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
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
        super(MaskRCNNMobileNetV3LargeFPN, self).__init__()
        weights = None
        weights_backbone = MobileNet_V3_Large_Weights.verify(
            MobileNet_V3_Large_Weights.IMAGENET1K_V1
        ) if backbone_pretrained else None

        if weights is not None:
            weights_backbone = None
            num_classes = _ovewrite_value_param(num_classes, len(weights.meta["categories"]))
        elif num_classes is None:
            num_classes = 91

        is_trained = (weights is not None) or (weights_backbone is not None)
        trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 6, 3)
        norm_layer = FrozenBatchNorm2d if is_trained else nn.BatchNorm2d

        backbone = mobilenet_v3_large(weights=weights_backbone, progress=progress, norm_layer=norm_layer)
        backbone = _mobilenet_extractor(backbone, True, trainable_backbone_layers)

        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        rpn_head = RPNHead(
            in_channels=backbone.out_channels,
            num_anchors=rpn_anchor_generator.num_anchors_per_location()[0],
            conv_depth=1,
        )

        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=7,
            sampling_ratio=2
        )

        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        box_head = TwoMLPHead(
            in_channels=backbone.out_channels * resolution ** 2,
            representation_size=representation_size,
        )

        box_predictor = FastRCNNPredictor(
            in_channels=representation_size,
            num_classes=num_classes,
        )

        mask_roi_pool = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=14,
            sampling_ratio=2
        )

        mask_head = MaskRCNNHeads(
            in_channels=backbone.out_channels,
            layers=[256, 256, 256, 256],
            dilation=1,
            norm_layer=nn.BatchNorm2d
        )

        mask_predictor_in_channels = 256  # == mask_layers[-1]
        mask_dim_reduced = 256
        mask_predictor = MaskRCNNPredictor(
            in_channels=mask_predictor_in_channels,
            dim_reduced=mask_dim_reduced,
            num_classes=num_classes,
        )

        self.model = MaskRCNN(
            backbone,
            # num_classes=num_classes,
            rpn_anchor_generator=rpn_anchor_generator,
            # Head
            rpn_head=rpn_head,
            # Boxes
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            # Masks
            mask_head=mask_head,
            mask_predictor=mask_predictor,
            # Transform
            min_size=min_size,
            max_size=max_size,
            image_mean=image_mean,
            image_std=image_std,
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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MaskRCNNMobileNetV3LargeFPN(
        pretrained=False,
        backbone_pretrained=True,
        num_classes=2,
        # Anchors parameters
        anchor_sizes=((32, 64, 128, 256, 512,),) * 3,
        aspect_ratios=((0.5, 1.0, 2.0),) * 3,
        # # Box parameters
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
    )
    model.eval().to(device)
    print(f'Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    image = cv2.imread('/home/phungpx/Downloads/cbimage.png')
    sample = torch.from_numpy(image).to(device)
    sample = sample.float().div(255.)
    sample = sample.permute(2, 0, 1)

    t1 = time.time()
    with torch.no_grad():
        preds = model([sample])
    t2 = time.time()

    print(t2 - t1)
    print(preds)
