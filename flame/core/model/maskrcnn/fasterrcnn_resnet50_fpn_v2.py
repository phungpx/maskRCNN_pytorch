from torch import nn
from .maskRCNN.rpn.rpn import RPNHead
from .maskRCNN.anchor import AnchorGenerator
from .maskRCNN.utils._utils import _ovewrite_value_param
from .maskRCNN.backbones.backbone import _resnet_fpn_extractor, _validate_trainable_layers
from .maskRCNN.backbones.resnet import ResNet50_Weights, resnet50
from .maskRCNN.faster_rcnn import FasterRCNN, FastRCNNConvFCHead, FasterRCNN_ResNet50_FPN_V2_Weights

from typing import Optional, List, Tuple, Any


class FasterRCNNResNet50FPNV2(nn.Module):
    def __init__(
        self,
        pretrained: bool = False,
        backbone_pretrained: bool = False,
        num_classes: int = None,
        progress: bool = True,
        trainable_backbone_layers: Optional[int] = None,
        # Anchors parameters
        anchor_sizes: Tuple[Tuple[int]] = ((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios: Tuple[Tuple[float]] = ((0.5, 1.0, 2.0),) * 5,
        # transform parameters
        min_size: int = 800,
        max_size: int = 1333,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        # Box parameters
        box_score_thresh: float = 0.05,
        box_nms_thresh: float = 0.5,
        **kwargs: Any,
    ):
        super(FasterRCNNResNet50FPNV2, self).__init__()
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.verify(
            FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        ) if pretrained else None

        weights_backbone = ResNet50_Weights.verify(
            ResNet50_Weights.IMAGENET1K_V1
        ) if backbone_pretrained else None

        if weights is not None:
            weights_backbone = None
            num_classes = _ovewrite_value_param(num_classes, len(weights.meta["categories"]))
        elif num_classes is None:
            num_classes = 91

        is_trained = weights is not None or weights_backbone is not None
        trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)

        backbone = resnet50(weights=weights_backbone, progress=progress)
        backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers, norm_layer=nn.BatchNorm2d)

        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        rpn_head = RPNHead(
            in_channels=backbone.out_channels,
            num_anchors=rpn_anchor_generator.num_anchors_per_location()[0],
            conv_depth=2,
        )

        box_head = FastRCNNConvFCHead(
            input_size=(backbone.out_channels, 7, 7),
            conv_layers=[256, 256, 256, 256],
            fc_layers=[1024],
            norm_layer=nn.BatchNorm2d,
        )

        self.model = FasterRCNN(
            backbone,
            num_classes=num_classes,
            rpn_anchor_generator=rpn_anchor_generator,
            rpn_head=rpn_head,
            box_head=box_head,
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
    model = FasterRCNNResNet50FPN(
        pretrained=False,
        backbone_pretrained=True,
        num_classes=2,
        # Anchors parameters
        anchor_sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.25, 0.5, 1.0, 2.0, 4.0),) * 5,
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
